"""
Triplet学習モジュール

Triplet Lossを用いて、感情が近い単語が近傍に配置され、
遠い単語が離れるように学習する。
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from tqdm import tqdm


class TripletDataset(Dataset):
    """Tripletデータセット"""

    def __init__(
        self,
        triplets: List[Tuple[str, str, str]],
        embedding_model
    ):
        """
        Args:
            triplets: (anchor, positive, negative)のリスト
            embedding_model: 埋め込みモデル
        """
        self.triplets = triplets
        self.embedding_model = embedding_model

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]

        # テキストに変換
        anchor_text = self.embedding_model.create_emotion_text(anchor)
        positive_text = self.embedding_model.create_emotion_text(positive)
        negative_text = self.embedding_model.create_emotion_text(negative)

        return anchor_text, positive_text, negative_text


class TripletTrainer:
    """Triplet学習トレーナー"""

    def __init__(
        self,
        base_model,
        learning_rate: float = 2e-5,
        margin: float = 0.3,
        device: str = None
    ):
        """
        Args:
            base_model: ベース埋め込みモデル（SentenceTransformer）
            learning_rate: 学習率
            margin: マージン
            device: デバイス
        """
        if device is None:
            device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.margin = margin

        # ファインチューニング可能なモデルを作成
        from src.embedding import FineTunableEmbedding
        self.model = FineTunableEmbedding(
            base_model=base_model,
            output_dim=1024,
            freeze_base=True
        ).to(device)

        # オプティマイザ
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        # 損失関数（Triplet Margin Loss）
        self.criterion = nn.TripletMarginLoss(
            margin=margin,
            p=2,  # L2距離
            reduction='mean'
        )

    def compute_distance(self, embeddings1, embeddings2):
        """
        コサイン距離を計算

        Args:
            embeddings1: 埋め込み1
            embeddings2: 埋め込み2

        Returns:
            コサイン距離（1 - cosine_similarity）
        """
        # コサイン類似度
        cosine_sim = torch.sum(embeddings1 * embeddings2, dim=1)

        # コサイン距離に変換
        distance = 1.0 - cosine_sim

        return distance

    def train_epoch(self, dataloader):
        """
        1エポックの学習

        Args:
            dataloader: データローダー

        Returns:
            平均損失
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training"):
            anchor_texts, positive_texts, negative_texts = batch

            # 埋め込みを取得
            anchor_embeddings = self.model(anchor_texts)
            positive_embeddings = self.model(positive_texts)
            negative_embeddings = self.model(negative_texts)

            # Triplet Lossを計算
            # PyTorchのTripletMarginLossは内部でL2距離を計算するので、
            # 正規化された埋め込みを直接渡せばOK
            loss = self.criterion(
                anchor_embeddings,
                positive_embeddings,
                negative_embeddings
            )

            # 逆伝播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def train(
        self,
        train_dataset,
        epochs: int = 10,
        batch_size: int = 32,
        save_path: str = None
    ):
        """
        学習を実行

        Args:
            train_dataset: 学習データセット
            epochs: エポック数
            batch_size: バッチサイズ
            save_path: モデルの保存パス

        Returns:
            学習済みモデル
        """
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        print(f"Starting Triplet training for {epochs} epochs...")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"  - Margin: {self.margin}")

        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # モデルを保存
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        return self.model

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        学習済みモデルでテキストを埋め込み

        Args:
            texts: テキストのリスト
            batch_size: バッチサイズ

        Returns:
            埋め込みベクトル
        """
        return self.model.encode(texts, batch_size=batch_size)


def compute_emotion_vector(
    word_embeddings: dict,
    word_emotions: dict,
    emotion_symbol: str,
    method: str = "mean_diff"
) -> np.ndarray:
    """
    感情変化ベクトルを計算

    Args:
        word_embeddings: 単語 -> 埋め込みベクトル の辞書
        word_emotions: 単語 -> 感情シンボルのセット の辞書
        emotion_symbol: 対象の感情シンボル
        method: 計算方法（"mean_diff" or "prototype"）

    Returns:
        感情変化ベクトル
    """
    if method == "mean_diff":
        # 案1: 平均との差分
        positive_words = [
            word for word, emotions in word_emotions.items()
            if emotion_symbol in emotions and word in word_embeddings
        ]

        negative_words = [
            word for word, emotions in word_emotions.items()
            if emotion_symbol not in emotions and word in word_embeddings
        ]

        if not positive_words or not negative_words:
            return np.zeros(1024)

        # 平均ベクトルを計算
        positive_mean = np.mean(
            [word_embeddings[w] for w in positive_words],
            axis=0
        )

        negative_mean = np.mean(
            [word_embeddings[w] for w in negative_words],
            axis=0
        )

        # 差分を計算
        emotion_vector = positive_mean - negative_mean

    else:
        raise ValueError(f"Unknown method: {method}")

    # 正規化
    emotion_vector = emotion_vector / np.linalg.norm(emotion_vector)

    return emotion_vector


if __name__ == "__main__":
    # テスト
    print("=== Triplet学習モジュールのテスト ===")

    # ダミーデータでテスト
    from sentence_transformers import SentenceTransformer

    # ベースモデルをロード
    base_model = SentenceTransformer("BAAI/bge-m3")

    # ダミーデータを作成
    triplets = [
        ("喜び", "楽しさ", "悲しみ"),
        ("楽しさ", "喜び", "怒り")
    ]

    # データセットを作成
    from src.embedding import EmbeddingModel
    embedding_model = EmbeddingModel()

    dataset = TripletDataset(
        triplets=triplets,
        embedding_model=embedding_model
    )

    print(f"Dataset size: {len(dataset)}")

    # トレーナーを作成
    trainer = TripletTrainer(
        base_model=base_model,
        learning_rate=2e-5,
        margin=0.3
    )

    # 学習（1エポックのみ）
    trainer.train(
        train_dataset=dataset,
        epochs=1,
        batch_size=2
    )

    print("Triplet training test completed.")
