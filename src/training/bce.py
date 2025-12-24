"""
BCE学習モジュール

Binary Cross Entropyを用いて、感情が近い単語が近傍に配置されるように学習する。
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from tqdm import tqdm


class EmotionPairDataset(Dataset):
    """感情ペアデータセット"""

    def __init__(
        self,
        positive_pairs: List[Tuple[str, str]],
        negative_pairs: List[Tuple[str, str]],
        embedding_model
    ):
        """
        Args:
            positive_pairs: 感情が近い単語ペア
            negative_pairs: 感情が遠い単語ペア
            embedding_model: 埋め込みモデル
        """
        self.pairs = []
        self.labels = []

        # Positiveペアを追加
        for word1, word2 in positive_pairs:
            self.pairs.append((word1, word2))
            self.labels.append(1.0)

        # Negativeペアを追加
        for word1, word2 in negative_pairs:
            self.pairs.append((word1, word2))
            self.labels.append(0.0)

        self.embedding_model = embedding_model

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        word1, word2 = self.pairs[idx]
        label = self.labels[idx]

        # テキストに変換
        text1 = self.embedding_model.create_emotion_text(word1)
        text2 = self.embedding_model.create_emotion_text(word2)

        return text1, text2, label


class BCETrainer:
    """BCE学習トレーナー"""

    def __init__(
        self,
        base_model,
        learning_rate: float = 2e-5,
        device: str = None,
        temperature: float = 20.0
    ):
        """
        Args:
            base_model: ベース埋め込みモデル（SentenceTransformer）
            learning_rate: 学習率
            device: デバイス
            temperature: 温度パラメータ（α）
        """
        if device is None:
            device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.temperature = temperature

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

        # 損失関数
        self.criterion = nn.BCEWithLogitsLoss()

    def compute_similarity(self, embeddings1, embeddings2):
        """
        コサイン類似度を計算してスケーリング

        Args:
            embeddings1: バッチ1の埋め込み
            embeddings2: バッチ2の埋め込み

        Returns:
            スケーリングされた類似度
        """
        # コサイン類似度
        cosine_sim = torch.sum(embeddings1 * embeddings2, dim=1)

        # 温度パラメータでスケーリング
        similarity = self.temperature * cosine_sim

        return similarity

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
            texts1, texts2, labels = batch

            # 埋め込みを取得
            embeddings1 = self.model(texts1)
            embeddings2 = self.model(texts2)

            # 類似度を計算
            similarities = self.compute_similarity(embeddings1, embeddings2)

            # ラベルをテンソルに変換
            labels = torch.tensor(labels, dtype=torch.float32).to(self.device)

            # 損失を計算
            loss = self.criterion(similarities, labels)

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

        print(f"Starting BCE training for {epochs} epochs...")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"  - Temperature: {self.temperature}")

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


if __name__ == "__main__":
    # テスト
    print("=== BCE学習モジュールのテスト ===")

    # ダミーデータでテスト
    from sentence_transformers import SentenceTransformer

    # ベースモデルをロード
    base_model = SentenceTransformer("BAAI/bge-m3")

    # ダミーデータを作成
    positive_pairs = [("喜び", "楽しさ"), ("悲しみ", "寂しさ")]
    negative_pairs = [("喜び", "悲しみ"), ("楽しさ", "怒り")]

    # データセットを作成
    from src.embedding import EmbeddingModel
    embedding_model = EmbeddingModel()

    dataset = EmotionPairDataset(
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
        embedding_model=embedding_model
    )

    print(f"Dataset size: {len(dataset)}")

    # トレーナーを作成
    trainer = BCETrainer(
        base_model=base_model,
        learning_rate=2e-5
    )

    # 学習（1エポックのみ）
    trainer.train(
        train_dataset=dataset,
        epochs=1,
        batch_size=2
    )

    print("BCE training test completed.")
