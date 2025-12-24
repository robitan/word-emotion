"""
Triplet-MLP学習モジュール

感情ラベル空間（K次元）でTriplet Lossを使用して学習する。
BCE-MLPと同一のMLPヘッドを使用し、損失関数のみを変更。
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from tqdm import tqdm


class TripletMLPDataset(Dataset):
    """Triplet-MLPデータセット"""

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


class TripletMLPTrainer:
    """Triplet-MLP学習トレーナー"""

    def __init__(
        self,
        model,
        learning_rate: float = 2e-5,
        margin: float = 0.3,
        device: str = None
    ):
        """
        Args:
            model: EmotionMLPHeadモデル
            learning_rate: 学習率
            margin: マージン
            device: デバイス
        """
        if device is None:
            device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.margin = margin
        self.model = model.to(device)

        # オプティマイザ（MLPヘッドのみを学習）
        self.optimizer = torch.optim.AdamW(
            self.model.mlp.parameters(),  # MLPヘッドのパラメータのみ
            lr=learning_rate
        )

        # 損失関数（Triplet Margin Loss）
        # コサイン距離を使用するため、distance_function を指定
        self.criterion = nn.TripletMarginLoss(
            margin=margin,
            p=2,  # L2距離（正規化後なのでコサイン距離と等価）
            reduction='mean'
        )

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

            # 感情埋め込みを取得（正規化済み）
            anchor_embeddings = self.model.get_emotion_embedding(
                anchor_texts,
                normalize=True
            )
            positive_embeddings = self.model.get_emotion_embedding(
                positive_texts,
                normalize=True
            )
            negative_embeddings = self.model.get_emotion_embedding(
                negative_texts,
                normalize=True
            )

            # Triplet Lossを計算
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

        print(f"Starting Triplet-MLP training for {epochs} epochs...")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"  - Margin: {self.margin}")
        print(f"  - Num emotions: {self.model.num_emotions}")
        print(f"  - Hidden dim: {self.model.hidden_dim}")

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
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        学習済みモデルでテキストを埋め込み

        Args:
            texts: テキストのリスト
            batch_size: バッチサイズ
            normalize: 正規化するか

        Returns:
            埋め込みベクトル
        """
        return self.model.encode(texts, batch_size=batch_size, normalize=normalize)


if __name__ == "__main__":
    # テスト
    print("=== Triplet-MLP学習モジュールのテスト ===")

    from sentence_transformers import SentenceTransformer
    from src.embedding import EmbeddingModel
    from src.training.mlp_model import EmotionMLPHead

    # ベースモデルをロード
    base_model = SentenceTransformer("BAAI/bge-m3")

    # MLPヘッドを作成
    num_emotions = 5  # テスト用
    model = EmotionMLPHead(
        base_model=base_model,
        num_emotions=num_emotions,
        hidden_dim=128
    )

    # ダミーデータを作成
    triplets = [
        ("喜び", "楽しさ", "悲しみ"),
        ("楽しさ", "喜び", "怒り")
    ]

    embedding_model = EmbeddingModel()

    dataset = TripletMLPDataset(
        triplets=triplets,
        embedding_model=embedding_model
    )

    print(f"Dataset size: {len(dataset)}")

    # トレーナーを作成
    trainer = TripletMLPTrainer(
        model=model,
        learning_rate=2e-5,
        margin=0.3
    )

    # 学習（1エポックのみ）
    trainer.train(
        train_dataset=dataset,
        epochs=1,
        batch_size=2
    )

    print("Triplet-MLP training test completed.")
