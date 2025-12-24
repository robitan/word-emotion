"""
K-MLP-BCE学習モジュール

感情ラベル空間（K次元）へのマルチラベル分類として学習する。
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from tqdm import tqdm


class EmotionLabelDataset(Dataset):
    """感情ラベルデータセット"""

    def __init__(
        self,
        word_emotions: Dict[str, set],
        emotion_map: Dict[str, str],
        embedding_model
    ):
        """
        Args:
            word_emotions: 単語 -> 感情シンボルのセット の辞書
            emotion_map: 感情シンボル -> 感情名 の辞書
            embedding_model: 埋め込みモデル
        """
        self.word_emotions = word_emotions
        self.emotion_symbols = sorted(emotion_map.keys())  # 感情シンボルを順序付け
        self.emotion_to_idx = {e: i for i,
                               e in enumerate(self.emotion_symbols)}
        self.embedding_model = embedding_model

        # 感情が付与されている単語のみを使用
        self.words = [
            word for word, emotions in word_emotions.items()
            if emotions  # 空でない
        ]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        emotions = self.word_emotions[word]

        # テキストに変換
        text = self.embedding_model.create_emotion_text(word)

        # マルチラベルベクトルを作成
        label = torch.zeros(len(self.emotion_symbols), dtype=torch.float32)
        for emotion_symbol in emotions:
            if emotion_symbol in self.emotion_to_idx:
                label[self.emotion_to_idx[emotion_symbol]] = 1.0

        return text, label


class BCEMLPTrainer:
    """K-MLP-BCE学習トレーナー"""

    def __init__(
        self,
        model,
        learning_rate: float = 2e-5,
        device: str = None
    ):
        """
        Args:
            model: EmotionMLPHeadモデル
            learning_rate: 学習率
            device: デバイス
        """
        if device is None:
            device = os.getenv(
                "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = model.to(device)

        # オプティマイザ（MLPヘッドのみを学習）
        self.optimizer = torch.optim.AdamW(
            self.model.mlp.parameters(),  # MLPヘッドのパラメータのみ
            lr=learning_rate
        )

        # 損失関数（BCEWithLogitsLoss）
        self.criterion = nn.BCEWithLogitsLoss()

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
            texts, labels = batch
            labels = labels.to(self.device)

            # 順伝播
            logits = self.model(texts, return_probs=False)

            # 損失を計算
            loss = self.criterion(logits, labels)

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

        print(f"Starting K-MLP-BCE training for {epochs} epochs...")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {self.optimizer.param_groups[0]['lr']}")
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
    print("=== K-MLP-BCE学習モジュールのテスト ===")

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
    word_emotions = {
        "喜び": {"喜", "楽"},
        "悲しみ": {"悲", "寂"},
        "怒り": {"怒"}
    }

    emotion_map = {
        "喜": "喜び",
        "楽": "楽しさ",
        "悲": "悲しみ",
        "寂": "寂しさ",
        "怒": "怒り"
    }

    embedding_model = EmbeddingModel()

    dataset = EmotionLabelDataset(
        word_emotions=word_emotions,
        emotion_map=emotion_map,
        embedding_model=embedding_model
    )

    print(f"Dataset size: {len(dataset)}")

    # トレーナーを作成
    trainer = BCEMLPTrainer(
        model=model,
        learning_rate=2e-5
    )

    # 学習（1エポックのみ）
    trainer.train(
        train_dataset=dataset,
        epochs=1,
        batch_size=2
    )

    print("K-MLP-BCE training test completed.")
