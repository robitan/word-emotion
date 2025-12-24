"""
MLP感情分類モデル

BCE と Triplet で共通のMLPヘッドを使用して、
損失関数の違いのみで比較する。
"""

import torch
import torch.nn as nn
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmotionMLPHead(nn.Module):
    """
    感情分類用MLPヘッド

    構造: Linear(1024 → h) → ReLU → Linear(h → K) → Sigmoid
    - 入力: bge-m3の埋め込み (1024次元)
    - 出力: 感情ラベル空間 (K次元)
    """

    def __init__(
        self,
        base_model: SentenceTransformer,
        num_emotions: int,
        hidden_dim: int = 256,
        freeze_base: bool = True
    ):
        """
        Args:
            base_model: ベースとなるSentenceTransformerモデル
            num_emotions: 感情カテゴリ数 (K)
            hidden_dim: 中間層の次元数 (h)
            freeze_base: ベースモデルをフリーズするか
        """
        super().__init__()

        self.base_model = base_model
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim

        # ベースモデルをフリーズ
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # MLPヘッド
        base_dim = base_model.get_sentence_embedding_dimension()

        self.mlp = nn.Sequential(
            nn.Linear(base_dim, hidden_dim),  # Linear1
            nn.ReLU(),                         # ReLU
            nn.Linear(hidden_dim, num_emotions)  # Linear2 (logits)
        )

        # Sigmoid（推論・可視化用、学習時はBCEWithLogitsLossが使うので不要）
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        texts: List[str],
        return_probs: bool = False
    ) -> torch.Tensor:
        """
        順伝播

        Args:
            texts: テキストのリスト
            return_probs: 確率を返すか（Sigmoidを適用）

        Returns:
            logits (default) または probs (return_probs=True)
        """
        # ベースモデルで埋め込みを取得
        with torch.no_grad():
            base_embeddings = self.base_model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True  # 正規化
            )

        # inference_modeで作成されたテンソルをcloneして通常のテンソルに変換
        base_embeddings = base_embeddings.detach().clone()

        # MLPヘッドを通す
        logits = self.mlp(base_embeddings)

        if return_probs:
            return self.sigmoid(logits)
        else:
            return logits

    def get_emotion_embedding(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        感情空間の埋め込みを取得（K次元）

        Args:
            texts: テキストのリスト
            normalize: 正規化するか

        Returns:
            感情埋め込みベクトル (shape: [batch_size, K])
        """
        logits = self.forward(texts, return_probs=False)

        if normalize:
            # L2正規化
            embeddings = torch.nn.functional.normalize(logits, p=2, dim=1)
        else:
            embeddings = logits

        return embeddings

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        テキストを感情埋め込みに変換（推論用）

        Args:
            texts: テキストのリスト
            batch_size: バッチサイズ
            normalize: 正規化するか

        Returns:
            感情埋め込みベクトル (numpy array)
        """
        if isinstance(texts, str):
            texts = [texts]

        self.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                embeddings = self.get_emotion_embedding(
                    batch_texts,
                    normalize=normalize
                )
                all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


if __name__ == "__main__":
    # テスト
    print("=== EmotionMLPHead のテスト ===")

    # ベースモデルをロード
    from sentence_transformers import SentenceTransformer
    base_model = SentenceTransformer("BAAI/bge-m3")

    # MLPヘッドを作成
    num_emotions = 20  # 感情カテゴリ数
    model = EmotionMLPHead(
        base_model=base_model,
        num_emotions=num_emotions,
        hidden_dim=256
    )

    # テスト
    texts = ["喜び という単語が示す感情", "悲しみ という単語が示す感情"]

    # Logitsを取得
    logits = model(texts, return_probs=False)
    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits: {logits}")

    # 確率を取得
    probs = model(texts, return_probs=True)
    print(f"\nProbs shape: {probs.shape}")
    print(f"Probs: {probs}")

    # 感情埋め込みを取得
    embeddings = model.get_emotion_embedding(texts, normalize=True)
    print(f"\nEmotion embeddings shape: {embeddings.shape}")
    print(f"L2 norm: {torch.norm(embeddings, dim=1)}")

    print("\nテスト完了")
