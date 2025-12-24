"""
埋め込み生成モジュール

BAAI/bge-m3を使用して、テキストを1024次元のベクトルに変換する。
"""

import os
import torch
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingModel:
    """埋め込みモデルクラス"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = None,
        normalize: bool = True
    ):
        """
        Args:
            model_name: モデル名
            device: デバイス（cuda/cpu）。Noneの場合は自動選択
            normalize: L2正規化を適用するか
        """
        if device is None:
            device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.normalize = normalize
        self.model_name = model_name

        print(f"Loading model: {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        print("Model loaded successfully.")

    def create_emotion_text(self, word: str) -> str:
        """
        単語から感情意味に寄せた入力テキストを生成

        Args:
            word: 単語

        Returns:
            入力テキスト
        """
        return f"{word} という単語が示す感情"

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        テキストを埋め込みベクトルに変換

        Args:
            texts: テキストまたはテキストのリスト
            batch_size: バッチサイズ
            show_progress: プログレスバーを表示するか

        Returns:
            埋め込みベクトル（shape: [n, 1024]）
        """
        if isinstance(texts, str):
            texts = [texts]

        # バッチ処理で埋め込みを生成
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )

        return embeddings

    def encode_words(
        self,
        words: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        単語リストを感情意味テキストに変換してから埋め込み

        Args:
            words: 単語のリスト
            batch_size: バッチサイズ
            show_progress: プログレスバーを表示するか

        Returns:
            埋め込みベクトル（shape: [n, 1024]）
        """
        # 単語から感情意味テキストを生成
        texts = [self.create_emotion_text(word) for word in words]

        # 埋め込みを生成
        embeddings = self.encode(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )

        return embeddings

    def encode_word_dict(
        self,
        word_dict: dict,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> dict:
        """
        単語辞書を埋め込み辞書に変換

        Args:
            word_dict: 単語をキーとする辞書
            batch_size: バッチサイズ
            show_progress: プログレスバーを表示するか

        Returns:
            単語 -> 埋め込みベクトル の辞書
        """
        words = list(word_dict.keys())
        embeddings = self.encode_words(
            words,
            batch_size=batch_size,
            show_progress=show_progress
        )

        # 辞書形式に変換
        embedding_dict = {
            word: embeddings[i]
            for i, word in enumerate(words)
        }

        return embedding_dict

    @property
    def embedding_dim(self) -> int:
        """埋め込み次元数を取得"""
        return self.model.get_sentence_embedding_dimension()


class FineTunableEmbedding(torch.nn.Module):
    """
    ファインチューニング可能な埋め込みモデル

    既存のbge-m3モデルの上に追加の変換層を載せる
    """

    def __init__(
        self,
        base_model: SentenceTransformer,
        output_dim: int = 1024,
        freeze_base: bool = True
    ):
        """
        Args:
            base_model: ベースとなるSentenceTransformerモデル
            output_dim: 出力次元数
            freeze_base: ベースモデルをフリーズするか
        """
        super().__init__()

        self.base_model = base_model
        self.output_dim = output_dim

        # ベースモデルをフリーズ
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # 変換層を追加
        base_dim = base_model.get_sentence_embedding_dimension()
        self.projection = torch.nn.Linear(base_dim, output_dim)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        順伝播

        Args:
            texts: テキストのリスト

        Returns:
            埋め込みベクトル（shape: [batch_size, output_dim]）
        """
        # ベースモデルで埋め込みを取得
        with torch.no_grad():
            base_embeddings = self.base_model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=False
            )

        # 変換層を通す
        embeddings = self.projection(base_embeddings)

        # L2正規化
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        テキストを埋め込みベクトルに変換（推論用）

        Args:
            texts: テキストまたはテキストのリスト
            batch_size: バッチサイズ

        Returns:
            埋め込みベクトル
        """
        if isinstance(texts, str):
            texts = [texts]

        self.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                embeddings = self.forward(batch_texts)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


if __name__ == "__main__":
    # テスト
    print("=== 埋め込みモデルのテスト ===")

    # モデルの初期化
    model = EmbeddingModel()

    # 単一テキストのテスト
    text = "喜び という単語が示す感情"
    embedding = model.encode(text, show_progress=False)
    print(f"\nテキスト: {text}")
    print(f"埋め込み形状: {embedding.shape}")
    print(f"埋め込み次元: {model.embedding_dim}")
    print(f"L2ノルム: {np.linalg.norm(embedding):.4f}")

    # 複数単語のテスト
    words = ["喜び", "悲しみ", "怒り", "楽しさ"]
    embeddings = model.encode_words(words, show_progress=False)
    print(f"\n単語数: {len(words)}")
    print(f"埋め込み形状: {embeddings.shape}")

    # コサイン類似度の計算
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings)
    print(f"\nコサイン類似度行列:")
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i < j:
                print(f"  {word1} - {word2}: {similarities[i, j]:.4f}")
