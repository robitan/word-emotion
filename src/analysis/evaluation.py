"""
CAとベクトル空間の整合性評価モジュール

コレスポンデンス分析（CA）の結果とベクトル空間の構造を比較し、
整合性を評価する。
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.stats import spearmanr


class StructureEvaluator:
    """構造整合性評価クラス"""

    def __init__(
        self,
        ca_word_coords: pd.DataFrame,
        vector_words: List[str],
        vector_embeddings: np.ndarray
    ):
        """
        Args:
            ca_word_coords: CA分析による単語の座標
            vector_words: ベクトル空間の単語リスト
            vector_embeddings: ベクトル空間の埋め込み（正規化済み）
        """
        # 共通の単語のみを使用
        common_words = list(set(ca_word_coords.index) & set(vector_words))
        common_words.sort()

        self.common_words = common_words

        # CA座標を抽出（共通単語のみ）
        self.ca_coords = ca_word_coords.loc[common_words]

        # ベクトル空間の埋め込みを抽出（共通単語のみ）
        vector_dict = {word: vector_embeddings[i] for i, word in enumerate(vector_words)}
        self.vector_embeddings = np.array([vector_dict[word] for word in common_words])

        print(f"Evaluating structure consistency for {len(common_words)} common words")

    def compute_ca_distances(self) -> np.ndarray:
        """
        CA空間での距離行列を計算

        Returns:
            距離行列
        """
        distances = euclidean_distances(self.ca_coords.values)
        return distances

    def compute_vector_distances(self) -> np.ndarray:
        """
        ベクトル空間での距離行列を計算（コサイン距離）

        Returns:
            距離行列
        """
        distances = cosine_distances(self.vector_embeddings)
        return distances

    def compute_neighbor_overlap(self, k: int = 10) -> Dict:
        """
        近傍一致率を計算

        Args:
            k: 最近傍の数

        Returns:
            評価指標を含む辞書
        """
        ca_distances = self.compute_ca_distances()
        vector_distances = self.compute_vector_distances()

        overlaps = []
        jaccards = []

        for i in range(len(self.common_words)):
            # CA空間での最近傍
            ca_neighbors = set(np.argsort(ca_distances[i])[1:k+1])

            # ベクトル空間での最近傍
            vector_neighbors = set(np.argsort(vector_distances[i])[1:k+1])

            # オーバーラップを計算
            overlap = len(ca_neighbors & vector_neighbors)
            overlaps.append(overlap)

            # Jaccard係数を計算
            jaccard = overlap / len(ca_neighbors | vector_neighbors)
            jaccards.append(jaccard)

        return {
            "overlap_at_k": k,
            "mean_overlap": np.mean(overlaps),
            "std_overlap": np.std(overlaps),
            "mean_jaccard": np.mean(jaccards),
            "std_jaccard": np.std(jaccards),
            "overlaps": overlaps,
            "jaccards": jaccards
        }

    def compute_distance_correlation(self) -> Dict:
        """
        距離相関を計算

        Returns:
            評価指標を含む辞書
        """
        ca_distances = self.compute_ca_distances()
        vector_distances = self.compute_vector_distances()

        # 上三角行列のみを使用（対角線を除く）
        n = len(self.common_words)
        indices = np.triu_indices(n, k=1)

        ca_dist_flat = ca_distances[indices]
        vector_dist_flat = vector_distances[indices]

        # Spearman順位相関を計算
        correlation, p_value = spearmanr(ca_dist_flat, vector_dist_flat)

        return {
            "spearman_correlation": correlation,
            "p_value": p_value,
            "n_pairs": len(ca_dist_flat)
        }

    def evaluate_all(self, k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        全ての評価指標を計算

        Args:
            k_values: 評価するkの値のリスト

        Returns:
            全評価指標を含む辞書
        """
        print("\n=== 構造整合性評価 ===")

        results = {}

        # 距離相関
        print("\n[1/2] 距離相関を計算中...")
        dist_corr = self.compute_distance_correlation()
        results["distance_correlation"] = dist_corr

        print(f"  - Spearman相関: {dist_corr['spearman_correlation']:.4f}")
        print(f"  - p値: {dist_corr['p_value']:.4e}")

        # 近傍一致率
        print("\n[2/2] 近傍一致率を計算中...")
        neighbor_overlaps = {}

        for k in k_values:
            overlap_result = self.compute_neighbor_overlap(k)
            neighbor_overlaps[f"k={k}"] = overlap_result

            print(f"  - k={k}:")
            print(f"    - 平均オーバーラップ: {overlap_result['mean_overlap']:.2f}")
            print(f"    - 平均Jaccard係数: {overlap_result['mean_jaccard']:.4f}")

        results["neighbor_overlaps"] = neighbor_overlaps

        return results

    def get_word_comparison(self, word: str, k: int = 10) -> Dict:
        """
        特定の単語について、CA空間とベクトル空間での最近傍を比較

        Args:
            word: 単語
            k: 最近傍の数

        Returns:
            比較結果を含む辞書
        """
        if word not in self.common_words:
            raise ValueError(f"Word '{word}' not found in common words.")

        word_idx = self.common_words.index(word)

        # CA空間での最近傍
        ca_distances = self.compute_ca_distances()
        ca_neighbor_indices = np.argsort(ca_distances[word_idx])[1:k+1]
        ca_neighbors = [
            {
                "word": self.common_words[i],
                "distance": ca_distances[word_idx, i]
            }
            for i in ca_neighbor_indices
        ]

        # ベクトル空間での最近傍
        vector_distances = self.compute_vector_distances()
        vector_neighbor_indices = np.argsort(vector_distances[word_idx])[1:k+1]
        vector_neighbors = [
            {
                "word": self.common_words[i],
                "distance": vector_distances[word_idx, i]
            }
            for i in vector_neighbor_indices
        ]

        # オーバーラップを計算
        ca_neighbor_set = set([n["word"] for n in ca_neighbors])
        vector_neighbor_set = set([n["word"] for n in vector_neighbors])
        overlap = ca_neighbor_set & vector_neighbor_set

        return {
            "word": word,
            "ca_neighbors": ca_neighbors,
            "vector_neighbors": vector_neighbors,
            "overlap": list(overlap),
            "overlap_count": len(overlap),
            "jaccard": len(overlap) / len(ca_neighbor_set | vector_neighbor_set)
        }


if __name__ == "__main__":
    # テスト
    print("=== 構造整合性評価のテスト ===")

    # ダミーデータを作成
    n_words = 10
    words = [f"word_{i}" for i in range(n_words)]

    # CA座標（ダミー）
    ca_coords = pd.DataFrame(
        np.random.randn(n_words, 2),
        index=words,
        columns=["dim_0", "dim_1"]
    )

    # ベクトル埋め込み（ダミー）
    vector_embeddings = np.random.randn(n_words, 100)
    # 正規化
    vector_embeddings = vector_embeddings / np.linalg.norm(
        vector_embeddings, axis=1, keepdims=True
    )

    # 評価器を作成
    evaluator = StructureEvaluator(
        ca_word_coords=ca_coords,
        vector_words=words,
        vector_embeddings=vector_embeddings
    )

    # 全評価を実行
    results = evaluator.evaluate_all(k_values=[3, 5])

    # 特定の単語の比較
    word_comparison = evaluator.get_word_comparison("word_0", k=5)
    print(f"\n'word_0' の比較:")
    print(f"  - オーバーラップ数: {word_comparison['overlap_count']}")
    print(f"  - Jaccard係数: {word_comparison['jaccard']:.4f}")
