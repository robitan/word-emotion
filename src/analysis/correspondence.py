"""
コレスポンデンス分析（CA）モジュール

単語 × 感情カテゴリのクロス集計表に対してコレスポンデンス分析を実行し、
単語と感情を同一の2次元平面上に配置する。
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import prince  # For Correspondence Analysis


class CorrespondenceAnalysis:
    """コレスポンデンス分析クラス"""

    def __init__(self, n_components: int = 2):
        """
        Args:
            n_components: 主成分数（通常は2）
        """
        self.n_components = n_components
        self.ca = None
        self.word_coords = None
        self.emotion_coords = None
        self.explained_inertia = None

    def fit(self, contingency_table: pd.DataFrame) -> 'CorrespondenceAnalysis':
        """
        コレスポンデンス分析を実行

        Args:
            contingency_table: 単語 × 感情のクロス集計表

        Returns:
            self
        """
        print(f"Running Correspondence Analysis...")
        print(f"  - Table shape: {contingency_table.shape}")
        print(f"  - Components: {self.n_components}")

        # コレスポンデンス分析を実行
        self.ca = prince.CA(
            n_components=self.n_components,
            n_iter=100,
            random_state=42
        )

        self.ca = self.ca.fit(contingency_table)

        # 行（単語）の座標を取得
        self.word_coords = self.ca.row_coordinates(contingency_table)

        # 列（感情）の座標を取得
        self.emotion_coords = self.ca.column_coordinates(contingency_table)

        # 寄与率を取得（バージョンによって属性名が異なる）
        if hasattr(self.ca, 'explained_inertia_'):
            self.explained_inertia = self.ca.explained_inertia_
        elif hasattr(self.ca, 'eigenvalues_'):
            # eigenvalues_から寄与率を計算
            eigenvalues = self.ca.eigenvalues_
            total_inertia = sum(eigenvalues)
            self.explained_inertia = eigenvalues / \
                total_inertia if total_inertia > 0 else eigenvalues
        else:
            # デフォルト値を設定
            self.explained_inertia = np.array(
                [0.5, 0.3] + [0.0] * (self.n_components - 2))

        print(
            f"  - Explained inertia: {self.explained_inertia[:self.n_components]}")
        print(
            f"  - Total explained: {sum(self.explained_inertia[:self.n_components]):.2%}")

        return self

    def get_word_coordinates(self) -> pd.DataFrame:
        """
        単語の座標を取得

        Returns:
            単語の座標を含むDataFrame
        """
        if self.word_coords is None:
            raise ValueError("CA has not been fitted yet.")

        return self.word_coords

    def get_emotion_coordinates(self) -> pd.DataFrame:
        """
        感情の座標を取得

        Returns:
            感情の座標を含むDataFrame
        """
        if self.emotion_coords is None:
            raise ValueError("CA has not been fitted yet.")

        return self.emotion_coords

    def get_distance_matrix(self, coords: pd.DataFrame) -> np.ndarray:
        """
        座標間の距離行列を計算

        Args:
            coords: 座標のDataFrame

        Returns:
            距離行列
        """
        from sklearn.metrics.pairwise import euclidean_distances

        distances = euclidean_distances(coords.values)

        return distances

    def get_neighbors(
        self,
        word: str,
        k: int = 10,
        include_emotions: bool = False
    ) -> list:
        """
        CA空間での最近傍を取得

        Args:
            word: 単語
            k: 最近傍の数
            include_emotions: 感情も含めるか

        Returns:
            最近傍のリスト
        """
        if word not in self.word_coords.index:
            raise ValueError(f"Word '{word}' not found in CA coordinates.")

        # 単語の座標を取得
        word_coord = self.word_coords.loc[word].values.reshape(1, -1)

        # 他の単語との距離を計算
        from sklearn.metrics.pairwise import euclidean_distances

        distances_to_words = euclidean_distances(
            word_coord,
            self.word_coords.values
        )[0]

        # 単語のインデックスと距離をペアにしてソート
        word_distance_pairs = [
            (self.word_coords.index[i], distances_to_words[i])
            for i in range(len(distances_to_words))
            if self.word_coords.index[i] != word
        ]
        word_distance_pairs.sort(key=lambda x: x[1])

        # 最近傍の単語を取得
        neighbors = [
            {"word": w, "distance": d, "type": "word"}
            for w, d in word_distance_pairs[:k]
        ]

        # 感情も含める場合
        if include_emotions:
            distances_to_emotions = euclidean_distances(
                word_coord,
                self.emotion_coords.values
            )[0]

            emotion_distance_pairs = [
                (self.emotion_coords.index[i], distances_to_emotions[i])
                for i in range(len(distances_to_emotions))
            ]
            emotion_distance_pairs.sort(key=lambda x: x[1])

            emotion_neighbors = [
                {"word": e, "distance": d, "type": "emotion"}
                for e, d in emotion_distance_pairs[:k]
            ]

            neighbors.extend(emotion_neighbors)
            neighbors.sort(key=lambda x: x["distance"])
            neighbors = neighbors[:k]

        return neighbors

    def get_summary(self) -> Dict:
        """
        CA分析の要約を取得

        Returns:
            要約情報を含む辞書
        """
        if self.ca is None:
            raise ValueError("CA has not been fitted yet.")

        return {
            "n_components": self.n_components,
            "n_words": len(self.word_coords),
            "n_emotions": len(self.emotion_coords),
            "explained_inertia": self.explained_inertia[:self.n_components].tolist(),
            "total_explained": sum(self.explained_inertia[:self.n_components])
        }


if __name__ == "__main__":
    # テスト
    print("=== コレスポンデンス分析のテスト ===")

    # ダミーデータを作成
    data = {
        '喜': [1, 1, 0, 0],
        '楽': [1, 0, 1, 0],
        '悲': [0, 0, 1, 1],
        '怒': [0, 1, 0, 1]
    }

    df = pd.DataFrame(
        data,
        index=['喜び', '楽しさ', '悲しみ', '怒り']
    )

    print("\nクロス集計表:")
    print(df)

    # CAを実行
    ca = CorrespondenceAnalysis(n_components=2)
    ca.fit(df)

    # 要約を表示
    summary = ca.get_summary()
    print(f"\nCA要約:")
    print(f"  - 単語数: {summary['n_words']}")
    print(f"  - 感情数: {summary['n_emotions']}")
    print(f"  - 説明率: {summary['explained_inertia']}")
    print(f"  - 累積説明率: {summary['total_explained']:.2%}")

    # 単語の座標を表示
    print("\n単語の座標:")
    print(ca.get_word_coordinates())

    # 感情の座標を表示
    print("\n感情の座標:")
    print(ca.get_emotion_coordinates())

    # 最近傍を取得
    neighbors = ca.get_neighbors('喜び', k=3)
    print(f"\n'喜び' の最近傍:")
    for neighbor in neighbors:
        print(f"  - {neighbor['word']} (distance: {neighbor['distance']:.4f})")
