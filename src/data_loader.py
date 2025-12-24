"""
データ読み込み・前処理モジュール

3人の作業者データを読み込み、和集合でマージして単語×感情のデータを提供する。
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class EmotionDataLoader:
    """感情データローダー"""

    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: データディレクトリのパス
        """
        self.data_dir = Path(data_dir)
        self.emotion_map = {}  # シンボル -> 感情名
        self.symbol_map = {}   # 感情名 -> シンボル
        self.word_emotions = defaultdict(set)  # 単語 -> 感情シンボルのセット

    def load_emotion_categories(self) -> Dict[str, str]:
        """
        感情分類データを読み込む

        Returns:
            シンボル -> 感情名 の辞書
        """
        emotion_file = self.data_dir / "感情表現18.7.24.xlsx - 感情分類.csv"
        df = pd.read_csv(emotion_file)

        for _, row in df.iterrows():
            emotion = row['Emotion']
            symbol = row['Symbol(全て全角)']
            self.emotion_map[symbol] = emotion
            self.symbol_map[emotion] = symbol

        print(f"感情カテゴリ数: {len(self.emotion_map)}")
        return self.emotion_map

    def load_worker_data(self, worker_name: str) -> pd.DataFrame:
        """
        作業者データを読み込む

        Args:
            worker_name: 作業者名（A, B, C）

        Returns:
            作業者のDataFrame
        """
        worker_file = self.data_dir / f"感情表現18.7.24.xlsx - 作業者{worker_name}.csv"
        df = pd.read_csv(worker_file)
        return df

    def merge_worker_data(self) -> Dict[str, Set[str]]:
        """
        3人の作業者データを和集合でマージする

        Returns:
            単語 -> 感情シンボルのセット の辞書
        """
        workers = ['A', 'B', 'C']

        for worker in workers:
            df = self.load_worker_data(worker)

            for _, row in df.iterrows():
                word = row['Word']
                emotion_str = str(row['Emotion'])

                # 感情シンボルを個別に分解（複数の感情が連結されている場合）
                # 例: "喜楽" -> ["喜", "楽"]
                if pd.notna(emotion_str) and emotion_str:
                    for emotion_symbol in emotion_str:
                        if emotion_symbol in self.emotion_map:
                            self.word_emotions[word].add(emotion_symbol)

        print(f"マージ後の単語数: {len(self.word_emotions)}")
        print(f"感情が付与された単語数: {sum(1 for emotions in self.word_emotions.values() if emotions)}")

        return dict(self.word_emotions)

    def get_words_by_emotion(self, emotion_symbol: str) -> List[str]:
        """
        特定の感情を持つ単語のリストを取得

        Args:
            emotion_symbol: 感情シンボル

        Returns:
            単語のリスト
        """
        return [word for word, emotions in self.word_emotions.items()
                if emotion_symbol in emotions]

    def get_emotion_similarity_pairs(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        感情が類似/非類似な単語ペアを生成（BCE学習用）

        Returns:
            (positive_pairs, negative_pairs)
            positive_pairs: 感情が近い単語ペアのリスト
            negative_pairs: 感情が遠い単語ペアのリスト
        """
        positive_pairs = []
        negative_pairs = []

        words = list(self.word_emotions.keys())

        for i, word1 in enumerate(words):
            emotions1 = self.word_emotions[word1]
            if not emotions1:
                continue

            for word2 in words[i+1:]:
                emotions2 = self.word_emotions[word2]
                if not emotions2:
                    continue

                # Jaccard類似度を計算
                intersection = len(emotions1 & emotions2)
                union = len(emotions1 | emotions2)

                if union == 0:
                    continue

                similarity = intersection / union

                # 類似度が高い（>= 0.5）ならpositive、低い（< 0.2）ならnegative
                if similarity >= 0.5:
                    positive_pairs.append((word1, word2))
                elif similarity < 0.2:
                    negative_pairs.append((word1, word2))

        print(f"Positive pairs: {len(positive_pairs)}")
        print(f"Negative pairs: {len(negative_pairs)}")

        return positive_pairs, negative_pairs

    def get_triplets(self, max_triplets_per_anchor: int = 10) -> List[Tuple[str, str, str]]:
        """
        Tripletデータを生成（Triplet学習用）

        K-MLP-BCEと公平に比較するため、各anchorからサンプリングする
        triplet数を制限する。

        Args:
            max_triplets_per_anchor: 各anchorから生成する最大triplet数

        Returns:
            (anchor, positive, negative) のリスト
        """
        import random

        triplets = []
        words = list(self.word_emotions.keys())

        for anchor in words:
            anchor_emotions = self.word_emotions[anchor]
            if not anchor_emotions:
                continue

            # Positiveを探す（感情が近い）
            positives = []
            negatives = []

            for word in words:
                if word == anchor:
                    continue

                word_emotions = self.word_emotions[word]
                if not word_emotions:
                    continue

                intersection = len(anchor_emotions & word_emotions)
                union = len(anchor_emotions | word_emotions)

                if union == 0:
                    continue

                similarity = intersection / union

                if similarity >= 0.5:
                    positives.append(word)
                elif similarity < 0.2:
                    negatives.append(word)

            # Tripletを生成（各anchorからmax_triplets_per_anchor個まで）
            if positives and negatives:
                # 可能なtriplet数を計算
                possible_triplets = len(positives) * len(negatives)
                num_triplets = min(max_triplets_per_anchor, possible_triplets)

                # ランダムにサンプリング
                for _ in range(num_triplets):
                    pos = random.choice(positives)
                    neg = random.choice(negatives)
                    triplets.append((anchor, pos, neg))

        print(f"Triplets: {len(triplets)} (max {max_triplets_per_anchor} per anchor)")
        return triplets

    def create_contingency_table(self) -> pd.DataFrame:
        """
        単語 × 感情カテゴリのクロス集計表を作成（CA分析用）

        Returns:
            単語を行、感情を列とするDataFrame
        """
        # 全感情シンボルのリスト
        all_emotions = sorted(self.emotion_map.keys())

        # クロス集計表を作成
        table_data = []

        for word in sorted(self.word_emotions.keys()):
            row = {'word': word}
            emotions = self.word_emotions[word]

            for emotion_symbol in all_emotions:
                # 感情が付与されていれば1、そうでなければ0
                row[emotion_symbol] = 1 if emotion_symbol in emotions else 0

            table_data.append(row)

        df = pd.DataFrame(table_data)
        df = df.set_index('word')

        print(f"クロス集計表のサイズ: {df.shape}")
        return df

    def load_all(self) -> Dict:
        """
        全データを読み込んで返す

        Returns:
            全データを含む辞書
        """
        self.load_emotion_categories()
        self.merge_worker_data()

        positive_pairs, negative_pairs = self.get_emotion_similarity_pairs()
        triplets = self.get_triplets()
        contingency_table = self.create_contingency_table()

        return {
            'emotion_map': self.emotion_map,
            'symbol_map': self.symbol_map,
            'word_emotions': dict(self.word_emotions),
            'positive_pairs': positive_pairs,
            'negative_pairs': negative_pairs,
            'triplets': triplets,
            'contingency_table': contingency_table
        }


if __name__ == "__main__":
    # テスト
    loader = EmotionDataLoader()
    data = loader.load_all()

    print("\n=== データ統計 ===")
    print(f"感情カテゴリ数: {len(data['emotion_map'])}")
    print(f"単語数: {len(data['word_emotions'])}")
    print(f"Positive pairs: {len(data['positive_pairs'])}")
    print(f"Negative pairs: {len(data['negative_pairs'])}")
    print(f"Triplets: {len(data['triplets'])}")
    print(f"クロス集計表: {data['contingency_table'].shape}")

    # サンプル表示
    print("\n=== 感情カテゴリ（最初の5つ） ===")
    for symbol, emotion in list(data['emotion_map'].items())[:5]:
        print(f"{symbol}: {emotion}")

    print("\n=== 単語と感情（最初の5つ） ===")
    for word, emotions in list(data['word_emotions'].items())[:5]:
        emotion_names = [data['emotion_map'][e] for e in emotions if e in data['emotion_map']]
        print(f"{word}: {', '.join(emotion_names)}")
