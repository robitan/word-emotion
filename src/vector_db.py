"""
ベクトルDB操作モジュール

Qdrantを使用して、3つのコレクション（baseline, bce, triplet）を管理する。
"""

import os
import uuid
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
import numpy as np


class VectorDB:
    """ベクトルDB管理クラス"""

    COLLECTIONS = {
        "baseline": "words_baseline",
        "bce": "words_bce",
        "triplet": "words_triplet"
    }

    def __init__(
        self,
        host: str = None,
        port: int = None,
        vector_dim: int = 1024
    ):
        """
        Args:
            host: Qdrantのホスト
            port: Qdrantのポート
            vector_dim: ベクトルの次元数
        """
        if host is None:
            host = os.getenv("QDRANT_HOST", "localhost")
        if port is None:
            port = int(os.getenv("QDRANT_PORT", 6333))

        self.host = host
        self.port = port
        self.vector_dim = vector_dim

        print(f"Connecting to Qdrant at {host}:{port}...")
        self.client = QdrantClient(host=host, port=port)
        print("Connected successfully.")

    def create_collection(self, collection_type: str, recreate: bool = False):
        """
        コレクションを作成

        Args:
            collection_type: コレクションタイプ（baseline/bce/triplet）
            recreate: 既存のコレクションを削除して再作成するか
        """
        collection_name = self.COLLECTIONS[collection_type]

        # 既存のコレクションを確認
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if exists:
            if recreate:
                print(f"Deleting existing collection: {collection_name}")
                self.client.delete_collection(collection_name)
            else:
                print(f"Collection already exists: {collection_name}")
                return

        # コレクションを作成
        print(f"Creating collection: {collection_name}")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.vector_dim,
                distance=Distance.COSINE
            )
        )
        print(f"Collection created: {collection_name}")

    def insert_words(
        self,
        collection_type: str,
        word_embeddings: Dict[str, np.ndarray],
        word_emotions: Dict[str, set] = None,
        batch_size: int = 100
    ):
        """
        単語とその埋め込みをコレクションに挿入

        Args:
            collection_type: コレクションタイプ
            word_embeddings: 単語 -> 埋め込みベクトル の辞書
            word_emotions: 単語 -> 感情シンボルのセット の辞書（オプション）
            batch_size: バッチサイズ
        """
        collection_name = self.COLLECTIONS[collection_type]

        points = []
        for word, embedding in word_embeddings.items():
            # ペイロードを構築
            payload = {
                "word": word,
                "input_text": f"{word} という単語が示す感情"
            }

            # 感情ラベルを追加
            if word_emotions and word in word_emotions:
                emotions = list(word_emotions[word])
                payload["emotions"] = emotions

            # ポイントを作成
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)

            # バッチサイズに達したら挿入
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                points = []

        # 残りを挿入
        if points:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

        print(f"Inserted {len(word_embeddings)} words into {collection_name}")

    def search(
        self,
        collection_type: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        類似検索

        Args:
            collection_type: コレクションタイプ
            query_vector: クエリベクトル
            top_k: 取得する上位K件
            score_threshold: スコアの閾値（オプション）

        Returns:
            検索結果のリスト
        """
        collection_name = self.COLLECTIONS[collection_type]

        # query_pointsメソッドを使用（searchは廃止されました）
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True
        )

        # 結果を整形
        formatted_results = []
        for result in results.points:
            formatted_results.append({
                "word": result.payload["word"],
                "score": result.score,
                "emotions": result.payload.get("emotions", []),
                "payload": result.payload
            })

        return formatted_results

    def search_by_word(
        self,
        collection_type: str,
        word: str,
        embedding_model,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        単語で検索

        Args:
            collection_type: コレクションタイプ
            word: 検索する単語
            embedding_model: 埋め込みモデル
            top_k: 取得する上位K件

        Returns:
            検索結果のリスト
        """
        # 単語を埋め込みベクトルに変換
        query_vector = embedding_model.encode_words([word], show_progress=False)[0]

        # 検索
        results = self.search(
            collection_type=collection_type,
            query_vector=query_vector,
            top_k=top_k
        )

        return results

    def search_with_emotion_shift(
        self,
        collection_type: str,
        word: str,
        emotion_vector: np.ndarray,
        lambda_: float,
        embedding_model,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        感情変化ベクトルを使った検索

        Args:
            collection_type: コレクションタイプ
            word: 元の単語
            emotion_vector: 感情変化ベクトル
            lambda_: 感情変化の強さ
            embedding_model: 埋め込みモデル
            top_k: 取得する上位K件

        Returns:
            検索結果のリスト
        """
        # 単語の埋め込みを取得
        word_vector = embedding_model.encode_words([word], show_progress=False)[0]

        # 感情変化を適用
        query_vector = word_vector + lambda_ * emotion_vector

        # 正規化
        query_vector = query_vector / np.linalg.norm(query_vector)

        # 検索
        results = self.search(
            collection_type=collection_type,
            query_vector=query_vector,
            top_k=top_k
        )

        return results

    def get_all_vectors(self, collection_type: str) -> Tuple[List[str], np.ndarray]:
        """
        コレクション内の全ベクトルを取得

        Args:
            collection_type: コレクションタイプ

        Returns:
            (単語リスト, ベクトル行列)
        """
        collection_name = self.COLLECTIONS[collection_type]

        # 全ポイントを取得（スクロール方式）
        all_points = []
        offset = None

        while True:
            records, offset = self.client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_vectors=True
            )

            all_points.extend(records)

            if offset is None:
                break

        # 単語とベクトルを抽出
        words = [point.payload["word"] for point in all_points]
        vectors = np.array([point.vector for point in all_points])

        return words, vectors

    def get_collection_info(self, collection_type: str) -> Dict[str, Any]:
        """
        コレクションの情報を取得

        Args:
            collection_type: コレクションタイプ

        Returns:
            コレクション情報
        """
        collection_name = self.COLLECTIONS[collection_type]

        info = self.client.get_collection(collection_name)

        # ポイント数を取得（count メソッドを使用）
        count_result = self.client.count(collection_name)
        points_count = count_result.count if hasattr(count_result, 'count') else count_result

        return {
            "name": collection_name,
            "vectors_count": points_count,  # ベクトル数 = ポイント数
            "points_count": points_count,
            "status": info.status
        }


if __name__ == "__main__":
    # テスト
    print("=== ベクトルDBのテスト ===")

    # DBに接続
    db = VectorDB()

    # Baselineコレクションを作成
    db.create_collection("baseline", recreate=True)

    # テストデータを挿入
    test_words = {
        "喜び": np.random.randn(1024),
        "悲しみ": np.random.randn(1024),
        "怒り": np.random.randn(1024)
    }

    # 正規化
    for word in test_words:
        test_words[word] = test_words[word] / np.linalg.norm(test_words[word])

    test_emotions = {
        "喜び": {"喜"},
        "悲しみ": {"悲"},
        "怒り": {"怒"}
    }

    db.insert_words("baseline", test_words, test_emotions)

    # コレクション情報を取得
    info = db.get_collection_info("baseline")
    print(f"\nCollection info: {info}")

    # 検索テスト
    query_vector = test_words["喜び"]
    results = db.search("baseline", query_vector, top_k=3)

    print(f"\n検索結果:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['word']} (score: {result['score']:.4f})")
