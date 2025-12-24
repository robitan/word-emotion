"""
Baseline空間構築スクリプト

既存のBAAI/bge-m3モデルを使用して、単語の埋め込みを生成し、
Qdrantのbaselineコレクションに登録する。
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import EmotionDataLoader
from src.embedding import EmbeddingModel
from src.vector_db import VectorDB


def main():
    print("=" * 80)
    print("Baseline空間の構築")
    print("=" * 80)

    # 1. データの読み込み
    print("\n[1/4] データの読み込み...")
    loader = EmotionDataLoader()
    data = loader.load_all()

    print(f"  - 感情カテゴリ数: {len(data['emotion_map'])}")
    print(f"  - 単語数: {len(data['word_emotions'])}")

    # 2. 埋め込みモデルの初期化
    print("\n[2/4] 埋め込みモデルの初期化...")
    embedding_model = EmbeddingModel()

    # 3. 単語の埋め込みを生成
    print("\n[3/4] 単語の埋め込みを生成...")
    words = list(data['word_emotions'].keys())
    print(f"  - 処理する単語数: {len(words)}")

    word_embeddings_dict = embedding_model.encode_word_dict(
        data['word_emotions'],
        batch_size=32,
        show_progress=True
    )

    print(f"  - 埋め込み生成完了: {len(word_embeddings_dict)} words")

    # 4. Qdrantに登録
    print("\n[4/4] Qdrantに登録...")
    db = VectorDB()

    # Baselineコレクションを作成（既存の場合は再作成）
    db.create_collection("baseline", recreate=True)

    # 単語と埋め込みを登録
    db.insert_words(
        collection_type="baseline",
        word_embeddings=word_embeddings_dict,
        word_emotions=data['word_emotions'],
        batch_size=100
    )

    # コレクション情報を表示
    info = db.get_collection_info("baseline")
    print(f"\n[完了] Baseline空間の構築が完了しました")
    print(f"  - コレクション名: {info['name']}")
    print(f"  - 登録された単語数: {info['points_count']}")
    print(f"  - ベクトル数: {info['vectors_count']}")
    print(f"  - ステータス: {info['status']}")

    # テスト検索
    print("\n[テスト] 類似検索のテスト...")
    test_word = words[0]
    results = db.search_by_word(
        collection_type="baseline",
        word=test_word,
        embedding_model=embedding_model,
        top_k=5
    )

    print(f"\n'{test_word}' の類似単語 (Top 5):")
    for i, result in enumerate(results):
        emotions = ', '.join([data['emotion_map'].get(e, e) for e in result['emotions']])
        print(f"  {i+1}. {result['word']} (score: {result['score']:.4f}) - [{emotions}]")

    print("\n" + "=" * 80)
    print("完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
