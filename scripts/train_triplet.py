"""
Triplet学習実行スクリプト

Triplet Lossを用いて学習し、学習済み埋め込みをQdrantに登録する。
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import EmotionDataLoader
from src.embedding import EmbeddingModel
from src.vector_db import VectorDB
from src.training.triplet import TripletTrainer, TripletDataset
from sentence_transformers import SentenceTransformer


def main():
    print("=" * 80)
    print("Triplet学習")
    print("=" * 80)

    # 設定
    epochs = int(os.getenv("TRIPLET_EPOCHS", 10))
    batch_size = int(os.getenv("TRIPLET_BATCH_SIZE", 32))
    learning_rate = float(os.getenv("TRIPLET_LEARNING_RATE", 2e-5))
    margin = float(os.getenv("TRIPLET_MARGIN", 0.3))
    model_path = "checkpoints/triplet_model.pth"

    # 1. データの読み込み
    print("\n[1/6] データの読み込み...")
    loader = EmotionDataLoader()
    data = loader.load_all()

    print(f"  - 感情カテゴリ数: {len(data['emotion_map'])}")
    print(f"  - 単語数: {len(data['word_emotions'])}")
    print(f"  - Triplets: {len(data['triplets'])}")

    # 2. 埋め込みモデルの初期化
    print("\n[2/6] 埋め込みモデルの初期化...")
    embedding_model = EmbeddingModel()

    # 3. データセットの作成
    print("\n[3/6] データセットの作成...")
    dataset = TripletDataset(
        triplets=data['triplets'],
        embedding_model=embedding_model
    )
    print(f"  - データセットサイズ: {len(dataset)}")

    # 4. Triplet学習
    print("\n[4/6] Triplet学習を開始...")
    base_model = SentenceTransformer("BAAI/bge-m3")

    trainer = TripletTrainer(
        base_model=base_model,
        learning_rate=learning_rate,
        margin=margin
    )

    trained_model = trainer.train(
        train_dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        save_path=model_path
    )

    # 5. 学習済みモデルで埋め込みを生成
    print("\n[5/6] 学習済みモデルで埋め込みを生成...")
    words = list(data['word_emotions'].keys())
    texts = [embedding_model.create_emotion_text(word) for word in words]

    embeddings = trainer.encode(texts, batch_size=batch_size)

    # 辞書形式に変換
    word_embeddings_dict = {
        word: embeddings[i]
        for i, word in enumerate(words)
    }

    print(f"  - 埋め込み生成完了: {len(word_embeddings_dict)} words")

    # 6. Qdrantに登録
    print("\n[6/6] Qdrantに登録...")
    db = VectorDB()

    # Tripletコレクションを作成（既存の場合は再作成）
    db.create_collection("triplet", recreate=True)

    # 単語と埋め込みを登録
    db.insert_words(
        collection_type="triplet",
        word_embeddings=word_embeddings_dict,
        word_emotions=data['word_emotions'],
        batch_size=100
    )

    # コレクション情報を表示
    info = db.get_collection_info("triplet")
    print(f"\n[完了] Triplet学習が完了しました")
    print(f"  - コレクション名: {info['name']}")
    print(f"  - 登録された単語数: {info['points_count']}")
    print(f"  - ベクトル数: {info['vectors_count']}")
    print(f"  - ステータス: {info['status']}")
    print(f"  - モデル保存先: {model_path}")

    # テスト検索
    print("\n[テスト] 類似検索のテスト...")
    test_word = words[0]
    results = db.search(
        collection_type="triplet",
        query_vector=word_embeddings_dict[test_word],
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
