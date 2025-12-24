"""
K-MLP-BCE学習実行スクリプト

感情ラベル空間（K次元）でマルチラベル分類として学習し、
学習済み埋め込みをQdrantに登録する。
"""

import torch
from sentence_transformers import SentenceTransformer
from src.training.bce_mlp import BCEMLPTrainer, EmotionLabelDataset
from src.training.mlp_model import EmotionMLPHead
from src.vector_db import VectorDB
from src.embedding import EmbeddingModel
from src.data_loader import EmotionDataLoader
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    print("=" * 80)
    print("K-MLP-BCE学習（感情ラベル空間K次元）")
    print("=" * 80)

    # 設定
    epochs = int(os.getenv("BCE_EPOCHS", 10))
    batch_size = int(os.getenv("BCE_BATCH_SIZE", 32))
    learning_rate = float(os.getenv("BCE_LEARNING_RATE", 2e-5))
    hidden_dim = int(os.getenv("MLP_HIDDEN_DIM", 256))
    model_path = "checkpoints/bce_mlp_model.pth"

    # 1. データの読み込み
    print("\n[1/6] データの読み込み...")
    loader = EmotionDataLoader()
    data = loader.load_all()

    num_emotions = len(data['emotion_map'])
    print(f"  - 感情カテゴリ数 (K): {num_emotions}")
    print(f"  - 単語数: {len(data['word_emotions'])}")

    # 2. 埋め込みモデルの初期化
    print("\n[2/6] 埋め込みモデルの初期化...")
    embedding_model = EmbeddingModel()
    base_model = SentenceTransformer("BAAI/bge-m3")

    # 3. MLPヘッドモデルの作成
    print("\n[3/6] MLPヘッドモデルの作成...")

    # 同一の初期重みで公平に比較するため、シードを設定
    torch.manual_seed(42)

    model = EmotionMLPHead(
        base_model=base_model,
        num_emotions=num_emotions,
        hidden_dim=hidden_dim,
        freeze_base=True
    )

    print(f"  - 入力次元: 1024")
    print(f"  - 中間次元: {hidden_dim}")
    print(f"  - 出力次元: {num_emotions}")

    # 4. データセットの作成
    print("\n[4/6] データセットの作成...")
    dataset = EmotionLabelDataset(
        word_emotions=data['word_emotions'],
        emotion_map=data['emotion_map'],
        embedding_model=embedding_model
    )
    print(f"  - データセットサイズ: {len(dataset)}")

    # 5. BCE学習
    print("\n[5/6] K-MLP-BCE学習を開始...")
    trainer = BCEMLPTrainer(
        model=model,
        learning_rate=learning_rate
    )

    trained_model = trainer.train(
        train_dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        save_path=model_path
    )

    # 6. 学習済みモデルで埋め込みを生成
    print("\n[6/6] 学習済みモデルで埋め込みを生成...")
    words = list(data['word_emotions'].keys())
    texts = [embedding_model.create_emotion_text(word) for word in words]

    embeddings = trainer.encode(texts, batch_size=batch_size, normalize=True)

    # 辞書形式に変換
    word_embeddings_dict = {
        word: embeddings[i]
        for i, word in enumerate(words)
    }

    print(f"  - 埋め込み生成完了: {len(word_embeddings_dict)} words")
    print(f"  - 埋め込み次元: {embeddings.shape[1]}")

    # 7. Qdrantに登録
    print("\n[7/7] Qdrantに登録...")
    db = VectorDB(vector_dim=num_emotions)  # K次元

    # K-MLP-BCEコレクションを作成（既存の場合は再作成）
    db.create_collection("bce_mlp", recreate=True)

    # 単語と埋め込みを登録
    db.insert_words(
        collection_type="bce_mlp",
        word_embeddings=word_embeddings_dict,
        word_emotions=data['word_emotions'],
        batch_size=100
    )

    # コレクション情報を表示
    info = db.get_collection_info("bce_mlp")
    print(f"\n[完了] K-MLP-BCE学習が完了しました")
    print(f"  - コレクション名: {info['name']}")
    print(f"  - 登録された単語数: {info['points_count']}")
    print(f"  - ベクトル次元: {num_emotions}")
    print(f"  - モデル保存先: {model_path}")

    # テスト検索
    print("\n[テスト] 類似検索のテスト...")
    test_word = words[0]
    results = db.search(
        collection_type="bce_mlp",
        query_vector=word_embeddings_dict[test_word],
        top_k=5
    )

    print(f"\n'{test_word}' の類似単語 (Top 5):")
    for i, result in enumerate(results):
        emotions = ', '.join([data['emotion_map'].get(e, e)
                             for e in result['emotions']])
        print(
            f"  {i+1}. {result['word']} (score: {result['score']:.4f}) - [{emotions}]")

    print("\n" + "=" * 80)
    print("完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
