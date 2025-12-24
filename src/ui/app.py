"""
比較UIアプリケーション

Streamlitを使用して、異なるベクトル空間とCA分析結果を比較・可視化する。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.data_loader import EmotionDataLoader
from src.embedding import EmbeddingModel
from src.vector_db import VectorDB
from src.analysis.correspondence import CorrespondenceAnalysis
from src.analysis.evaluation import StructureEvaluator
from src.training.triplet import compute_emotion_vector


# ページ設定
st.set_page_config(
    page_title="単語×感情 ベクトル空間比較システム",
    page_icon="📊",
    layout="wide"
)

# セッション状態の初期化
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'ca_fitted' not in st.session_state:
    st.session_state.ca_fitted = False


@st.cache_resource
def load_data():
    """データを読み込む"""
    loader = EmotionDataLoader()
    data = loader.load_all()
    return data, loader


@st.cache_resource
def load_models():
    """モデルとDBを読み込む"""
    embedding_model = EmbeddingModel()
    db = VectorDB()
    return embedding_model, db


@st.cache_resource
def fit_ca(_data):
    """コレスポンデンス分析を実行"""
    ca = CorrespondenceAnalysis(n_components=2)
    ca.fit(_data['contingency_table'])
    return ca


def main():
    st.title("📊 単語 × 感情 ベクトル空間比較システム")
    st.markdown("---")

    # データの読み込み
    with st.spinner("データを読み込んでいます..."):
        data, loader = load_data()
        embedding_model, db = load_models()
        ca = fit_ca(data)

    st.session_state.data_loaded = True
    st.session_state.ca_fitted = True

    # サイドバー
    st.sidebar.header("設定")

    # ページ選択
    page = st.sidebar.selectbox(
        "機能を選択",
        ["類似検索", "感情変化検索", "CA可視化", "構造整合性評価"]
    )

    # データ統計を表示
    st.sidebar.markdown("---")
    st.sidebar.subheader("データ統計")
    st.sidebar.metric("単語数", len(data['word_emotions']))
    st.sidebar.metric("感情カテゴリ数", len(data['emotion_map']))
    st.sidebar.metric("Positive pairs", len(data['positive_pairs']))
    st.sidebar.metric("Negative pairs", len(data['negative_pairs']))

    # ページごとの処理
    if page == "類似検索":
        show_similarity_search(data, embedding_model, db, loader)
    elif page == "感情変化検索":
        show_emotion_shift_search(data, embedding_model, db, loader, ca)
    elif page == "CA可視化":
        show_ca_visualization(data, ca, loader)
    elif page == "構造整合性評価":
        show_structure_evaluation(data, ca, db, embedding_model, loader)


def show_similarity_search(data, embedding_model, db, loader):
    """類似検索ページ"""
    st.header("🔍 類似検索")

    # 入力
    col1, col2 = st.columns([3, 1])

    with col1:
        word = st.text_input("検索する単語を入力", placeholder="例: 喜び")

    with col2:
        top_k = st.number_input("上位K件", min_value=1, max_value=50, value=10)

    # DB選択
    selected_dbs = st.multiselect(
        "比較するベクトル空間を選択",
        ["Baseline", "BCE", "Triplet"],
        default=["Baseline", "BCE", "Triplet"]
    )

    if st.button("検索") and word:
        if word not in data['word_emotions']:
            st.error(f"単語 '{word}' はデータに存在しません")
            return

        # 各DBで検索
        results_dict = {}

        for db_name in selected_dbs:
            db_type = db_name.lower()

            try:
                results = db.search_by_word(
                    collection_type=db_type,
                    word=word,
                    embedding_model=embedding_model,
                    top_k=top_k
                )
                results_dict[db_name] = results
            except Exception as e:
                st.warning(f"{db_name} での検索に失敗しました: {e}")

        # 結果を横並びで表示
        if results_dict:
            cols = st.columns(len(results_dict))

            for i, (db_name, results) in enumerate(results_dict.items()):
                with cols[i]:
                    st.subheader(f"{db_name}")

                    # 結果をDataFrameに変換
                    df_results = []
                    for j, result in enumerate(results):
                        emotions = ', '.join([
                            data['emotion_map'].get(e, e)
                            for e in result['emotions']
                        ])
                        df_results.append({
                            "順位": j + 1,
                            "単語": result['word'],
                            "スコア": f"{result['score']:.4f}",
                            "感情": emotions
                        })

                    st.dataframe(
                        pd.DataFrame(df_results),
                        hide_index=True,
                        use_container_width=True
                    )


def show_emotion_shift_search(data, embedding_model, db, loader, ca):
    """感情変化検索ページ"""
    st.header("🎭 感情変化検索")

    st.markdown("""
    単語に感情方向ベクトルを加えることで、感情の変化を表現します。
    """)

    # 入力
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        word = st.text_input("元の単語を入力", placeholder="例: 喜び")

    with col2:
        # 感情シンボルの選択肢を作成
        emotion_options = {
            data['emotion_map'][symbol]: symbol
            for symbol in data['emotion_map'].keys()
        }
        target_emotion = st.selectbox(
            "目標の感情を選択",
            options=list(emotion_options.keys())
        )
        target_emotion_symbol = emotion_options[target_emotion]

    with col3:
        lambda_ = st.slider("感情変化の強さ (λ)", 0.0, 2.0, 1.0, 0.1)

    # DB選択
    db_type = st.selectbox(
        "使用するベクトル空間",
        ["Triplet", "Baseline", "BCE"]
    ).lower()

    top_k = st.number_input("上位K件", min_value=1, max_value=50, value=10)

    if st.button("検索") and word:
        if word not in data['word_emotions']:
            st.error(f"単語 '{word}' はデータに存在しません")
            return

        # ベクトル空間から全単語の埋め込みを取得
        try:
            words, vectors = db.get_all_vectors(db_type)

            # 感情変化ベクトルを計算
            word_embeddings_dict = {w: vectors[i] for i, w in enumerate(words)}

            emotion_vector = compute_emotion_vector(
                word_embeddings=word_embeddings_dict,
                word_emotions=data['word_emotions'],
                emotion_symbol=target_emotion_symbol,
                method="mean_diff"
            )

            # 感情変化検索を実行
            results = db.search_with_emotion_shift(
                collection_type=db_type,
                word=word,
                emotion_vector=emotion_vector,
                lambda_=lambda_,
                embedding_model=embedding_model,
                top_k=top_k
            )

            # 結果を表示
            st.subheader(f"結果: '{word}' + {lambda_}λ × '{target_emotion}'")

            df_results = []
            for j, result in enumerate(results):
                emotions = ', '.join([
                    data['emotion_map'].get(e, e)
                    for e in result['emotions']
                ])
                df_results.append({
                    "順位": j + 1,
                    "単語": result['word'],
                    "スコア": f"{result['score']:.4f}",
                    "感情": emotions
                })

            st.dataframe(
                pd.DataFrame(df_results),
                hide_index=True,
                use_container_width=True
            )

        except Exception as e:
            st.error(f"検索に失敗しました: {e}")


def show_ca_visualization(data, ca, loader):
    """CA可視化ページ"""
    st.header("📈 コレスポンデンス分析（CA）可視化")

    # CA要約を表示
    summary = ca.get_summary()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("単語数", summary['n_words'])
    with col2:
        st.metric("感情数", summary['n_emotions'])
    with col3:
        st.metric("累積説明率", f"{summary['total_explained']:.2%}")

    # 座標を取得
    word_coords = ca.get_word_coordinates()
    emotion_coords = ca.get_emotion_coordinates()

    # 可視化オプション
    show_words = st.checkbox("単語を表示", value=True)
    show_emotions = st.checkbox("感情を表示", value=True)

    # プロット作成
    fig = go.Figure()

    if show_words:
        # 単語をプロット
        fig.add_trace(go.Scatter(
            x=word_coords.iloc[:, 0],
            y=word_coords.iloc[:, 1],
            mode='markers+text',
            name='単語',
            text=word_coords.index,
            textposition="top center",
            marker=dict(size=5, color='blue', opacity=0.6),
            textfont=dict(size=8)
        ))

    if show_emotions:
        # 感情をプロット
        # 感情名に変換
        emotion_names = [data['emotion_map'].get(e, e) for e in emotion_coords.index]

        fig.add_trace(go.Scatter(
            x=emotion_coords.iloc[:, 0],
            y=emotion_coords.iloc[:, 1],
            mode='markers+text',
            name='感情',
            text=emotion_names,
            textposition="top center",
            marker=dict(size=15, color='red', symbol='diamond', opacity=0.8),
            textfont=dict(size=12, color='red')
        ))

    fig.update_layout(
        title="CA 2次元プロット",
        xaxis_title=f"次元 1 ({summary['explained_inertia'][0]:.2%})",
        yaxis_title=f"次元 2 ({summary['explained_inertia'][1]:.2%})",
        height=700,
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True)

    # 特定の単語の近傍を表示
    st.subheader("単語の近傍分析")

    word = st.text_input("分析する単語を入力", placeholder="例: 喜び")
    k = st.number_input("近傍数", min_value=1, max_value=20, value=10)

    if word and word in word_coords.index:
        neighbors = ca.get_neighbors(word, k=k, include_emotions=True)

        df_neighbors = pd.DataFrame(neighbors)
        df_neighbors['distance'] = df_neighbors['distance'].apply(lambda x: f"{x:.4f}")

        st.dataframe(df_neighbors, hide_index=True, use_container_width=True)


def show_structure_evaluation(data, ca, db, embedding_model, loader):
    """構造整合性評価ページ"""
    st.header("📊 構造整合性評価")

    st.markdown("""
    CAの結果とベクトル空間の構造がどの程度一致しているかを評価します。
    """)

    # DB選択
    db_type = st.selectbox(
        "評価するベクトル空間",
        ["Baseline", "BCE", "Triplet"]
    ).lower()

    if st.button("評価を実行"):
        with st.spinner("評価中..."):
            try:
                # ベクトル空間から全単語の埋め込みを取得
                words, vectors = db.get_all_vectors(db_type)

                # CA座標を取得
                ca_coords = ca.get_word_coordinates()

                # 評価器を作成
                evaluator = StructureEvaluator(
                    ca_word_coords=ca_coords,
                    vector_words=words,
                    vector_embeddings=vectors
                )

                # 全評価を実行
                results = evaluator.evaluate_all(k_values=[5, 10, 20])

                # 距離相関を表示
                st.subheader("1. 距離相関")

                dist_corr = results['distance_correlation']

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Spearman相関係数", f"{dist_corr['spearman_correlation']:.4f}")
                with col2:
                    st.metric("p値", f"{dist_corr['p_value']:.4e}")

                # 近傍一致率を表示
                st.subheader("2. 近傍一致率")

                overlap_data = []
                for k_name, overlap_result in results['neighbor_overlaps'].items():
                    overlap_data.append({
                        "k": k_name,
                        "平均オーバーラップ": f"{overlap_result['mean_overlap']:.2f}",
                        "平均Jaccard係数": f"{overlap_result['mean_jaccard']:.4f}"
                    })

                st.dataframe(pd.DataFrame(overlap_data), hide_index=True, use_container_width=True)

                # 特定の単語の比較
                st.subheader("3. 単語ごとの比較")

                word = st.text_input("比較する単語を入力", placeholder="例: 喜び")
                k_compare = st.number_input("比較する近傍数", min_value=1, max_value=20, value=10)

                if word and word in evaluator.common_words:
                    comparison = evaluator.get_word_comparison(word, k=k_compare)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**CA空間での近傍**")
                        df_ca = pd.DataFrame(comparison['ca_neighbors'])
                        df_ca['distance'] = df_ca['distance'].apply(lambda x: f"{x:.4f}")
                        st.dataframe(df_ca, hide_index=True, use_container_width=True)

                    with col2:
                        st.write(f"**ベクトル空間での近傍**")
                        df_vector = pd.DataFrame(comparison['vector_neighbors'])
                        df_vector['distance'] = df_vector['distance'].apply(lambda x: f"{x:.4f}")
                        st.dataframe(df_vector, hide_index=True, use_container_width=True)

                    st.write(f"**オーバーラップ**: {comparison['overlap']}")
                    st.metric("Jaccard係数", f"{comparison['jaccard']:.4f}")

            except Exception as e:
                st.error(f"評価に失敗しました: {e}")


if __name__ == "__main__":
    main()
