# 単語 × 感情 ベクトル空間構築・比較システム

異なる学習方針で構築した複数の感情指向ベクトル空間と、元データのコレスポンデンス分析（CA）結果を同一UI上で比較・検証するシステム。

## セットアップ（Docker使用）

### 1. 環境変数の設定

```bash
cp .env.example .env
# 必要に応じて .env を編集
```

### 2. Dockerイメージのビルド

```bash
make build
# または
docker-compose build
```

### 3. コンテナの起動

```bash
make up
# または
docker-compose up -d
```

## クイックスタート

システムを起動して使い始めるまでの手順：

```bash
# 1. コンテナをビルド・起動
make build
make up

# 2. Baseline空間を構築（必須）
make baseline

# 3. （オプション）BCE学習を実行
make bce

# 4. （オプション）Triplet学習を実行
make triplet

# 5. UIを起動
make ui
```

UIは http://localhost:8501 で起動します。

## 詳細な使い方

### データローダーのテスト

3人の作業者データを和集合でマージし、単語×感情のデータを生成：

```bash
make test
```

### 1. Baseline空間の構築（必須）

既存のBAAI/bge-m3モデルを使用して埋め込みを生成：

```bash
make baseline
```

- 全単語の埋め込みを生成
- Qdrantの`words_baseline`コレクションに登録
- 完了まで数分かかる場合があります

### 2. BCE学習（オプション）

感情が近い単語ペアを学習し、類似検索に特化した空間を構築：

```bash
make bce
```

- Binary Cross Entropyで学習
- 学習済みモデルは`checkpoints/bce_model.pth`に保存
- Qdrantの`words_bce`コレクションに登録
- デフォルト: 10エポック（.envで変更可能）

### 3. Triplet学習（オプション）

感情変化演算に特化した空間を構築：

```bash
make triplet
```

- Triplet Lossで学習
- 学習済みモデルは`checkpoints/triplet_model.pth`に保存
- Qdrantの`words_triplet`コレクションに登録
- デフォルト: 10エポック（.envで変更可能）

### 4. UIの起動

Streamlit UIを起動して、各空間を比較・可視化：

```bash
make ui
```

ブラウザで http://localhost:8501 にアクセス。

#### UI機能

1. **類似検索**
   - 任意の単語を入力し、Baseline/BCE/Tripletの各空間で類似単語を検索
   - 複数の空間を横並びで比較可能

2. **感情変化検索**
   - 単語に感情方向ベクトルを加算し、意味変化を探索
   - λパラメータで変化の強さを調整

3. **CA可視化**
   - コレスポンデンス分析の結果を2次元プロットで表示
   - 単語と感情を同一平面上に配置

4. **構造整合性評価**
   - CA空間とベクトル空間の構造を定量的に比較
   - 近傍一致率、距離相関を計算

### コンテナ操作

#### コンテナに入る

```bash
make shell
```

#### ログの確認

```bash
make logs
```

#### コンテナの停止

```bash
make down
```

#### クリーンアップ（全削除）

```bash
make clean
```

## プロジェクト構造

- `data/`: 元データ（単語×感情のCSVファイル）
- `src/`: ソースコード
  - `data_loader.py`: データ読み込み・前処理
  - `embedding.py`: 埋め込み生成
  - `vector_db.py`: Qdrant操作
  - `training/`: 学習モジュール
  - `analysis/`: 分析モジュール
  - `ui/`: Streamlit UI
- `scripts/`: 実行スクリプト

## 詳細

詳細な設計については `RD.md` を参照してください。
