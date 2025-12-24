.PHONY: build up down logs shell test baseline bce triplet k-mlp-bce k-mlp-triplet ui clean

# Docker環境の構築
build:
	docker compose build

# コンテナの起動
up:
	docker compose up -d

# コンテナの停止
down:
	docker compose down

# ログの表示
logs:
	docker compose logs -f app

# アプリケーションコンテナに入る
shell:
	docker compose exec app /bin/bash

# データローダーのテスト
test:
	docker compose exec app python src/data_loader.py

# Baseline空間の構築
baseline:
	docker compose exec app python scripts/build_baseline.py

# BCE学習の実行（1024次元）
bce:
	docker compose exec app python scripts/train_bce.py

# Triplet学習の実行（1024次元）
triplet:
	docker compose exec app python scripts/train_triplet.py

# BCE-MLP学習の実行（K次元）
k-mlp-bce:
	docker compose exec app python scripts/train_bce_mlp.py

# Triplet-MLP学習の実行（K次元）
k-mlp-triplet:
	docker compose exec app python scripts/train_triplet_mlp.py

# UIの起動
ui:
	docker compose exec app streamlit run src/ui/app.py

# クリーンアップ
clean:
	docker compose down -v
	rm -rf qdrant_storage
