単語 × 感情 ベクトル空間構築・比較システム

追加機能設計書（RD_add）

⸻

付録 A. 追加実装：同一モデル（Linear → ReLU → Linear → Sigmoid）で BCE / Triplet を比較

A-1. 目的

BCE と Triplet の差分を 損失関数の違いのみに限定して比較する。
• ベースの埋め込みモデル：BAAI/bge-m3（共通）
• 入力テンプレ："{word} という単語が示す感情"（共通）
• 学習可能パラメータ：同一のヘッド（MLP）
• 変更点：損失関数のみ（BCE vs Triplet）

A-2. モデル定義（共通）

A-2.1 表記
• E(x) : bge-m3 による埋め込み（出力次元 1024）
• K : 感情カテゴリ数（ラベル次元）
• h : 中間次元（例：256 または 512）

A-2.2 ネットワーク

単語 w に対して入力文 x=f(w) を作り、以下を計算する。 1. ベース埋め込み

    •	v = normalize(E(x))  （v ∈ R^1024）

    2.	共通ヘッド（MLP）

    •	a = Linear1(v)          （a ∈ R^h）
    •	r = ReLU(a)
    •	z = Linear2(r)          （z ∈ R^K）  ※ここは logits
    •	p = sigmoid(z)          （p ∈ (0,1)^K）  ※感情の確率解釈

注：Triplet 学習では p 自体は損失に使わない（forward は同一のまま）。

A-3. 学習パターン 1：BCE（マルチラベル分類）

A-3.1 教師信号
• 各単語 w に対して感情ラベル y(w) ∈ {0,1}^K（またはスコア）を与える。

A-3.2 損失
• L_BCE = BCEWithLogitsLoss(z, y)

（実装上は logits z に対して BCEWithLogits を使う。sigmoid は可視化・推論用。）

A-3.3 出力として DB に登録するベクトル

2 系統を明示的に作る（推奨：両方保存）。
• BCE-embed（推薦）：e_bce(w) = normalize(z) （K 次元）
• BCE-prob：p(w) （K 次元、比較・説明用）

DB 登録（方式 A のまま）
• words_bce：主ベクトルは e_bce(w) を格納
• payload に p(w) を入れてもよい（サイズ次第で保存/非保存を選択）

A-4. 学習パターン 2：Triplet（同一モデル、損失だけ変更）

A-4.1 Triplet 用の埋め込み

BCE と同一 forward で得られる z を埋め込みとして用いる。
• e_tri(w) = normalize(z) （K 次元）

A-4.2 Triplet データ
• anchor a
• positive p
• negative n

（近い/遠いの定義は本文の「感情距離」定義に基づく。最初は共通感情の有無で良い。）

A-4.3 損失
• 距離：d(u,v) = 1 - cosine(u,v)
• L_tri = max(0, d(e(a), e(p)) - d(e(a), e(n)) + margin)

A-4.4 出力として DB に登録するベクトル
• words_triplet：主ベクトルは e_tri(w)（= normalize(z)）を格納

A-5. 注意点（重要） 1. 次元が 1024 → K に変わる

    •	この追加実装は「感情ラベル次元K」に射影するため、最終ベクトル次元は K になる。
    •	既存の「bge-m3 1024次元のまま学習するBCE/Triplet」とは別実験軸。

    2.	比較の公平性

    •	BCE版とTriplet版は 同一初期重み（同一 seed / 同一初期化）から開始する。
    •	学習率・バッチサイズ・エポック数・データサンプリングも揃える。

    3.	DBの構成

    •	既存の words_bce / words_triplet を使う場合、次元不一致が起きるため、

以下のいずれかを採用する：
• (推奨) コレクションを追加：words_bce_mlpK / words_triplet_mlpK
• 既存コレクションとは別プロジェクトとして扱う

A-6. UI 追加要件
• DB 選択に以下を追加（またはタブで切替）：
• Baseline(1024) / BCE(1024) / Triplet(1024)
• BCE-MLP(K) / Triplet-MLP(K)
• 結果比較は「同次元同士」で行う（1024 系と K 系は横並び比較対象を分ける）。

A-7. 目的別の期待される差
• BCE（分類）は「感情カテゴリの識別」が安定しやすい
• Triplet（距離学習）は「近傍構造（似た単語のまとまり）」が鋭くなる可能性

本付録は 損失関数だけを変えたときに上記差が出るかを確認するための追加実装である。
