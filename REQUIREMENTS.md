# REQUIREMENTS.md
# Human-in-the-Loop データ駆動型最適化システム
# 定性絶対評価（順序カテゴリ）対応版

## 1. システム概要

限られた初期データから出発し、**人間によるランク評価（A/B/C等）**を唯一の教師信号として、
2つの推論器を同時に育成するクローズドループ最適化システム。

### 育成する推論器

| 推論器 | 入出力 | 役割 |
|--------|--------|------|
| Reward Model `h(C,A)` | ランクH → 連続スコアŶ | 定性評価を定量化する |
| Surrogate Model `f(C,A)` | (C,A) → (μ, σ) | スコアの予測と不確実性推定 |

---

## 2. 変数定義

```
C : Condition（条件ベクトル）
    - 前提環境・素材特性など「制御できない」入力
    - 連続値の特徴量ベクトル（記述子）として扱う
    - 初期は数種類のバリエーションしか存在しない（Cold Start）

A : Action（操作量ベクトル）
    - 温度・圧力・比率など「自由に制御できる」連続値パラメータ
    - 各次元に物理的な上下限（bounds）を持つ

S : State（生の観測結果）
    - 実験後に得られる質感・画像・波形などの観測
    - Phase 0 でどこまで数値化するか決定する
    - 初期実装では文字列メモ or 数値特徴量として記録のみ

H : Human Feedback（人間のランク評価）
    - 順序付きの離散カテゴリ値: e.g., "A" > "B" > "C"
    - ランク数・ラベルは設定ファイルで定義（可変）

Y : Latent Yield（潜在スコア）
    - H の背後にある真の連続的スコア（直接観測不可）
    - Reward Model が推定する対象
```

---

## 3. システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                    DataStore (CSV / SQLite)              │
│   D = { (C_i, A_i, S_i, H_i, Ŷ_i) }                   │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌─────────────────┐       ┌──────────────────────┐
│  Reward Model   │       │   Surrogate Model     │
│  h(C,A) → Ŷ    │       │   f(C,A) → (μ, σ)    │
│                 │       │                       │
│  Ordinal        │       │  Gaussian Process     │
│  Regression     │       │  (GPyTorch / GPy)     │
│  (mord / custom)│       │                       │
└────────┬────────┘       └──────────┬────────────┘
         │  Ŷ を D に付与            │ μ, σ を出力
         └────────────┬──────────────┘
                      ▼
           ┌──────────────────────┐
           │  Acquisition Func.   │
           │  α(C,A) = UCB/EI     │
           │  argmax over A       │
           │  (scipy.optimize)    │
           └──────────┬───────────┘
                      ▼
           ┌──────────────────────┐
           │  Next Experiment     │
           │  Proposal: (C*, A*)  │
           └──────────┬───────────┘
                      ▼
           ┌──────────────────────┐
           │  Human Evaluation    │
           │  実験 → S → H入力    │
           └──────────────────────┘
```

---

## 4. フェーズ定義（実装単位）

### Phase 0: 初期化・スキーマ定義
- **入力**: 既存の実験記録（形式TBD）
- **処理**:
  - データスキーマ（C/A/S/Hの次元・型・範囲）の定義と保存
  - 既存データのインポートとバリデーション
  - 記述子化パイプラインの定義（数値化・正規化）
- **出力**: `data/schema.yaml`, `data/dataset.csv`

### Phase 1: Reward Model 学習（Ordinal → Continuous）
- **入力**: `data/dataset.csv`（H列を使用）
- **処理**:
  - 順序回帰モデル（Ordinal Regression）の学習
  - 全データに対して潜在スコアŶを計算・付与
  - モデルの保存
- **出力**: `models/reward_model.pkl`, D に Ŷ 列追加
- **アルゴリズム候補**:
  - `mord.LogisticAT` または `mord.LogisticIT`（cumulative link model）
  - GPyTorch による Ordinal GP（拡張オプション）

### Phase 2: Surrogate Model 学習
- **入力**: `data/dataset.csv`（Ŷ列を使用）
- **処理**:
  - GP モデル `f(C,A) → (μ, σ)` の学習・更新
  - ハイパーパラメータ最適化（marginal likelihood 最大化）
  - モデルの保存
- **出力**: `models/surrogate_model.pt`
- **アルゴリズム**: GPyTorch `ExactGP` + RBF/Matern Kernel

### Phase 3: 獲得関数最大化・次実験提案
- **入力**: Surrogate Model, スキーマ（A の bounds）
- **処理**:
  - 獲得関数の構築（UCB: μ + κσ、または EI）
  - `scipy.optimize.differential_evolution` による A* の探索
  - 既存データとの重複チェック・多様性確保
- **出力**: 次回実験の提案 `(C*, A*)` をコンソールに表示・CSV保存

### Phase 4: 人間評価・データ更新
- **入力**: 実験結果の手動入力（CLI プロンプト）
- **処理**:
  - `S` の記録（テキストメモ or 数値）
  - `H`（ランク）の入力とバリデーション
  - データセットへの追加
  - Phase 1 へのループバック
- **出力**: `data/dataset.csv`（行追加）

---

## 5. 技術スタック

```yaml
language: Python 3.11+

core:
  - numpy
  - pandas
  - scikit-learn        # 前処理・評価
  - mord                # Ordinal Regression
  - gpytorch            # Gaussian Process Surrogate
  - torch               # GPyTorch backend
  - scipy               # 最適化ソルバー
  - pyyaml              # スキーマ定義

cli:
  - click               # CLIフレームワーク
  - rich                # コンソール表示（テーブル・プログレスバー）

dev:
  - pytest
  - black
  - ruff
```

---

## 6. ディレクトリ構成

```
hitl_optimizer/
├── CLAUDE.md
├── REQUIREMENTS.md
├── README.md
├── pyproject.toml
│
├── config/
│   └── default.yaml          # ランク定義・GPハイパラ・UCBのκ等
│
├── data/
│   ├── schema.yaml           # C/A/Sの次元・型・bounds定義（Phase0生成）
│   ├── dataset.csv           # メインデータストア
│   └── proposals/            # Phase3の提案履歴
│
├── models/
│   ├── reward_model.pkl      # Reward Model（Ordinal Regression）
│   └── surrogate_model.pt    # Surrogate Model（GP）
│
├── hitl_optimizer/
│   ├── __init__.py
│   ├── cli.py                # Click CLI エントリーポイント
│   │
│   ├── data/
│   │   ├── schema.py         # スキーマ定義・バリデーション
│   │   ├── loader.py         # データ読み込み・変換
│   │   └── descriptor.py     # 記述子化パイプライン
│   │
│   ├── models/
│   │   ├── reward_model.py   # Ordinal Regression ラッパー
│   │   ├── surrogate_model.py# GPyTorch GP ラッパー
│   │   └── base.py           # 共通インターフェース
│   │
│   ├── optimization/
│   │   ├── acquisition.py    # UCB / EI 実装
│   │   └── optimizer.py      # scipy ラッパー・提案生成
│   │
│   ├── loop/
│   │   ├── phase0_init.py
│   │   ├── phase1_reward.py
│   │   ├── phase2_surrogate.py
│   │   ├── phase3_propose.py
│   │   └── phase4_evaluate.py
│   │
│   └── utils/
│       ├── config.py
│       └── logging.py
│
└── tests/
    ├── test_reward_model.py
    ├── test_surrogate_model.py
    ├── test_acquisition.py
    └── fixtures/
        └── sample_data.csv
```

---

## 7. CLIコマンド設計

```bash
# スキーマ定義（初回のみ）
python -m hitl_optimizer init --config config/default.yaml

# データインポート（既存記録から）
python -m hitl_optimizer import --file path/to/existing_data.csv

# Phase 1: Reward Model 学習
python -m hitl_optimizer train-reward

# Phase 2: Surrogate Model 学習
python -m hitl_optimizer train-surrogate

# Phase 3: 次実験提案
python -m hitl_optimizer propose --condition "c1=0.5,c2=1.2"

# Phase 4: 評価入力
python -m hitl_optimizer evaluate --proposal-id latest

# ループ一括実行（Phase1→2→3→人間入力待ち）
python -m hitl_optimizer loop --condition "c1=0.5,c2=1.2"

# 現状サマリー表示
python -m hitl_optimizer status
```

---

## 8. 設定ファイル（config/default.yaml）

```yaml
ranks:
  labels: ["A", "B", "C"]   # 降順（A が最良）
  order: descending

reward_model:
  algorithm: mord_logistic_at   # or: mord_logistic_it, custom_gp
  regularization: 1.0

surrogate_model:
  kernel: matern              # or: rbf
  noise_variance: 0.01
  n_restarts: 5

acquisition:
  function: ucb               # or: ei
  kappa: 2.0                  # UCB の探索係数

optimizer:
  method: differential_evolution
  max_iter: 1000
  popsize: 15
  seed: 42
```

---

## 9. 未解決事項（Phase 0 で決定する）

| 項目 | 選択肢 | 決定方法 |
|------|--------|---------|
| C（条件）の次元数・型 | 未定 | schema.yaml で定義 |
| A（操作量）の次元数・bounds | 未定 | schema.yaml で定義 |
| S（観測）の扱い | テキストメモ / 数値特徴量 / 画像パス | Phase 0 で決定 |
| ランク数 | A/B/C 3段階 / A〜E 5段階 / 他 | config.yaml で設定 |
| 記述子化の手法 | そのまま数値 / 標準化 / PCA | Phase 0 で決定 |

---

## 10. 将来拡張（スコープ外）

- Streamlit / React による可視化 UI
- バッチ提案（複数実験の並列提案）
- GPyTorch Ordinal GP による Phase1/2 統合
- 不確実性の可視化（ペアプロット・信頼区間）
- MLflow による実験管理
