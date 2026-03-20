# CLAUDE.md
# AI開発エージェント向け指示書
# Human-in-the-Loop データ駆動型最適化システム

## プロジェクトの本質を理解せよ

これは「2つの推論器を人間のランク評価から同時に育てる」システムである。

```
人間のランク(H)  →  Reward Model  →  潜在スコアŶ  →  Surrogate Model  →  次の実験提案
     ↑_____________________________ 実験 ← 提案 __________________________________|
```

**Reward Model**（ランク→連続値）と**Surrogate Model**（GP：予測＋不確実性）は
独立したモジュールだが、データフローで密結合している。
この依存関係を常に意識してコードを書け。

---

## アーキテクチャ制約（変更禁止）

```
C（条件）: 連続値ベクトル。記述子化済み。制御不可。
A（操作量）: 連続値ベクトル。boundsあり。制御対象。
H（ランク）: 順序付き離散カテゴリ。唯一の教師信号。
Ŷ（潜在スコア）: Reward Modelが推定する連続値。SurrogateのY軸。

データストア: data/dataset.csv（単一ソースオブトゥルース）
スキーマ定義: data/schema.yaml（Phase 0 で生成）
```

**H は決して数値にハードコーディングするな。**
ランクの数・ラベル・順序は必ず `config/default.yaml` の `ranks` セクションから読む。

---

## 技術スタック

```python
# 順序回帰
import mord  # mord.LogisticAT が第一選択

# Gaussian Process
import gpytorch
import torch

# CLI
import click
from rich.console import Console
from rich.table import Table

# 最適化
from scipy.optimize import differential_evolution

# 設定
import yaml  # config/default.yaml の読み書き
```

`mord` が使えない環境では `sklearn` の `LogisticRegression` をワンホット順序エンコードで
代替する（`models/reward_model.py` に fallback 実装を用意すること）。

---

## コーディング規約

### 命名規則

```python
# 変数名は仕様書の記号に合わせる
c_vec: np.ndarray   # Condition ベクトル (n_c,)
a_vec: np.ndarray   # Action ベクトル    (n_a,)
h_label: str        # Human rank label  e.g. "A"
y_hat: float        # Latent score      推定値
mu: float           # GP 予測平均
sigma: float        # GP 予測標準偏差

# モデルの命名
reward_model    # h(C,A) → Ŷ
surrogate_model # f(C,A) → (μ, σ)
```

### モデルの共通インターフェース（`models/base.py` を継承）

```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...
```

Surrogate Model は加えて:
```python
def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mu, sigma)"""
```

### データフレームのカラム規約

```python
# dataset.csv の必須カラム
REQUIRED_COLS = [
    "id",           # int, 自動採番
    "timestamp",    # ISO8601
    # C_ プレフィックス: Condition 特徴量
    # A_ プレフィックス: Action パラメータ
    "h_rank",       # str, e.g. "A"
    "y_hat",        # float, Reward Model の出力（Phase1後に付与）
    "s_note",       # str, 観測メモ（任意）
]
```

---

## フェーズ実装の優先順位と注意点

### Phase 0（スキーマ定義）
- `data/schema.yaml` を生成するインタラクティブCLIを作る
- C・A の各次元について: 名前・型・最小値・最大値・単位 を聞く
- ランク定義（ラベルリスト・最良が先頭か末尾か）を設定する
- **スキーマが存在しない場合、他のフェーズはすべて実行を拒否すること**

### Phase 1（Reward Model）
- `mord.LogisticAT` はランクを `0, 1, 2...` の整数にマッピングして渡す
- ランクと整数のマッピングは `schema.yaml` の `ranks.labels` の順序に従う
- `predict` の出力はそのままŶとして `dataset.csv` の `y_hat` 列に書き戻す
- **データが少ない（< 10件）場合は警告を出しつつも実行は許可する**

### Phase 2（Surrogate Model）
- GPyTorch `ExactGP` + `MaternKernel(nu=2.5)` を標準構成とする
- `y_hat` が NaN の行は学習から除外する（警告を出す）
- `likelihood.train()` と `model.train()` / `eval()` の切り替えを必ず行う
- **学習後に train loss を rich テーブルで表示すること**

### Phase 3（提案生成）
- `differential_evolution` は `bounds` を schema.yaml の A 定義から構築する
- 獲得関数の `kappa` は `config/default.yaml` から読む（ハードコード禁止）
- 提案を `data/proposals/proposal_{timestamp}.yaml` に保存する
- **提案時に μ, σ, α 値をコンソールに表示して人間が判断できるようにすること**

### Phase 4（評価入力）
- `click.prompt()` でランク入力を受け取り、schema のラベル一覧でバリデーション
- 入力後に確認プロンプトを表示（`Are you sure? [y/N]`）
- `y_hat` はこの時点では NaN でよい（次の Phase 1 で埋まる）

---

## エラーハンドリング方針

```python
# フェーズ実行前に必ず前提条件チェック
def _check_prerequisites(phase: int) -> None:
    if phase >= 1 and not Path("data/schema.yaml").exists():
        raise RuntimeError("schema.yaml が存在しません。先に `init` を実行してください")
    if phase >= 1 and not Path("data/dataset.csv").exists():
        raise RuntimeError("dataset.csv が存在しません。先に `import` を実行してください")
    if phase >= 2 and not Path("models/reward_model.pkl").exists():
        raise RuntimeError("Reward Model が存在しません。先に `train-reward` を実行してください")
```

データが少なすぎる場合: `raise` ではなく `warnings.warn` して続行。
スキーマ不整合: `raise ValueError` で即座に停止。

---

## テスト方針

```
tests/
├── test_reward_model.py      # mord との統合テスト
├── test_surrogate_model.py   # GPyTorch 予測テスト（μ, σ の shape確認）
├── test_acquisition.py       # UCB/EI の単体テスト
└── fixtures/
    └── sample_data.csv       # C 2次元・A 2次元・ランクA/B/C の合成データ
```

**テストでは実データ不要。`fixtures/sample_data.csv` は自動生成スクリプトで作成すること。**

---

## 実装の進め方（Research → Plan → Implement サイクル）

```
1. REQUIREMENTS.md を読む（常に）
2. 該当フェーズの仕様を確認する
3. 既存コードとのインターフェースを確認する（特に dataset.csv のカラム）
4. 実装する
5. pytest を走らせる
6. rich でコンソール出力を確認する
```

**1つのコミットで複数フェーズをまたがない。フェーズ単位で実装・テスト・コミットを完結させること。**

---

## よくやりがちなミス（禁止事項）

```python
# ❌ ランクをハードコード
rank_to_int = {"A": 0, "B": 1, "C": 2}

# ✅ スキーマから動的に構築
schema = load_schema()
rank_to_int = {label: i for i, label in enumerate(schema["ranks"]["labels"])}

# ❌ GPyTorchで eval() を忘れる
model.train()
# ... 学習 ...
pred = model(X_test)   # ← モードが train のまま

# ✅ eval に切り替える
model.eval()
likelihood.eval()
with torch.no_grad():
    pred = likelihood(model(X_test))

# ❌ boundsをハードコード
bounds = [(0, 100), (0, 1), (200, 500)]

# ✅ スキーマから構築
bounds = [(a["min"], a["max"]) for a in schema["actions"]]
```

---

## 初回実装チェックリスト

- [ ] `pyproject.toml` に依存パッケージをすべて記載
- [ ] `config/default.yaml` の雛形を作成
- [ ] `data/schema.yaml` の JSON Schema を定義（バリデーション用）
- [ ] `models/base.py` の抽象クラスを実装
- [ ] `fixtures/sample_data.csv` の生成スクリプトを実装
- [ ] Phase 0（`init` コマンド）を実装
- [ ] Phase 1（`train-reward` コマンド）を実装
- [ ] Phase 2（`train-surrogate` コマンド）を実装
- [ ] Phase 3（`propose` コマンド）を実装
- [ ] Phase 4（`evaluate` コマンド）を実装
- [ ] `loop` コマンド（Phase1→2→3→4の一括実行）を実装
- [ ] `status` コマンドの実装
- [ ] 全フェーズの pytest が通ること
