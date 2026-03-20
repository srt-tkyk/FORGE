"""Phase 1: Train Reward Model and assign y_hat to dataset."""

import warnings

import numpy as np
from rich.console import Console
from rich.table import Table

from forge.data.loader import load_dataset, save_dataset, get_feature_matrix, get_ranked_rows
from forge.models.reward_model import RewardModel
from forge.utils.config import load_config, load_schema, REWARD_MODEL_PATH, get_rank_mapping

console = Console()


def run_train_reward() -> None:
    """Train reward model and update y_hat in dataset."""
    config = load_config()
    schema = load_schema()
    df = load_dataset()

    ranked = get_ranked_rows(df)
    if len(ranked) == 0:
        console.print("[red]ランク付きデータがありません。[/red]")
        return

    rank_map = get_rank_mapping(config)
    X = get_feature_matrix(ranked, schema)
    y = ranked["h_rank"].map(rank_map).values.astype(int)

    # Check for unmapped ranks
    if np.any(np.isnan(y.astype(float))):
        invalid = set(ranked["h_rank"].unique()) - set(rank_map.keys())
        raise ValueError(f"不正なランク値: {invalid}")

    alpha = config["reward_model"]["regularization"]
    model = RewardModel(alpha=alpha)
    model.fit(X, y)

    # Predict y_hat for ALL rows
    X_all = get_feature_matrix(df, schema)
    y_hat = model.predict(X_all)
    df["y_hat"] = y_hat

    # Save
    model.save(REWARD_MODEL_PATH)
    save_dataset(df)

    # Display results
    table = Table(title="Reward Model 学習結果")
    table.add_column("項目", style="cyan")
    table.add_column("値", justify="right")
    table.add_row("学習データ数", str(len(ranked)))
    table.add_row("ランク数", str(len(rank_map)))
    table.add_row("特徴量次元", str(X.shape[1]))
    table.add_row("Ŷ 平均", f"{y_hat.mean():.4f}")
    table.add_row("Ŷ 標準偏差", f"{y_hat.std():.4f}")
    table.add_row("Ŷ 最小", f"{y_hat.min():.4f}")
    table.add_row("Ŷ 最大", f"{y_hat.max():.4f}")
    console.print(table)
    console.print(f"[green]Reward Model を {REWARD_MODEL_PATH} に保存しました。[/green]")
