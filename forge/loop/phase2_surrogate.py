"""Phase 2: Train Surrogate Model (GP)."""

import warnings

import numpy as np
from rich.console import Console
from rich.table import Table

from forge.data.loader import load_dataset, get_feature_matrix, get_scored_rows
from forge.models.surrogate_model import SurrogateModel
from forge.utils.config import load_config, load_schema, SURROGATE_MODEL_PATH

console = Console()


def run_train_surrogate() -> None:
    """Train GP surrogate model on y_hat values."""
    config = load_config()
    schema = load_schema()
    df = load_dataset()

    scored = get_scored_rows(df)
    n_dropped = len(df) - len(scored)
    if n_dropped > 0:
        warnings.warn(f"y_hat が NaN の {n_dropped} 行を除外しました。", stacklevel=2)

    if len(scored) == 0:
        console.print("[red]y_hat が付与されたデータがありません。先に train-reward を実行してください。[/red]")
        return

    X = get_feature_matrix(scored, schema)
    y = scored["y_hat"].values.astype(float)

    surrogate_cfg = config["surrogate_model"]
    model = SurrogateModel(
        nu=surrogate_cfg.get("nu", 2.5),
        noise_variance=surrogate_cfg.get("noise_variance", 0.01),
        training_iterations=surrogate_cfg.get("training_iterations", 100),
    )

    losses = model.fit(X, y)
    model.save(SURROGATE_MODEL_PATH)

    # Display training loss
    table = Table(title="Surrogate Model 学習結果")
    table.add_column("項目", style="cyan")
    table.add_column("値", justify="right")
    table.add_row("学習データ数", str(len(scored)))
    table.add_row("特徴量次元", str(X.shape[1]))
    table.add_row("初期 Loss", f"{losses[0]:.4f}")
    table.add_row("最終 Loss", f"{losses[-1]:.4f}")
    table.add_row("学習 iterations", str(len(losses)))
    console.print(table)

    # Loss every 10 steps
    loss_table = Table(title="Training Loss 推移")
    loss_table.add_column("Step", justify="right")
    loss_table.add_column("Loss", justify="right")
    step = max(1, len(losses) // 10)
    for i in range(0, len(losses), step):
        loss_table.add_row(str(i + 1), f"{losses[i]:.4f}")
    loss_table.add_row(str(len(losses)), f"{losses[-1]:.4f}")
    console.print(loss_table)

    console.print(f"[green]Surrogate Model を {SURROGATE_MODEL_PATH} に保存しました。[/green]")
