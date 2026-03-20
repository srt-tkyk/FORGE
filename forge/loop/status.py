"""Status command: display current system state."""

from rich.console import Console
from rich.table import Table

from forge.utils.config import (
    SCHEMA_PATH,
    DATASET_PATH,
    REWARD_MODEL_PATH,
    SURROGATE_MODEL_PATH,
    PROPOSALS_DIR,
)

console = Console()


def run_status() -> None:
    """Display current system status summary."""
    table = Table(title="FORGE ステータス")
    table.add_column("項目", style="cyan")
    table.add_column("状態", justify="left")

    # Schema
    if SCHEMA_PATH.exists():
        from forge.utils.config import load_schema
        schema = load_schema()
        n_c = len(schema["conditions"])
        n_a = len(schema["actions"])
        ranks = schema["ranks"]["labels"]
        table.add_row("スキーマ", f"[green]定義済み[/green] (C:{n_c}, A:{n_a}, Ranks:{ranks})")
    else:
        table.add_row("スキーマ", "[red]未定義[/red] → `forge init` を実行")

    # Dataset
    if DATASET_PATH.exists():
        import pandas as pd
        df = pd.read_csv(DATASET_PATH)
        n_total = len(df)
        n_ranked = df["h_rank"].notna().sum()
        n_scored = df["y_hat"].notna().sum()
        table.add_row("データセット", f"[green]{n_total} 行[/green] (ランク付き: {n_ranked}, Ŷ付き: {n_scored})")

        # Rank distribution
        if n_ranked > 0:
            dist = df["h_rank"].value_counts().to_dict()
            dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(dist.items()))
            table.add_row("ランク分布", dist_str)
    else:
        table.add_row("データセット", "[red]なし[/red] → `forge import` を実行")

    # Reward Model
    if REWARD_MODEL_PATH.exists():
        table.add_row("Reward Model", "[green]学習済み[/green]")
    else:
        table.add_row("Reward Model", "[yellow]未学習[/yellow]")

    # Surrogate Model
    if SURROGATE_MODEL_PATH.exists():
        table.add_row("Surrogate Model", "[green]学習済み[/green]")
    else:
        table.add_row("Surrogate Model", "[yellow]未学習[/yellow]")

    # Proposals
    if PROPOSALS_DIR.exists():
        proposals = sorted(PROPOSALS_DIR.glob("proposal_*.yaml"))
        if proposals:
            table.add_row("提案数", f"{len(proposals)} 件 (最新: {proposals[-1].name})")
        else:
            table.add_row("提案数", "0 件")
    else:
        table.add_row("提案数", "0 件")

    console.print(table)
