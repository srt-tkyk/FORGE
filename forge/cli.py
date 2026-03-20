"""FORGE CLI entry point."""

import click
from rich.console import Console

console = Console()


@click.group()
def cli():
    """FORGE: Feedback-Oriented Reward & Gaussian Estimation."""
    pass


@cli.command()
def init():
    """Phase 0: スキーマ定義（インタラクティブ）"""
    from forge.loop.phase0_init import run_init_interactive

    run_init_interactive()


@cli.command(name="import")
@click.option("--file", "file_path", required=True, type=click.Path(exists=True), help="CSV file")
def import_data(file_path):
    """既存データのインポート"""
    from forge.loop.phase0_init import _display_schema_summary
    from forge.data.loader import save_dataset, create_empty_dataset
    from forge.data.schema import validate_schema
    from forge.utils.config import load_schema, check_prerequisites, DATASET_PATH

    import pandas as pd

    check_prerequisites(phase=1)
    schema = load_schema()

    # Read source CSV
    src = pd.read_csv(file_path)
    console.print(f"[cyan]読み込み: {len(src)} 行[/cyan]")

    # Create target dataset with schema columns
    target = create_empty_dataset(schema)
    c_cols = [f"C_{c['name']}" for c in schema["conditions"]]
    a_cols = [f"A_{a['name']}" for a in schema["actions"]]

    # Map columns: try exact match first, then C_/A_ prefixed
    for col in c_cols + a_cols:
        bare = col.split("_", 1)[1]
        if col in src.columns:
            target[col] = src[col]
        elif bare in src.columns:
            target[col] = src[bare]
        else:
            raise ValueError(f"カラム '{col}' (or '{bare}') がソースCSVに見つかりません")

    # Map h_rank
    if "h_rank" in src.columns:
        target["h_rank"] = src["h_rank"]
    elif "rank" in src.columns:
        target["h_rank"] = src["rank"]
    else:
        raise ValueError("ランク列 ('h_rank' or 'rank') がソースCSVに見つかりません")

    # Validate ranks
    valid_labels = set(schema["ranks"]["labels"])
    invalid = set(target["h_rank"].dropna().unique()) - valid_labels
    if invalid:
        raise ValueError(f"不正なランク値: {invalid}。有効: {valid_labels}")

    # Fill metadata columns
    import numpy as np
    from datetime import datetime, timezone

    target["id"] = range(1, len(src) + 1)
    target["timestamp"] = datetime.now(timezone.utc).isoformat()
    target["y_hat"] = np.nan
    if "s_note" in src.columns:
        target["s_note"] = src["s_note"]
    else:
        target["s_note"] = ""

    save_dataset(target)
    console.print(f"[green]{len(target)} 行を {DATASET_PATH} に保存しました。[/green]")


@cli.command(name="train-reward")
def train_reward():
    """Phase 1: Reward Model 学習"""
    from forge.loop.phase1_reward import run_train_reward
    from forge.utils.config import check_prerequisites

    check_prerequisites(phase=1)
    run_train_reward()


@cli.command(name="train-surrogate")
def train_surrogate():
    """Phase 2: Surrogate Model 学習"""
    from forge.loop.phase2_surrogate import run_train_surrogate
    from forge.utils.config import check_prerequisites

    check_prerequisites(phase=2)
    run_train_surrogate()


@cli.command()
@click.option("--condition", "condition_str", default=None, help="C values: 'c1=0.5,c2=1.2'")
def propose(condition_str):
    """Phase 3: 次実験の提案生成"""
    from forge.loop.phase3_propose import run_propose
    from forge.utils.config import check_prerequisites

    check_prerequisites(phase=2)
    run_propose(condition_str)


@cli.command()
@click.option("--proposal-id", default="latest", help="Proposal ID or 'latest'")
def evaluate(proposal_id):
    """Phase 4: 人間評価の入力"""
    from forge.loop.phase4_evaluate import run_evaluate
    from forge.utils.config import check_prerequisites

    check_prerequisites(phase=1)
    run_evaluate(proposal_id)


@cli.command()
@click.option("--condition", "condition_str", default=None, help="C values: 'c1=0.5,c2=1.2'")
def loop(condition_str):
    """Phase 1→2→3→4 の一括実行"""
    from forge.loop.phase1_reward import run_train_reward
    from forge.loop.phase2_surrogate import run_train_surrogate
    from forge.loop.phase3_propose import run_propose
    from forge.loop.phase4_evaluate import run_evaluate
    from forge.utils.config import check_prerequisites

    check_prerequisites(phase=1)

    console.print("\n[bold cyan]== Phase 1: Reward Model 学習 ==[/bold cyan]")
    run_train_reward()

    console.print("\n[bold cyan]== Phase 2: Surrogate Model 学習 ==[/bold cyan]")
    run_train_surrogate()

    console.print("\n[bold cyan]== Phase 3: 提案生成 ==[/bold cyan]")
    run_propose(condition_str)

    console.print("\n[bold cyan]== Phase 4: 評価入力 ==[/bold cyan]")
    run_evaluate("latest")


@cli.command()
def status():
    """現状サマリーの表示"""
    from forge.loop.status import run_status

    run_status()


@cli.command(name="__main__")
def main_compat():
    """Compatibility for python -m forge."""
    cli()


def main():
    cli()


if __name__ == "__main__":
    cli()
