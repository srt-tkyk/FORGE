"""Phase 4: Human evaluation input."""

from pathlib import Path

import click
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from forge.data.loader import load_dataset, save_dataset, append_row
from forge.utils.config import load_config, load_schema, PROPOSALS_DIR

console = Console()


def _load_latest_proposal() -> dict | None:
    """Load the most recent proposal file."""
    if not PROPOSALS_DIR.exists():
        return None
    proposals = sorted(PROPOSALS_DIR.glob("proposal_*.yaml"))
    if not proposals:
        return None
    with open(proposals[-1]) as f:
        return yaml.safe_load(f)


def _load_proposal(proposal_id: str) -> dict | None:
    """Load a specific proposal."""
    if proposal_id == "latest":
        return _load_latest_proposal()
    # Try to find by partial match
    if PROPOSALS_DIR.exists():
        for p in PROPOSALS_DIR.glob(f"proposal_*{proposal_id}*.yaml"):
            with open(p) as f:
                return yaml.safe_load(f)
    return None


def run_evaluate(proposal_id: str = "latest") -> None:
    """Run human evaluation: display proposal, get rank input, append to dataset."""
    config = load_config()
    schema = load_schema()

    proposal = _load_proposal(proposal_id)
    if proposal is None:
        console.print("[red]提案が見つかりません。先に `propose` を実行してください。[/red]")
        return

    # Display proposal
    table = Table(title="評価対象の提案")
    table.add_column("パラメータ", style="cyan")
    table.add_column("値", justify="right")

    for name, val in proposal["conditions"].items():
        table.add_row(f"C: {name}", f"{val:.4f}")
    for name, val in proposal["actions"].items():
        table.add_row(f"A: {name}", f"{val:.4f}")

    pred = proposal.get("prediction", {})
    if pred:
        table.add_row("---", "---")
        table.add_row("μ (予測)", f"{pred.get('mu', 'N/A')}")
        table.add_row("σ (不確実性)", f"{pred.get('sigma', 'N/A')}")
    console.print(table)

    # Get rank input
    valid_labels = config["ranks"]["labels"]
    console.print(f"\n有効なランク: {valid_labels}")

    h_rank = click.prompt(
        "ランクを入力してください",
        type=click.Choice(valid_labels, case_sensitive=True),
    )

    s_note = click.prompt("観測メモ（任意）", default="")

    # Confirmation
    console.print(f"\n  ランク: [bold]{h_rank}[/bold]")
    console.print(f"  メモ: {s_note}")
    if not click.confirm("この内容で記録しますか？", default=True):
        console.print("[yellow]キャンセルしました。[/yellow]")
        return

    # Append to dataset
    df = load_dataset()
    df = append_row(
        df,
        c_values=proposal["conditions"],
        a_values=proposal["actions"],
        h_rank=h_rank,
        s_note=s_note,
    )
    save_dataset(df)

    console.print(f"[green]データを追加しました。（合計 {len(df)} 行）[/green]")
