"""Phase 0: Schema definition via interactive CLI."""

import click
from rich.console import Console
from rich.table import Table

from forge.data.schema import validate_schema
from forge.utils.config import load_config, save_schema, SCHEMA_PATH

console = Console()


def run_init_interactive() -> dict:
    """Run interactive schema definition and return schema dict."""
    config = load_config()
    schema = {
        "conditions": [],
        "actions": [],
        "ranks": config["ranks"],
    }

    console.print("\n[bold cyan]== FORGE スキーマ定義 ==[/bold cyan]\n")

    # Conditions
    console.print("[bold]Condition（条件）の定義[/bold]")
    console.print("制御不可な前提環境・素材特性を定義します。")
    n_c = click.prompt("Condition の次元数", type=int)
    for i in range(n_c):
        console.print(f"\n[yellow]Condition {i + 1}[/yellow]")
        name = click.prompt("  名前")
        dtype = click.prompt("  型", type=click.Choice(["float", "int"]), default="float")
        min_val = click.prompt("  最小値", type=float)
        max_val = click.prompt("  最大値", type=float)
        unit = click.prompt("  単位（任意）", default="")
        schema["conditions"].append({
            "name": name,
            "type": dtype,
            "min": min_val,
            "max": max_val,
            "unit": unit,
        })

    # Actions
    console.print("\n[bold]Action（操作量）の定義[/bold]")
    console.print("制御可能なパラメータを定義します。")
    n_a = click.prompt("Action の次元数", type=int)
    for i in range(n_a):
        console.print(f"\n[yellow]Action {i + 1}[/yellow]")
        name = click.prompt("  名前")
        dtype = click.prompt("  型", type=click.Choice(["float", "int"]), default="float")
        min_val = click.prompt("  最小値", type=float)
        max_val = click.prompt("  最大値", type=float)
        unit = click.prompt("  単位（任意）", default="")
        schema["actions"].append({
            "name": name,
            "type": dtype,
            "min": min_val,
            "max": max_val,
            "unit": unit,
        })

    # Ranks
    console.print("\n[bold]ランク定義[/bold]")
    use_default = click.confirm(
        f"  デフォルトのランク {config['ranks']['labels']} を使用しますか？", default=True
    )
    if not use_default:
        labels_str = click.prompt("  ランクラベル（カンマ区切り、最良が先頭）")
        schema["ranks"]["labels"] = [l.strip() for l in labels_str.split(",")]
        schema["ranks"]["order"] = "descending"

    validate_schema(schema)

    # Display summary
    _display_schema_summary(schema)

    if click.confirm("\nこの内容で保存しますか？", default=True):
        save_schema(schema)
        console.print(f"\n[green]スキーマを {SCHEMA_PATH} に保存しました。[/green]")
    else:
        console.print("[yellow]キャンセルしました。[/yellow]")
        raise click.Abort()

    return schema


def _display_schema_summary(schema: dict) -> None:
    """Display schema summary as rich table."""
    console.print("\n[bold cyan]== スキーマ概要 ==[/bold cyan]")

    # Conditions table
    table = Table(title="Conditions (C)")
    table.add_column("名前", style="cyan")
    table.add_column("型")
    table.add_column("最小値", justify="right")
    table.add_column("最大値", justify="right")
    table.add_column("単位")
    for c in schema["conditions"]:
        table.add_row(c["name"], c["type"], str(c["min"]), str(c["max"]), c.get("unit", ""))
    console.print(table)

    # Actions table
    table = Table(title="Actions (A)")
    table.add_column("名前", style="green")
    table.add_column("型")
    table.add_column("最小値", justify="right")
    table.add_column("最大値", justify="right")
    table.add_column("単位")
    for a in schema["actions"]:
        table.add_row(a["name"], a["type"], str(a["min"]), str(a["max"]), a.get("unit", ""))
    console.print(table)

    # Ranks
    console.print(f"\n[bold]ランク:[/bold] {schema['ranks']['labels']} ({schema['ranks']['order']})")
