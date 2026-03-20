"""Phase 3: Generate next experiment proposal."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from forge.data.loader import load_dataset, get_scored_rows, get_feature_matrix
from forge.models.surrogate_model import SurrogateModel
from forge.optimization.optimizer import optimize_acquisition
from forge.utils.config import (
    load_config,
    load_schema,
    get_action_bounds,
    get_action_names,
    get_condition_names,
    SURROGATE_MODEL_PATH,
    PROPOSALS_DIR,
)

console = Console()


def _parse_condition_str(condition_str: str, schema: dict) -> np.ndarray:
    """Parse 'c1=0.5,c2=1.2' into condition vector."""
    pairs = {}
    for pair in condition_str.split(","):
        k, v = pair.strip().split("=")
        pairs[k.strip()] = float(v.strip())

    c_names = [c["name"] for c in schema["conditions"]]
    c_vec = []
    for name in c_names:
        if name in pairs:
            c_vec.append(pairs[name])
        else:
            raise ValueError(f"Condition '{name}' が指定されていません。必要: {c_names}")
    return np.array(c_vec)


def run_propose(condition_str: str | None = None) -> dict | None:
    """Generate a proposal for the next experiment."""
    config = load_config()
    schema = load_schema()
    df = load_dataset()

    # Load surrogate
    surrogate = SurrogateModel()
    surrogate.load(SURROGATE_MODEL_PATH)

    # Get condition vector
    if condition_str:
        c_vec = _parse_condition_str(condition_str, schema)
    else:
        # Use mean of existing conditions
        scored = get_scored_rows(df)
        if len(scored) == 0:
            console.print("[red]データがありません。[/red]")
            return None
        c_cols = get_condition_names(schema)
        c_vec = scored[c_cols].mean().values

    bounds = get_action_bounds(schema)
    acq_cfg = config["acquisition"]
    opt_cfg = config["optimizer"]

    # y_best for EI
    scored = get_scored_rows(df)
    y_best = scored["y_hat"].max() if len(scored) > 0 else 0.0

    result = optimize_acquisition(
        surrogate_model=surrogate,
        c_vec=c_vec,
        bounds=bounds,
        acq_func=acq_cfg["function"],
        kappa=acq_cfg.get("kappa", 2.0),
        y_best=y_best,
        method_kwargs={
            "maxiter": opt_cfg.get("max_iter", 1000),
            "popsize": opt_cfg.get("popsize", 15),
            "seed": opt_cfg.get("seed", 42),
        },
    )

    # Build proposal
    a_names = [a["name"] for a in schema["actions"]]
    c_names = [c["name"] for c in schema["conditions"]]

    proposal = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conditions": {name: float(val) for name, val in zip(c_names, c_vec)},
        "actions": {name: float(val) for name, val in zip(a_names, result["a_vec"])},
        "prediction": {
            "mu": float(result["mu"]),
            "sigma": float(result["sigma"]),
            "alpha": float(result["alpha"]),
            "acquisition_function": acq_cfg["function"],
        },
    }

    # Save proposal
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    proposal_path = PROPOSALS_DIR / f"proposal_{ts}.yaml"
    with open(proposal_path, "w") as f:
        yaml.dump(proposal, f, default_flow_style=False, allow_unicode=True)

    # Display
    table = Table(title="提案: 次の実験条件")
    table.add_column("パラメータ", style="cyan")
    table.add_column("値", justify="right")

    for name, val in zip(c_names, c_vec):
        table.add_row(f"C: {name}", f"{val:.4f}")
    for name, val in zip(a_names, result["a_vec"]):
        table.add_row(f"A: {name}", f"{val:.4f}")

    table.add_row("---", "---")
    table.add_row("μ (予測平均)", f"{result['mu']:.4f}")
    table.add_row("σ (不確実性)", f"{result['sigma']:.4f}")
    table.add_row(f"α ({acq_cfg['function'].upper()})", f"{result['alpha']:.4f}")
    console.print(table)
    console.print(f"[green]提案を {proposal_path} に保存しました。[/green]")

    return proposal
