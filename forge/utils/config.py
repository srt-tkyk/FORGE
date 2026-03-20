"""Configuration and schema loading utilities."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
SCHEMA_PATH = PROJECT_ROOT / "data" / "schema.yaml"
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"
REWARD_MODEL_PATH = PROJECT_ROOT / "models" / "reward_model.pkl"
SURROGATE_MODEL_PATH = PROJECT_ROOT / "models" / "surrogate_model.pt"
PROPOSALS_DIR = PROJECT_ROOT / "data" / "proposals"


def load_config() -> dict:
    """Load config/default.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_schema() -> dict:
    """Load data/schema.yaml."""
    if not SCHEMA_PATH.exists():
        raise RuntimeError("schema.yaml が存在しません。先に `init` を実行してください")
    with open(SCHEMA_PATH) as f:
        return yaml.safe_load(f)


def save_schema(schema: dict) -> None:
    """Save data/schema.yaml."""
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCHEMA_PATH, "w") as f:
        yaml.dump(schema, f, default_flow_style=False, allow_unicode=True)


def get_rank_mapping(config: dict | None = None) -> dict[str, int]:
    """Build rank label -> integer mapping from config."""
    if config is None:
        config = load_config()
    labels = config["ranks"]["labels"]
    return {label: i for i, label in enumerate(labels)}


def get_action_bounds(schema: dict | None = None) -> list[tuple[float, float]]:
    """Build action bounds list from schema."""
    if schema is None:
        schema = load_schema()
    return [(a["min"], a["max"]) for a in schema["actions"]]


def get_condition_names(schema: dict | None = None) -> list[str]:
    """Get condition column names (C_ prefixed)."""
    if schema is None:
        schema = load_schema()
    return [f"C_{c['name']}" for c in schema["conditions"]]


def get_action_names(schema: dict | None = None) -> list[str]:
    """Get action column names (A_ prefixed)."""
    if schema is None:
        schema = load_schema()
    return [f"A_{a['name']}" for a in schema["actions"]]


def check_prerequisites(phase: int) -> None:
    """Check that prerequisites for the given phase are met."""
    if phase >= 1 and not SCHEMA_PATH.exists():
        raise RuntimeError("schema.yaml が存在しません。先に `init` を実行してください")
    if phase >= 1 and not DATASET_PATH.exists():
        raise RuntimeError("dataset.csv が存在しません。先に `import` を実行してください")
    if phase >= 2 and not REWARD_MODEL_PATH.exists():
        raise RuntimeError("Reward Model が存在しません。先に `train-reward` を実行してください")
