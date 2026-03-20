"""Data loading and manipulation for dataset.csv."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import forge.utils.config as cfg


def load_dataset() -> pd.DataFrame:
    """Load the main dataset."""
    if not cfg.DATASET_PATH.exists():
        raise RuntimeError("dataset.csv が存在しません。先に `import` を実行してください")
    return pd.read_csv(cfg.DATASET_PATH)


def save_dataset(df: pd.DataFrame) -> None:
    """Save dataset to CSV."""
    cfg.DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.DATASET_PATH, index=False)


def create_empty_dataset(schema: dict) -> pd.DataFrame:
    """Create an empty dataset with proper columns from schema."""
    c_cols = [f"C_{c['name']}" for c in schema["conditions"]]
    a_cols = [f"A_{a['name']}" for a in schema["actions"]]
    columns = ["id", "timestamp"] + c_cols + a_cols + ["h_rank", "y_hat", "s_note"]
    return pd.DataFrame(columns=columns)


def append_row(
    df: pd.DataFrame,
    c_values: dict[str, float],
    a_values: dict[str, float],
    h_rank: str,
    s_note: str = "",
) -> pd.DataFrame:
    """Append a new row to the dataset."""
    new_id = int(df["id"].max()) + 1 if len(df) > 0 else 1
    row = {
        "id": new_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **{f"C_{k}": v for k, v in c_values.items()},
        **{f"A_{k}": v for k, v in a_values.items()},
        "h_rank": h_rank,
        "y_hat": np.nan,
        "s_note": s_note,
    }
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def get_feature_matrix(df: pd.DataFrame, schema: dict | None = None) -> np.ndarray:
    """Extract [C, A] feature matrix from dataset."""
    if schema is None:
        schema = cfg.load_schema()
    c_cols = cfg.get_condition_names(schema)
    a_cols = cfg.get_action_names(schema)
    return df[c_cols + a_cols].values.astype(float)


def get_ranked_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that have a rank assigned."""
    return df[df["h_rank"].notna() & (df["h_rank"] != "")]


def get_scored_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that have y_hat assigned."""
    return df[df["y_hat"].notna()]
