"""Generate sample_data.csv fixture: 2 conditions, 2 actions, ranks A/B/C."""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path


def generate_sample_data(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic sample data."""
    rng = np.random.default_rng(seed)

    c1 = rng.uniform(0, 100, n)
    c2 = rng.uniform(0, 10, n)
    a1 = rng.uniform(100, 500, n)
    a2 = rng.uniform(0, 1, n)

    # Generate a hidden score and assign ranks based on it
    score = 0.3 * c1 + 1.5 * c2 + 0.01 * a1 + 5 * a2 + rng.normal(0, 3, n)
    ranks = pd.qcut(score, q=3, labels=["C", "B", "A"])

    df = pd.DataFrame({
        "id": range(1, n + 1),
        "timestamp": [datetime.now(timezone.utc).isoformat()] * n,
        "C_temp": c1,
        "C_pressure": c2,
        "A_speed": a1,
        "A_ratio": a2,
        "h_rank": ranks,
        "y_hat": np.nan,
        "s_note": "",
    })
    return df


def generate_schema() -> dict:
    """Generate matching schema for the sample data."""
    return {
        "conditions": [
            {"name": "temp", "type": "float", "min": 0.0, "max": 100.0, "unit": "°C"},
            {"name": "pressure", "type": "float", "min": 0.0, "max": 10.0, "unit": "MPa"},
        ],
        "actions": [
            {"name": "speed", "type": "float", "min": 100.0, "max": 500.0, "unit": "rpm"},
            {"name": "ratio", "type": "float", "min": 0.0, "max": 1.0, "unit": ""},
        ],
        "ranks": {
            "labels": ["A", "B", "C"],
            "order": "descending",
        },
    }


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    df = generate_sample_data()
    df.to_csv(fixtures_dir / "sample_data.csv", index=False)
    print(f"Generated {len(df)} rows -> {fixtures_dir / 'sample_data.csv'}")
