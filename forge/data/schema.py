"""Schema definition and validation."""

import yaml

from forge.utils.config import SCHEMA_PATH


SCHEMA_TEMPLATE = {
    "conditions": [],
    "actions": [],
    "ranks": {
        "labels": [],
        "order": "descending",
    },
}


def validate_schema(schema: dict) -> None:
    """Validate a schema dict has all required fields."""
    if "conditions" not in schema or not schema["conditions"]:
        raise ValueError("スキーマに conditions が定義されていません")
    if "actions" not in schema or not schema["actions"]:
        raise ValueError("スキーマに actions が定義されていません")
    if "ranks" not in schema or "labels" not in schema["ranks"]:
        raise ValueError("スキーマに ranks.labels が定義されていません")

    for c in schema["conditions"]:
        for field in ("name", "type", "min", "max"):
            if field not in c:
                raise ValueError(f"Condition '{c.get('name', '?')}' に '{field}' がありません")

    for a in schema["actions"]:
        for field in ("name", "type", "min", "max"):
            if field not in a:
                raise ValueError(f"Action '{a.get('name', '?')}' に '{field}' がありません")
        if a["min"] >= a["max"]:
            raise ValueError(f"Action '{a['name']}' の min >= max です")

    if len(schema["ranks"]["labels"]) < 2:
        raise ValueError("ランクは少なくとも2つ必要です")


def schema_exists() -> bool:
    return SCHEMA_PATH.exists()
