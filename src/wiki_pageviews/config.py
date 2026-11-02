from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML and apply dotted ``key=value`` command-line overrides."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config = deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, received: {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        cursor = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot set nested value below {part!r}")
        cursor[parts[-1]] = yaml.safe_load(raw_value)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = ["experiment", "tracking", "data", "validation", "output"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(missing)}")
    validation = config["validation"]
    for key in ("horizon", "n_folds", "step", "min_train_days"):
        if int(validation[key]) <= 0:
            raise ValueError(f"validation.{key} must be positive")


def flattened(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten configuration into MLflow-safe parameter values."""
    result: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flattened(value, name))
        elif isinstance(value, list):
            result[name] = json.dumps(value, sort_keys=True)
        elif value is None:
            result[name] = "null"
        else:
            result[name] = value
    return result
