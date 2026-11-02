from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _fallback(history: np.ndarray) -> float:
    finite = history[np.isfinite(history)]
    return float(np.median(finite)) if len(finite) else 0.0


def _series_forecast(history: np.ndarray, horizon: int, spec: dict[str, Any]) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    name = spec["name"]
    fallback = _fallback(values)
    if name == "last_value":
        finite = values[np.isfinite(values)]
        value = float(finite[-1]) if len(finite) else fallback
        return np.full(horizon, value)
    if name == "median":
        return np.full(horizon, fallback)
    if name == "rolling_median":
        window = int(spec.get("window", 28))
        return np.full(horizon, _fallback(values[-window:]))
    if name == "seasonal_naive":
        period = int(spec.get("period", 7))
        if len(values) < period:
            return np.full(horizon, fallback)
        pattern = values[-period:].copy()
        pattern[~np.isfinite(pattern)] = fallback
        return np.resize(pattern, horizon)
    raise ValueError(f"Unknown baseline model: {name}")


def model_label(spec: dict[str, Any]) -> str:
    extras = "-".join(f"{key}{value}" for key, value in spec.items() if key != "name")
    return spec["name"] if not extras else f"{spec['name']}-{extras}"


def forecast_baseline(
    train: pd.DataFrame,
    future: pd.DataFrame,
    spec: dict[str, Any],
) -> pd.DataFrame:
    """Forecast exactly the requested future rows, preserving their dates and metadata."""
    outputs: list[pd.DataFrame] = []
    history_by_id = {
        key: group["y"].to_numpy() for key, group in train.groupby("unique_id", sort=False)
    }
    for unique_id, rows in future.groupby("unique_id", sort=False):
        ordered = rows.sort_values("ds").copy()
        history = history_by_id.get(unique_id, np.array([], dtype=float))
        ordered["yhat"] = np.clip(_series_forecast(history, len(ordered), spec), 0.0, None)
        outputs.append(ordered)
    return pd.concat(outputs, ignore_index=True) if outputs else future.assign(yhat=np.nan)
