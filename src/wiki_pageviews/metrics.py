from __future__ import annotations

import numpy as np
import pandas as pd


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Competition-style SMAPE on a 0-100 scale.

    Terms where both truth and prediction are zero contribute zero.
    """
    actual = np.asarray(y_true, dtype=float)
    forecast = np.asarray(y_pred, dtype=float)
    denominator = np.abs(actual) + np.abs(forecast)
    terms = np.divide(
        2.0 * np.abs(forecast - actual),
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    return float(np.nanmean(terms) * 100.0)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    actual = np.asarray(y_true, dtype=float)
    forecast = np.asarray(y_pred, dtype=float)
    denominator = np.nansum(np.abs(actual))
    if denominator == 0:
        return 0.0 if np.nansum(np.abs(forecast)) == 0 else 100.0
    return float(100.0 * np.nansum(np.abs(forecast - actual)) / denominator)


def metric_summary(predictions: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    required = {"unique_id", "y", "yhat"}
    if not required.issubset(predictions):
        raise ValueError(f"Predictions require columns: {sorted(required)}")
    clean = predictions.dropna(subset=["y", "yhat"]).copy()
    per_series = (
        clean.groupby("unique_id", sort=False)
        .apply(lambda g: smape(g["y"].to_numpy(), g["yhat"].to_numpy()), include_groups=False)
        .rename("smape")
        .reset_index()
    )
    metrics = {
        "smape": smape(clean["y"].to_numpy(), clean["yhat"].to_numpy()),
        "wape": wape(clean["y"].to_numpy(), clean["yhat"].to_numpy()),
        "series_smape_median": float(per_series["smape"].median()),
        "series_smape_p90": float(per_series["smape"].quantile(0.9)),
        "n_observations": float(len(clean)),
        "n_series": float(clean["unique_id"].nunique()),
    }
    return metrics, per_series
