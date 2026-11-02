from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ["project", "language", "access", "agent"]


def feature_columns(lags: Iterable[int], rolling_windows: Iterable[int]) -> list[str]:
    calendar = [
        "day_of_week",
        "day_of_month",
        "month",
        "day_of_year_sin",
        "day_of_year_cos",
        "is_weekend",
        "trend_days",
    ]
    lag_columns = [f"lag_{lag}" for lag in lags]
    rolling = [
        name
        for window in rolling_windows
        for name in (f"rolling_mean_{window}", f"rolling_median_{window}")
    ]
    return calendar + lag_columns + rolling + CATEGORICAL_FEATURES


def make_features(
    panel: pd.DataFrame,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
) -> pd.DataFrame:
    """Create leakage-safe features; every target-derived feature is shifted."""
    result = panel.sort_values(["unique_id", "ds"]).copy()
    grouped = result.groupby("unique_id", sort=False)["y"]
    for lag in lags:
        result[f"lag_{lag}"] = grouped.shift(int(lag))
    shifted = grouped.shift(1)
    shifted_groups = shifted.groupby(result["unique_id"], sort=False)
    for window in rolling_windows:
        window = int(window)
        result[f"rolling_mean_{window}"] = shifted_groups.transform(
            lambda values, n=window: values.rolling(n, min_periods=1).mean()
        )
        result[f"rolling_median_{window}"] = shifted_groups.transform(
            lambda values, n=window: values.rolling(n, min_periods=1).median()
        )
    result["day_of_week"] = result["ds"].dt.dayofweek.astype("int8")
    result["day_of_month"] = result["ds"].dt.day.astype("int8")
    result["month"] = result["ds"].dt.month.astype("int8")
    day_of_year = result["ds"].dt.dayofyear
    result["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    result["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    result["is_weekend"] = (result["day_of_week"] >= 5).astype("int8")
    origin = result["ds"].min()
    result["trend_days"] = (result["ds"] - origin).dt.days.astype("int32")
    for column in CATEGORICAL_FEATURES:
        if column not in result:
            result[column] = "unknown"
        result[column] = result[column].fillna("unknown").astype(str)
    return result
