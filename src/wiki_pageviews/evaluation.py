from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from wiki_pageviews.data import fill_training_values
from wiki_pageviews.metrics import metric_summary
from wiki_pageviews.splits import Fold, split_panel

ForecastFunction = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def backtest(
    panel: pd.DataFrame,
    folds: Sequence[Fold],
    build_forecaster: Callable[[], Any] | None = None,
    forecast_function: ForecastFunction | None = None,
    fill_method: str = "zero",
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    if (build_forecaster is None) == (forecast_function is None):
        raise ValueError("Provide exactly one of build_forecaster or forecast_function")
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float]] = []
    for fold in folds:
        raw_train, valid = split_panel(panel, fold)
        train = fill_training_values(raw_train, fill_method)
        if build_forecaster is not None:
            forecaster = build_forecaster()
            forecaster.fit(train)
            predicted = forecaster.predict(train, valid)
        else:
            predicted = forecast_function(train, valid)  # type: ignore[misc]
        predicted["fold"] = fold.number
        metrics, _ = metric_summary(predicted)
        metrics["fold"] = float(fold.number)
        metric_rows.append(metrics)
        prediction_parts.append(predicted)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    fold_metrics = pd.DataFrame(metric_rows)
    aggregate = {
        f"cv_{column}_mean": float(fold_metrics[column].mean())
        for column in ["smape", "wape", "series_smape_median", "series_smape_p90"]
    }
    aggregate["cv_smape_std"] = float(fold_metrics["smape"].std(ddof=0))
    return aggregate, fold_metrics, predictions


def slice_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dimension in ["language", "access", "agent"]:
        if dimension not in predictions:
            continue
        for value, group in predictions.groupby(dimension, dropna=False):
            metrics, _ = metric_summary(group)
            rows.append({"dimension": dimension, "value": value, **metrics})
    return pd.DataFrame(rows)
