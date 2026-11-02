from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    number: int
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def rolling_origin_folds(
    dates: pd.Series,
    horizon: int,
    n_folds: int,
    step: int,
    min_train_days: int,
) -> list[Fold]:
    unique_dates = pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    required = min_train_days + horizon + (n_folds - 1) * step
    if len(unique_dates) < required:
        raise ValueError(f"Need at least {required} distinct dates; found {len(unique_dates)}")
    folds: list[Fold] = []
    for chronological_number, offset in enumerate(reversed(range(n_folds))):
        valid_end_idx = len(unique_dates) - 1 - offset * step
        valid_start_idx = valid_end_idx - horizon + 1
        train_end_idx = valid_start_idx - 1
        folds.append(
            Fold(
                number=chronological_number,
                train_end=unique_dates[train_end_idx],
                valid_start=unique_dates[valid_start_idx],
                valid_end=unique_dates[valid_end_idx],
            )
        )
    return folds


def split_panel(panel: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = panel[panel["ds"] <= fold.train_end].copy()
    valid = panel[(panel["ds"] >= fold.valid_start) & (panel["ds"] <= fold.valid_end)].copy()
    return train, valid
