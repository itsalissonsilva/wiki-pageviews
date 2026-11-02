import pandas as pd

from wiki_pageviews.splits import rolling_origin_folds


def test_rolling_folds_are_chronological_and_fixed_horizon():
    dates = pd.Series(pd.date_range("2024-01-01", periods=100))
    folds = rolling_origin_folds(dates, horizon=10, n_folds=3, step=10, min_train_days=50)
    assert len(folds) == 3
    assert folds[0].valid_start < folds[1].valid_start < folds[2].valid_start
    assert all((fold.valid_end - fold.valid_start).days == 9 for fold in folds)
    assert all(fold.train_end < fold.valid_start for fold in folds)
