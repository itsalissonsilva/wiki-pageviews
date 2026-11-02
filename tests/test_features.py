import numpy as np
import pandas as pd

from wiki_pageviews.features import make_features


def test_lags_and_rolls_do_not_include_current_target():
    panel = pd.DataFrame(
        {
            "unique_id": ["a"] * 5,
            "ds": pd.date_range("2024-01-01", periods=5),
            "y": [1.0, 2.0, 3.0, 1000.0, 5.0],
        }
    )
    featured = make_features(panel, lags=[1], rolling_windows=[2])
    row = featured.iloc[3]
    assert row["lag_1"] == 3.0
    assert row["rolling_mean_2"] == 2.5
    assert row["rolling_median_2"] == 2.5


def test_series_features_do_not_cross_boundaries():
    panel = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
            "y": [10.0, 11.0, 100.0, 101.0],
        }
    )
    featured = make_features(panel, lags=[1], rolling_windows=[2])
    first_b = featured[featured["unique_id"] == "b"].iloc[0]
    assert np.isnan(first_b["lag_1"])
    assert np.isnan(first_b["rolling_mean_2"])
