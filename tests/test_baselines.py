import pandas as pd

from wiki_pageviews.models.baselines import forecast_baseline


def test_seasonal_naive_repeats_last_period():
    train = pd.DataFrame(
        {
            "unique_id": ["a"] * 4,
            "ds": pd.date_range("2024-01-01", periods=4),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    future = pd.DataFrame(
        {"unique_id": ["a"] * 5, "ds": pd.date_range("2024-01-05", periods=5), "y": [0.0] * 5}
    )
    result = forecast_baseline(train, future, {"name": "seasonal_naive", "period": 2})
    assert result["yhat"].tolist() == [3.0, 4.0, 3.0, 4.0, 3.0]
