from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from wiki_pageviews.features import CATEGORICAL_FEATURES, feature_columns, make_features


@dataclass
class GlobalForecaster:
    lags: list[int]
    rolling_windows: list[int]
    params: dict[str, Any]
    seed: int = 42

    def __post_init__(self) -> None:
        self.columns = feature_columns(self.lags, self.rolling_windows)
        numeric = [column for column in self.columns if column not in CATEGORICAL_FEATURES]
        preprocessor = ColumnTransformer(
            [
                (
                    "categorical",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    CATEGORICAL_FEATURES,
                ),
                ("numeric", "passthrough", numeric),
            ],
            verbose_feature_names_out=False,
        )
        defaults = {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 63,
            "l2_regularization": 1.0,
            "min_samples_leaf": 30,
        }
        defaults.update(self.params)
        estimator = HistGradientBoostingRegressor(random_state=self.seed, **defaults)
        self.pipeline = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])

    def fit(self, train: pd.DataFrame) -> GlobalForecaster:
        featured = make_features(train, self.lags, self.rolling_windows)
        # The longest lag is required so all rows are comparable across feature sets.
        featured = featured.dropna(subset=[f"lag_{max(self.lags)}", "y"])
        if featured.empty:
            raise ValueError(f"No training rows remain after requiring lag {max(self.lags)}")
        self.pipeline.fit(featured[self.columns], np.log1p(featured["y"].clip(lower=0)))
        self.training_rows_ = len(featured)
        return self

    def predict(self, train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        """Recursive multi-step forecast using only history and earlier predictions."""
        history = train.copy()
        output_rows: list[pd.DataFrame] = []
        metadata_columns = [column for column in CATEGORICAL_FEATURES if column in future]
        for date in sorted(pd.to_datetime(future["ds"].unique())):
            requested = future[future["ds"] == date].copy()
            new_rows = requested[["unique_id", "ds", *metadata_columns]].copy()
            # A placeholder cannot affect features for this date because all
            # target-derived features are shifted. Avoid an all-NA concat here;
            # pandas is deprecating its current dtype inference for that case.
            new_rows["y"] = 0.0
            combined = pd.concat([history, new_rows], ignore_index=True, sort=False)
            featured = make_features(combined, self.lags, self.rolling_windows)
            current = featured[featured["ds"] == date].copy()
            prediction = np.expm1(self.pipeline.predict(current[self.columns]))
            current_predictions = current[["unique_id", "ds"]].copy()
            current_predictions["yhat"] = np.clip(prediction, 0.0, None)
            requested = requested.merge(current_predictions, on=["unique_id", "ds"], how="left")
            output_rows.append(requested)
            appended = new_rows.merge(current_predictions, on=["unique_id", "ds"], how="left")
            appended["y"] = appended.pop("yhat")
            history = pd.concat([history, appended], ignore_index=True, sort=False)
        return pd.concat(output_rows, ignore_index=True)

    def transformed_feature_names(self) -> list[str]:
        return list(self.pipeline.named_steps["preprocessor"].get_feature_names_out())
