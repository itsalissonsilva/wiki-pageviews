from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wiki_pageviews.data import dataset_fingerprint, fill_training_values, load_panel
from wiki_pageviews.evaluation import backtest, slice_metrics
from wiki_pageviews.features import feature_columns, make_features
from wiki_pageviews.models.baselines import forecast_baseline, model_label
from wiki_pageviews.models.global_model import GlobalForecaster
from wiki_pageviews.splits import rolling_origin_folds, split_panel
from wiki_pageviews.tracking import configure_tracking, log_dataframe, log_run_context, save_json


def _load(config: dict[str, Any]) -> tuple[pd.DataFrame, list, dict[str, Any]]:
    data_cfg = config["data"]
    seed = int(config["experiment"].get("seed", 42))
    panel = load_panel(data_cfg["path"], data_cfg.get("max_series"), seed)
    validation = config["validation"]
    folds = rolling_origin_folds(
        panel["ds"],
        int(validation["horizon"]),
        int(validation["n_folds"]),
        int(validation["step"]),
        int(validation["min_train_days"]),
    )
    info = {
        "fingerprint": dataset_fingerprint(data_cfg["path"], panel),
        "rows": len(panel),
        "series": panel["unique_id"].nunique(),
        "start": str(panel["ds"].min().date()),
        "end": str(panel["ds"].max().date()),
    }
    return panel, folds, info


def _global_factory(config: dict[str, Any], params: dict[str, Any] | None = None):
    model_cfg = config["models"]["global_model"]
    resolved = dict(model_cfg.get("params", {}))
    resolved.update(params or {})
    return GlobalForecaster(
        lags=[int(value) for value in config["features"]["lags"]],
        rolling_windows=[int(value) for value in config["features"]["rolling_windows"]],
        params=resolved,
        seed=int(config["experiment"].get("seed", 42)),
    )


def _fold_frame(folds) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": fold.number,
                "train_end": fold.train_end,
                "valid_start": fold.valid_start,
                "valid_end": fold.valid_end,
            }
            for fold in folds
        ]
    )


def _log_backtest(mlflow, metrics, fold_metrics, predictions) -> None:
    mlflow.log_metrics(metrics)
    log_dataframe(mlflow, fold_metrics, "metrics/folds.csv")
    log_dataframe(mlflow, slice_metrics(predictions), "metrics/slices.csv")
    log_dataframe(mlflow, predictions, "predictions/backtest.csv")


def _fit_and_log_global_model(mlflow, config, panel, params, fill_method) -> None:
    """Fit a deployable full-history model and record its preprocessing contract."""
    final_model = _global_factory(config, params).fit(fill_training_values(panel, fill_method))
    mlflow.sklearn.log_model(
        final_model.pipeline,
        name="model",
        serialization_format="cloudpickle",
    )
    mlflow.log_dict(
        {
            "input_columns": final_model.columns,
            "transformed_columns": final_model.transformed_feature_names(),
            "training_rows": final_model.training_rows_,
        },
        "model/feature_contract.json",
    )


def run_experiment(config: dict[str, Any]) -> dict[str, float]:
    mlflow = configure_tracking(config)
    panel, folds, info = _load(config)
    fill_method = config["data"].get("fill_method", "zero")
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    scoreboard: list[dict[str, Any]] = []
    with mlflow.start_run(run_name="comparison"):
        log_run_context(mlflow, config, info)
        log_dataframe(mlflow, _fold_frame(folds), "validation/fold_boundaries.csv")
        for spec in config.get("models", {}).get("baselines", []):
            label = model_label(spec)
            with mlflow.start_run(run_name=label, nested=True):
                metrics, fold_metrics, predictions = backtest(
                    panel,
                    folds,
                    forecast_function=partial(forecast_baseline, spec=spec),
                    fill_method=fill_method,
                )
                mlflow.log_params({f"model.{key}": value for key, value in spec.items()})
                _log_backtest(mlflow, metrics, fold_metrics, predictions)
                scoreboard.append({"model": label, **metrics})
        global_cfg = config.get("models", {}).get("global_model", {})
        if global_cfg.get("enabled", False):
            with mlflow.start_run(run_name="global-hist-gradient-boosting", nested=True):
                metrics, fold_metrics, predictions = backtest(
                    panel,
                    folds,
                    build_forecaster=lambda: _global_factory(config),
                    fill_method=fill_method,
                )
                mlflow.log_params(
                    {f"model.{key}": value for key, value in global_cfg.get("params", {}).items()}
                )
                _log_backtest(mlflow, metrics, fold_metrics, predictions)
                _fit_and_log_global_model(mlflow, config, panel, {}, fill_method)
                scoreboard.append({"model": "global-hgb", **metrics})
        score_frame = pd.DataFrame(scoreboard).sort_values("cv_smape_mean")
        log_dataframe(mlflow, score_frame, "metrics/scoreboard.csv")
        score_path = output_dir / "scoreboard.csv"
        score_frame.to_csv(score_path, index=False)
    return {str(row["model"]): float(row["cv_smape_mean"]) for row in scoreboard}


def _suggest_params(trial, search_space: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", *search_space["learning_rate"], log=True
        ),
        "max_iter": trial.suggest_int("max_iter", *search_space["max_iter"], step=25),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", *search_space["max_leaf_nodes"]),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", *search_space["l2_regularization"], log=True
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", *search_space["min_samples_leaf"]
        ),
    }


def tune_experiment(config: dict[str, Any]) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Optuna is required. Install the project with: pip install -e ."
        ) from exc
    mlflow = configure_tracking(config)
    panel, folds, info = _load(config)
    if len(folds) < 2:
        raise ValueError("Tuning requires at least two folds: inner CV plus final holdout")
    inner_folds, holdout = folds[:-1], folds[-1:]
    tuning = config["tuning"]
    search_space = tuning["search_space"]
    fill_method = config["data"].get("fill_method", "zero")
    seed = int(config["experiment"].get("seed", 42))
    with mlflow.start_run(run_name="optuna-tuning"):
        log_run_context(mlflow, config, info)
        log_dataframe(mlflow, _fold_frame(folds), "validation/fold_boundaries.csv")

        def objective(trial) -> float:
            params = _suggest_params(trial, search_space)
            with mlflow.start_run(run_name=f"trial-{trial.number:04d}", nested=True):
                metrics, fold_metrics, _ = backtest(
                    panel,
                    inner_folds,
                    build_forecaster=lambda: _global_factory(config, params),
                    fill_method=fill_method,
                )
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                log_dataframe(mlflow, fold_metrics, "metrics/folds.csv")
                return float(metrics["cv_smape_mean"])

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(
            objective,
            n_trials=int(tuning["n_trials"]),
            timeout=int(tuning.get("timeout_seconds", 0)) or None,
            n_jobs=1,
        )
        best_params = study.best_params
        log_dataframe(mlflow, study.trials_dataframe(), "selection/optuna_trials.csv")
        mlflow.log_params({f"best.{key}": value for key, value in best_params.items()})
        mlflow.log_metric("best_inner_cv_smape", study.best_value)
        holdout_metrics, holdout_folds, holdout_predictions = backtest(
            panel,
            holdout,
            build_forecaster=lambda: _global_factory(config, best_params),
            fill_method=fill_method,
        )
        mlflow.log_metrics({f"holdout_{key}": value for key, value in holdout_metrics.items()})
        log_dataframe(mlflow, holdout_folds, "metrics/final_holdout.csv")
        log_dataframe(mlflow, holdout_predictions, "predictions/final_holdout.csv")
        _fit_and_log_global_model(mlflow, config, panel, best_params, fill_method)
        output = {
            "best_params": best_params,
            "best_inner_cv_smape": study.best_value,
            "final_holdout_smape": holdout_metrics["cv_smape_mean"],
            "n_completed_trials": len(study.trials),
        }
        output_path = save_json(output, Path(config["output"]["directory"]) / "best_params.json")
        mlflow.log_artifact(str(output_path), artifact_path="selection")
    return output


def explain_experiment(config: dict[str, Any]) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    mlflow = configure_tracking(config)
    panel, folds, info = _load(config)
    params_path = Path(config["output"]["directory"]) / "best_params.json"
    params: dict[str, Any] = {}
    if params_path.exists():
        import json

        params = json.loads(params_path.read_text(encoding="utf-8"))["best_params"]
    train, valid = split_panel(panel, folds[-1])
    train = fill_training_values(train, config["data"].get("fill_method", "zero"))
    model = _global_factory(config, params).fit(train)
    combined = pd.concat([train, valid], ignore_index=True, sort=False)
    featured = make_features(combined, model.lags, model.rolling_windows)
    validation_rows = featured[
        (featured["ds"] >= folds[-1].valid_start) & (featured["ds"] <= folds[-1].valid_end)
    ]
    validation_rows = validation_rows.dropna(subset=["y"])
    if len(validation_rows) > 50_000:
        validation_rows = validation_rows.sample(50_000, random_state=model.seed)
    x_valid = validation_rows[feature_columns(model.lags, model.rolling_windows)]
    y_valid = np.log1p(validation_rows["y"].clip(lower=0))
    result = permutation_importance(
        model.pipeline,
        x_valid,
        y_valid,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=model.seed,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": x_valid.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    with mlflow.start_run(run_name="feature-importance"):
        log_run_context(mlflow, config, info)
        log_dataframe(mlflow, _fold_frame(folds), "validation/fold_boundaries.csv")
        mlflow.log_params({f"model.{key}": value for key, value in params.items()})
        log_dataframe(mlflow, importance, "explanations/permutation_importance.csv")
        mlflow.log_metrics(
            {
                f"importance_{row.feature}": float(row.importance_mean)
                for row in importance.head(20).itertuples()
            }
        )
    output_path = Path(config["output"]["directory"]) / "permutation_importance.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)
    return importance
