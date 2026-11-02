from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from wiki_pageviews.data import dataset_fingerprint, fill_training_values, load_panel
from wiki_pageviews.metrics import metric_summary
from wiki_pageviews.tracking import configure_tracking, log_dataframe, log_run_context


def run_nhits(config: dict[str, Any]) -> dict[str, float]:
    """Run the optional N-HiTS benchmark with NeuralForecast cross-validation."""
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('N-HiTS requires: pip install -e ".[neural]"') from exc

    seed = int(config["experiment"].get("seed", 42))
    panel = load_panel(config["data"]["path"], config["data"].get("max_series"), seed)
    panel = fill_training_values(panel, config["data"].get("fill_method", "zero"))
    model_cfg = config["model"]
    validation = config["validation"]
    model = NHITS(
        h=int(validation["horizon"]),
        input_size=int(model_cfg["input_size"]),
        max_steps=int(model_cfg["max_steps"]),
        learning_rate=float(model_cfg["learning_rate"]),
        batch_size=int(model_cfg["batch_size"]),
        windows_batch_size=int(model_cfg["windows_batch_size"]),
        scaler_type=model_cfg.get("scaler_type", "robust"),
        random_seed=seed,
    )
    nf = NeuralForecast(models=[model], freq="D")
    predictions = nf.cross_validation(
        df=panel[["unique_id", "ds", "y"]],
        n_windows=int(validation["n_folds"]),
        step_size=int(validation["step"]),
        refit=True,
    ).rename(columns={"NHITS": "yhat"})
    metrics, per_series = metric_summary(predictions)
    fold_rows = []
    for number, (_, fold) in enumerate(predictions.groupby("cutoff", sort=True)):
        fold_metrics, _ = metric_summary(fold)
        fold_rows.append({"fold": number, **fold_metrics})
    mlflow = configure_tracking(config)
    info = {
        "fingerprint": dataset_fingerprint(config["data"]["path"], panel),
        "rows": len(panel),
        "series": panel["unique_id"].nunique(),
        "start": str(panel["ds"].min().date()),
        "end": str(panel["ds"].max().date()),
    }
    output_dir = Path(config["output"]["directory"])
    model_dir = output_dir / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    nf.save(path=str(model_dir), overwrite=True, save_dataset=False)
    predictions.to_parquet(output_dir / "backtest_predictions.parquet", index=False)
    with mlflow.start_run(run_name="nhits-cross-validation"):
        log_run_context(mlflow, config, info)
        mlflow.log_metrics(metrics)
        log_dataframe(mlflow, pd.DataFrame(fold_rows), "metrics/folds.csv")
        log_dataframe(mlflow, per_series, "metrics/per_series.csv")
        log_dataframe(mlflow, predictions, "predictions/backtest.csv")
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
    return metrics
