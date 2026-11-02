from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from wiki_pageviews.config import flattened


def require_mlflow():
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "MLflow is required. Install the project with: pip install -e ."
        ) from exc
    return mlflow


def configure_tracking(config: dict[str, Any]):
    # Keep MLflow's temporary artifact staging inside the project. This is also
    # friendlier to locked-down Windows runners where the system temp directory
    # may not allow chmod/cleanup operations used by tempfile.
    temp_dir = Path(".tmp/mlflow").resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMP", str(temp_dir))
    os.environ.setdefault("TEMP", str(temp_dir))
    tempfile.tempdir = str(temp_dir)
    mlflow = require_mlflow()
    uri = str(config["tracking"].get("uri", "./mlruns"))
    if "://" not in uri:
        uri = Path(uri).resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config["experiment"]["name"])
    return mlflow


def log_run_context(mlflow, config: dict[str, Any], dataset_info: dict[str, Any]) -> None:
    mlflow.log_params(flattened(config))
    mlflow.log_params({f"dataset.{key}": value for key, value in dataset_info.items()})
    mlflow.set_tags(config["experiment"].get("tags", {}))
    mlflow.set_tags(
        {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": os.getenv("GIT_COMMIT", "unknown"),
        }
    )
    mlflow.log_dict(config, "config/resolved_config.json")


def log_dataframe(mlflow, frame: pd.DataFrame, artifact_file: str) -> None:
    """Log a dataframe without relying on MLflow dataset API version details."""
    mlflow.log_text(frame.to_csv(index=False), artifact_file)


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
