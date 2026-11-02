from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

METADATA_COLUMNS = ["page", "project", "language", "access", "agent"]


def parse_page_id(value: str) -> dict[str, str]:
    """Parse the three stable suffix fields from a competition page identifier."""
    parts = str(value).rsplit("_", 3)
    if len(parts) != 4:
        return {
            "page": str(value),
            "project": "unknown",
            "language": "unknown",
            "access": "unknown",
            "agent": "unknown",
        }
    page, project, access, agent = parts
    language = project.split(".", 1)[0] if "." in project else project
    return {
        "page": page,
        "project": project,
        "language": language,
        "access": access,
        "agent": agent,
    }


def prepare_wide_csv(
    input_path: str | Path,
    output_path: str | Path,
    max_series: int | None = None,
    seed: int = 42,
    chunksize: int = 2_000,
) -> Path:
    """Stream Kaggle's wide CSV into long Parquet row groups."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = pd.read_csv(input_path, nrows=0)
    if "Page" not in header.columns:
        raise ValueError("Expected a 'Page' column in the challenge CSV")

    selected: set[int] | None = None
    if max_series is not None:
        row_count = sum(
            len(chunk) for chunk in pd.read_csv(input_path, usecols=["Page"], chunksize=50_000)
        )
        take = min(max_series, row_count)
        selected = set(
            np.random.default_rng(seed).choice(row_count, size=take, replace=False).tolist()
        )

    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    offset = 0
    try:
        for chunk in pd.read_csv(input_path, chunksize=chunksize):
            original_length = len(chunk)
            if selected is not None:
                mask = [offset + i in selected for i in range(original_length)]
                chunk = chunk.loc[mask]
            offset += original_length
            if chunk.empty:
                continue
            long = chunk.melt(id_vars="Page", var_name="ds", value_name="y")
            long = long.rename(columns={"Page": "unique_id"})
            long["ds"] = pd.to_datetime(long["ds"], errors="raise")
            long["y"] = pd.to_numeric(long["y"], errors="coerce").astype("float32")
            unique_ids = long["unique_id"].drop_duplicates()
            metadata = unique_ids.map(parse_page_id).apply(pd.Series)
            metadata.index = unique_ids.values
            long = long.join(metadata, on="unique_id")
            table = pa.Table.from_pandas(long, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary_path, table.schema, compression="zstd")
            writer.write_table(table)
            rows_written += len(long)
    finally:
        if writer is not None:
            writer.close()
    if rows_written == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("No rows were selected from the input dataset")
    temporary_path.replace(output_path)
    return output_path


def load_panel(path: str | Path, max_series: int | None = None, seed: int = 42) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        panel = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        panel = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported data format: {path.suffix}")
    required = {"unique_id", "ds", "y"}
    if not required.issubset(panel):
        raise ValueError(f"Dataset requires columns: {sorted(required)}")
    panel["ds"] = pd.to_datetime(panel["ds"], errors="raise")
    panel["y"] = pd.to_numeric(panel["y"], errors="coerce")
    if max_series is not None and panel["unique_id"].nunique() > max_series:
        ids = panel["unique_id"].drop_duplicates().sample(n=max_series, random_state=seed)
        panel = panel[panel["unique_id"].isin(ids)]
    for column in METADATA_COLUMNS:
        if column not in panel:
            mapping = panel["unique_id"].drop_duplicates().map(parse_page_id).apply(pd.Series)
            mapping.index = panel["unique_id"].drop_duplicates().values
            panel[column] = panel["unique_id"].map(mapping[column])
    return panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def fill_training_values(panel: pd.DataFrame, method: str) -> pd.DataFrame:
    result = panel.copy()
    if method == "zero":
        result["y"] = result["y"].fillna(0.0)
    elif method == "forward":
        result["y"] = result.groupby("unique_id", sort=False)["y"].ffill().fillna(0.0)
    elif method == "median":
        medians = result.groupby("unique_id", sort=False)["y"].transform("median")
        result["y"] = result["y"].fillna(medians).fillna(0.0)
    elif method != "none":
        raise ValueError(f"Unknown fill method: {method}")
    return result


def dataset_fingerprint(path: str | Path, panel: pd.DataFrame) -> str:
    stat = Path(path).stat()
    payload = {
        "path": str(Path(path).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "rows": len(panel),
        "series": int(panel["unique_id"].nunique()),
        "min_ds": str(panel["ds"].min()),
        "max_ds": str(panel["ds"].max()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
