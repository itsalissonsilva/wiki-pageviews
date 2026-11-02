from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_panel(n_series: int, n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    rows = []
    languages = ["en", "de", "fr", "ja"]
    access_types = ["all-access", "desktop", "mobile-web"]
    for index in range(n_series):
        language = languages[index % len(languages)]
        access = access_types[index % len(access_types)]
        agent = "all-agents" if index % 5 else "spider"
        project = f"{language}.wikipedia.org"
        base = rng.lognormal(mean=3.5, sigma=0.8)
        trend = rng.uniform(-0.02, 0.08) * np.arange(n_days)
        weekly = base * 0.25 * np.sin(2 * np.pi * (np.arange(n_days) + index) / 7)
        annual = base * 0.12 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
        noise = rng.normal(0, max(1, base * 0.12), n_days)
        traffic = np.clip(base + trend + weekly + annual + noise, 0, None).round()
        if index % 7 == 0:
            traffic[rng.random(n_days) < 0.12] = 0
        unique_id = f"Synthetic_page_{index}_{project}_{access}_{agent}"
        rows.append(
            pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": dates,
                    "y": traffic.astype("float32"),
                    "page": f"Synthetic_page_{index}",
                    "project": project,
                    "language": language,
                    "access": access,
                    "agent": agent,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--series", type=int, default=24)
    parser.add_argument("--days", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_panel(args.series, args.days, args.seed).to_parquet(path, index=False)
    print(f"Wrote {path} with {args.series} series x {args.days} days")


if __name__ == "__main__":
    main()
