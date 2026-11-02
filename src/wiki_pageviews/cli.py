from __future__ import annotations

import argparse
import json
from typing import Any

from wiki_pageviews.config import load_config
from wiki_pageviews.data import prepare_wide_csv
from wiki_pageviews.experiment import explain_experiment, run_experiment, tune_experiment


def _config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return load_config(args.config, args.set or [])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wiki-forecast")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="normalize a wide Kaggle CSV")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--max-series", type=int)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--chunksize", type=int, default=2_000)

    for name, help_text in [
        ("run", "run baseline/global comparisons"),
        ("tune", "optimize the global model and score the final holdout"),
        ("explain", "compute permutation feature importance"),
        ("sota", "run the optional N-HiTS benchmark"),
    ]:
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", required=True)
        command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        path = prepare_wide_csv(args.input, args.output, args.max_series, args.seed, args.chunksize)
        print(f"Prepared dataset: {path}")
        return 0
    config = _config_from_args(args)
    if args.command == "run":
        result = run_experiment(config)
    elif args.command == "tune":
        result = tune_experiment(config)
    elif args.command == "explain":
        result = explain_experiment(config).head(20).to_dict(orient="records")
    elif args.command == "sota":
        from wiki_pageviews.neural import run_nhits

        result = run_nhits(config)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
