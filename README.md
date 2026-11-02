# wiki-pageviews

Reproducible experiments for the Kaggle **Web Traffic Time Series Forecasting**
challenge. The repository starts with hard-to-beat seasonal baselines, then moves to a
global tree model with Optuna optimization and feature importance, and finally exposes an
optional neural-forecasting stage for N-HiTS. Every experiment is recorded in MLflow.

## Experiment ladder

| Stage | Models | Purpose |
|---|---|---|
| 0 | Last value, median, rolling median | Cheap sanity checks |
| 1 | Seasonal naive (7, 28, 364 days) | Strong traffic baselines |
| 2 | Global histogram gradient boosting | One cross-series model using calendar, lag, rolling, and page metadata features |
| 3 | Optuna-tuned global model | Time-series cross-validation without look-ahead leakage |
| 4 | N-HiTS (optional) | Modern neural benchmark through NeuralForecast |

The primary metric is symmetric mean absolute percentage error (SMAPE), matching the
competition. Model selection uses rolling-origin folds; the final untouched holdout is not
used by Optuna.

## Quick start

Python 3.11-3.13 is recommended because some optional forecasting libraries may lag the
newest Python release.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev,boosting]"

python scripts/make_synthetic.py --output data/processed/synthetic.parquet
wiki-forecast run --config configs/smoke.yaml
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open <http://127.0.0.1:5000>. The smoke configuration deliberately uses a tiny
synthetic panel and is suitable for validating the entire pipeline.

## Real challenge data

Place Kaggle's `train_1.csv` (or another wide challenge training file) in `data/raw/` and
normalize it once:

```bash
wiki-forecast prepare \
  --input data/raw/train_1.csv \
  --output data/processed/train.parquet
wiki-forecast run --config configs/baselines.yaml
wiki-forecast tune --config configs/advanced.yaml
wiki-forecast explain --config configs/advanced.yaml
```

The normalized schema is `unique_id`, `ds`, `y`, plus parsed page metadata. Missing traffic
values remain missing during ingestion; each training window applies its configured fill
policy independently so validation data cannot influence training.

For quick iteration, `prepare` supports deterministic series sampling:

```bash
wiki-forecast prepare --input data/raw/train_1.csv \
  --output data/processed/train-10k.parquet --max-series 10000 --seed 42
```

## MLflow contract

Each parent run stores the full resolved configuration, dataset fingerprint, row/series
counts, split boundaries, aggregate metrics, per-series metrics, and predictions.
Each model or Optuna trial is a nested run. The selected model also logs its serialized
estimator, feature contract, and permutation-importance table. Set the configuration's
tracking URI to a shared server if desired; otherwise run metadata is written to local
`mlflow.db` and artifacts remain local.

## Leakage controls

- Lag and rolling features are shifted before aggregation.
- Validation folds are chronological and expanding-window.
- Recursive forecasts only consume prior predictions after the cutoff.
- Page identity is excluded from one-hot features by default, avoiding an enormous sparse
  matrix; language/access/agent metadata are retained.
- The final holdout is evaluated only after hyperparameter selection.

## Configuration

YAML files under `configs/` define the dataset, folds, models, search space, and tracking.
Override a config value from the command line with repeated `--set key=value`, for example:

```bash
wiki-forecast tune --config configs/advanced.yaml \
  --set tuning.n_trials=50 --set data.max_series=20000
```

The optional neural stage needs `pip install -e ".[neural]"` and is intentionally separate
because GPU/runtime requirements differ from the CPU baseline pipeline.

## Repository layout

```text
configs/                 experiment definitions
scripts/                 reproducible utility scripts
src/wiki_pageviews/      data, features, models, evaluation, tracking, CLI
tests/                   metric, feature, split, and end-to-end tests
data/                    ignored raw/processed datasets
artifacts/               ignored local reports and model outputs
```

See [`docs/experiment_plan.md`](docs/experiment_plan.md) for the phased research plan and
decision gates.

