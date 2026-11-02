.PHONY: install smoke baselines tune explain mlflow test lint

install:
	python -m pip install -e ".[dev,boosting]"

smoke:
	python scripts/make_synthetic.py --output data/processed/synthetic.parquet
	python -m wiki_pageviews.cli run --config configs/smoke.yaml

baselines:
	python -m wiki_pageviews.cli run --config configs/baselines.yaml

tune:
	python -m wiki_pageviews.cli tune --config configs/advanced.yaml

explain:
	python -m wiki_pageviews.cli explain --config configs/advanced.yaml

mlflow:
	python -m mlflow ui --backend-store-uri sqlite:///mlflow.db

test:
	python -m pytest

lint:
	ruff check .

