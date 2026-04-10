.PHONY: setup test lint format serve docker-up docker-down generate detect clean

setup:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=spectral --cov-report=term-missing -m "not slow and not integration"

test-all:
	pytest tests/ -v --cov=spectral --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

serve:
	uvicorn spectral.api.app:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up -d

docker-down:
	docker compose down -v

generate:
	spectral generate --points 5000 --anomaly-ratio 0.03 --output data/synthetic.csv

detect:
	spectral detect data/synthetic.csv --config configs/default.yml --config configs/default.yml

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
