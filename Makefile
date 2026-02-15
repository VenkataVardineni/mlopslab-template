.PHONY: setup repro mlflow-up mlflow-down clean help

help:
	@echo "Available targets:"
	@echo "  make setup      - Install dependencies and setup environment"
	@echo "  make repro      - Run full DVC pipeline (dvc repro)"
	@echo "  make mlflow-up  - Start MLflow tracking server via docker-compose"
	@echo "  make mlflow-down - Stop MLflow tracking server"
	@echo "  make clean      - Clean generated artifacts and caches"

setup:
	@echo "Running setup..."
	@if [ -f setup.sh ]; then \
		./setup.sh; \
	else \
		pip install -r requirements-lock.txt; \
		pip install dvc; \
		dvc init --no-scm || true; \
		mkdir -p dvc_storage; \
		dvc remote add -d local_storage ./dvc_storage || true; \
	fi

repro:
	dvc repro

mlflow-up:
	cd docker && docker-compose up -d

mlflow-down:
	cd docker && docker-compose down

clean:
	rm -rf artifacts/* metrics/* plots/* data/raw/* data/processed/*
	rm -rf mlruns/ .dvc/cache
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

