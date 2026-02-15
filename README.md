# MLOps Lab Template

A reproducible MLOps pipeline template for time series forecasting with DVC, MLflow tracking, and model serving.

## Project Contract: Inputs/Outputs

### Inputs
- **Raw Data**: `data/raw/source.csv` - Source time series dataset
- **Configuration**: `configs/data.yaml`, `params.yaml` - Data and model parameters

### Outputs
- **Model Artifacts**: `artifacts/model/` - Trained model weights, scaler, and configuration
- **Metrics**: `metrics/metrics.json` - Evaluation metrics (MAE, RMSE, coverage, etc.)
- **Plots**: `plots/forecast.csv` - Forecast visualization data (timestamp, actual, pred_p50, pred_p10, pred_p90)

### Artifact Tracking Strategy
- **Git-tracked**: Configuration files, source code, scripts
- **DVC-tracked**: 
  - `data/raw/` - Raw datasets
  - `data/processed/` - Processed features
  - `artifacts/` - Model artifacts and preprocessors
  - `metrics/` - Evaluation metrics
  - `plots/` - Plot data files

## Quick Start

```bash
# Setup environment
make setup

# Run full pipeline
make repro

# Start MLflow UI
make mlflow-up
# Access at http://localhost:5000

# Stop MLflow
make mlflow-down
```

## Project Structure

```
mlopslab-template/
├── src/
│   ├── stages/          # Pipeline stages
│   └── utils/           # Utility functions
├── configs/             # Configuration files
├── data/                # Data directory (DVC-tracked)
│   ├── raw/            # Raw data
│   └── processed/      # Processed features
├── artifacts/           # Model artifacts (DVC-tracked)
├── metrics/             # Evaluation metrics (DVC-tracked)
├── plots/               # Plot data (DVC-tracked)
├── docker/              # Docker compose for MLflow
├── docs/                # Documentation
└── scripts/             # Helper scripts
```

## How to Compare Experiments

Use `dvc metrics diff` to compare metrics between commits:

```bash
# Compare current workspace with HEAD
dvc metrics diff

# Compare two specific commits
dvc metrics diff HEAD~1 HEAD

# Or use the comparison script
./scripts/compare_metrics.sh
```

## Serving

See `serve/README.md` for model serving instructions.

