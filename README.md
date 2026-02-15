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
- **DVC-tracked** (stored in `dvc_storage/` remote): 
  - `data/raw/` - Raw datasets
  - `data/processed/` - Processed features
  - `artifacts/` - Model artifacts and preprocessors
  - `metrics/` - Evaluation metrics
  - `plots/` - Plot data files

**Note**: After running `dvc init`, configure the local remote with:
```bash
dvc remote add -d local_storage ./dvc_storage
```

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

# Set MLflow tracking URI (before running pipeline)
export MLFLOW_TRACKING_URI=http://localhost:5000
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

# Compare specific commits
./scripts/compare_metrics.sh HEAD~1 HEAD
```

The script will show metric deltas and provide instructions for viewing plots comparisons.

## MLflow Tracking

The project uses MLflow to track experiments, parameters, metrics, and artifacts.

### Starting MLflow Server

```bash
# Start MLflow tracking server with PostgreSQL backend
make mlflow-up

# Set tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Run pipeline (experiments will be logged to MLflow)
make repro
```

### Accessing MLflow UI

Once the server is running, access the MLflow UI at:
- **URL**: http://localhost:5000

The UI allows you to:
- Browse all experiment runs
- Compare runs side-by-side
- View metrics, parameters, and artifacts
- Download model artifacts

### MLflow Configuration

The MLflow server uses:
- **Backend Store**: PostgreSQL (persistent)
- **Artifact Store**: Local volume (`mlflow_artifacts`)
- **Port**: 5000

Data persists across container restarts via Docker volumes.

## Serving

### Quick Start

```bash
# Run the FastAPI server locally
cd serve
uvicorn app:app --reload

# Or using Docker
docker build -t mlopslab-serve -f serve/Dockerfile .
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts mlopslab-serve
```

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Generate forecast
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"context": [1.0, 2.0, 3.0, ...]}'
```

For detailed API documentation, see `serve/README.md`.

