# MLOps Lab Template

A reproducible MLOps pipeline template for time series forecasting with DVC, MLflow tracking, and model serving.

## Features & Functionalities

### 🔄 End-to-End Reproducible Pipeline
- **DVC Pipeline Management**: Complete data versioning and pipeline orchestration
- **Deterministic Execution**: Reproducible runs with seed control and git commit tracking
- **Pipeline Stages**: Modular stages for data preparation, feature engineering, training, and evaluation
- **Dependency Tracking**: Automatic dependency resolution and caching

### 📊 Experiment Tracking & Comparison
- **MLflow Integration**: Comprehensive experiment tracking with parameters, metrics, and artifacts
- **PostgreSQL Backend**: Persistent storage for experiment metadata
- **Metrics Tracking**: Automatic tracking of MAE, RMSE, coverage, and interval width
- **Cross-Commit Comparison**: Compare metrics and plots across git commits using `dvc metrics diff`
- **Interactive Plots**: Generate HTML plots for forecast visualization

### 🤖 Machine Learning Models
- **Seq2Seq Attention Model**: Deep learning model with attention mechanism for quantile forecasting
- **ARIMA Baseline**: Statistical baseline model for comparison
- **Quantile Regression**: Predicts multiple quantiles (p10, p50, p90) for uncertainty estimation
- **Feature Engineering**: Automatic generation of lag, rolling, and calendar features

### 🚀 Model Serving
- **FastAPI REST API**: Production-ready API for model inference
- **Docker Support**: Containerized deployment
- **Real-time Forecasting**: Generate forecasts from context windows via HTTP API
- **Uncertainty Intervals**: Returns prediction intervals (p10, p50, p90)

### 🔧 DevOps & CI/CD
- **GitHub Actions CI**: Automated testing and reproducibility checks
- **Metrics Regression Gates**: Automatic failure on performance degradation
- **Docker Compose**: One-command MLflow server setup
- **Environment Locking**: Pinned dependencies for reproducibility

### 📈 Data Management
- **Data Versioning**: DVC tracks datasets, models, and metrics
- **Feature Store**: Processed features with train/test splits
- **Scaler Persistence**: Save and load preprocessing scalers
- **Schema Validation**: Automatic validation of input data

### 📝 Documentation & Scripts
- **Comprehensive README**: Detailed setup and usage instructions
- **Helper Scripts**: Automation scripts for common tasks
- **API Documentation**: Complete API reference for model serving
- **Plot Documentation**: Guide for visualizing results

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

## Setup Instructions

### Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.8+** (Python 3.9 recommended)
- **pip** (Python package manager)
- **Git** (for version control)
- **Docker & Docker Compose** (optional, for MLflow server)

### Installation Steps

#### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/VenkataVardineni/mlopslab-template.git
cd mlopslab-template

# Run the setup script
./setup.sh
```

The setup script will:
- Check Python version compatibility
- Install all Python dependencies
- Initialize and configure DVC
- Create necessary directories
- Verify installation

#### Option 2: Manual Setup

```bash
# Clone the repository
git clone https://github.com/VenkataVardineni/mlopslab-template.git
cd mlopslab-template

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-lock.txt

# Install DVC
pip install dvc

# Initialize DVC
dvc init --no-scm

# Configure DVC remote
mkdir -p dvc_storage
dvc remote add -d local_storage ./dvc_storage

# Create necessary directories
mkdir -p data/raw data/processed artifacts/model artifacts/preprocess metrics plots
```

#### Option 3: Using Makefile

```bash
# Install dependencies
make setup
```

### Verify Installation

After setup, verify that everything is installed correctly:

```bash
# Check Python packages
python3 -c "import numpy, pandas, torch, mlflow, dvc, fastapi; print('✓ All packages installed')"

# Check DVC
dvc --version

# Check Docker (optional)
docker --version
docker-compose --version
```

## Quick Start

Once setup is complete, you can start using the project:

```bash
# Run the full pipeline
make repro

# Start MLflow tracking server (optional)
make mlflow-up
export MLFLOW_TRACKING_URI=http://localhost:5000

# Access MLflow UI
# Open browser: http://localhost:5000

# Stop MLflow server
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

## Additional Documentation

- **[SETUP.md](SETUP.md)**: Comprehensive setup guide with troubleshooting
- **[docs/plots.md](docs/plots.md)**: Guide for visualizing forecast plots
- **[serve/README.md](serve/README.md)**: Model serving API documentation

## Troubleshooting

If you encounter issues during setup:

1. **Check Prerequisites**: Ensure Python 3.8+, pip, and Git are installed
2. **Use Virtual Environment**: Create and activate a virtual environment
3. **Check DVC**: Verify DVC is installed and configured: `dvc --version`
4. **Review SETUP.md**: See [SETUP.md](SETUP.md) for detailed troubleshooting

Common issues:
- **Permission errors**: Use virtual environment or `pip install --user`
- **DVC not found**: Install with `pip install dvc`
- **Docker issues**: MLflow works without Docker (use `mlflow ui`)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

