# Detailed Setup Guide

This document provides comprehensive setup instructions for the MLOps Lab Template.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Configuration](#configuration)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Operating System**: Linux, macOS, or Windows (with WSL2 recommended)
- **Python**: 3.8 or higher (3.9+ recommended)
- **RAM**: 4GB minimum (8GB recommended for training)
- **Disk Space**: 2GB free space
- **Internet**: Required for downloading dependencies and datasets

### Optional Requirements

- **Docker**: For running MLflow server
- **Docker Compose**: For MLflow server orchestration
- **CUDA**: For GPU acceleration (optional, CPU works fine)

## Installation Methods

### Method 1: Automated Setup Script (Recommended)

The easiest way to set up the project:

```bash
# Clone the repository
git clone https://github.com/VenkataVardineni/mlopslab-template.git
cd mlopslab-template

# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

The script will:
- ✅ Check Python version
- ✅ Install all dependencies
- ✅ Initialize DVC
- ✅ Configure DVC remote
- ✅ Create necessary directories
- ✅ Verify installation

### Method 2: Using Makefile

```bash
# Clone repository
git clone https://github.com/VenkataVardineni/mlopslab-template.git
cd mlopslab-template

# Run setup
make setup
```

### Method 3: Manual Setup

#### Step 1: Clone Repository

```bash
git clone https://github.com/VenkataVardineni/mlopslab-template.git
cd mlopslab-template
```

#### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Step 3: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements-lock.txt
```

#### Step 4: Install DVC

```bash
pip install dvc
```

#### Step 5: Initialize DVC

```bash
# Initialize DVC
dvc init --no-scm

# Create DVC storage directory
mkdir -p dvc_storage

# Configure DVC remote
dvc remote add -d local_storage ./dvc_storage
```

#### Step 6: Create Project Directories

```bash
mkdir -p data/raw data/processed
mkdir -p artifacts/model artifacts/preprocess
mkdir -p metrics plots
```

## Configuration

### DVC Configuration

DVC is pre-configured with a local remote. To verify:

```bash
dvc remote list
```

You should see:
```
local_storage	./dvc_storage
```

### MLflow Configuration (Optional)

If you want to use MLflow tracking:

1. **Start MLflow Server**:
   ```bash
   make mlflow-up
   ```

2. **Set Tracking URI**:
   ```bash
   export MLFLOW_TRACKING_URI=http://localhost:5000
   ```

3. **Access MLflow UI**:
   Open http://localhost:5000 in your browser

### Environment Variables

You can set the following environment variables:

- `MLFLOW_TRACKING_URI`: MLflow tracking server URI (default: local file store)
- `PYTHONPATH`: Add project root to Python path (if needed)

## Verification

### Check Python Installation

```bash
python3 --version
# Should show Python 3.8 or higher
```

### Check Installed Packages

```bash
python3 -c "import numpy, pandas, torch, mlflow, dvc, fastapi; print('✓ All packages installed')"
```

### Check DVC

```bash
dvc --version
dvc remote list
```

### Test Pipeline

Run a quick test to verify everything works:

```bash
# This will download data and run the full pipeline
make repro
```

## Troubleshooting

### Common Issues

#### 1. Python Version Error

**Error**: `Python 3.8 or higher is required`

**Solution**:
```bash
# Check Python version
python3 --version

# If version is too old, install Python 3.9+
# On macOS:
brew install python@3.9

# On Ubuntu:
sudo apt update
sudo apt install python3.9
```

#### 2. pip Installation Fails

**Error**: `pip install` fails with permission errors

**Solution**:
```bash
# Use virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-lock.txt
```

#### 3. DVC Not Found

**Error**: `command not found: dvc`

**Solution**:
```bash
pip install dvc
# Or add to PATH if installed via package manager
```

#### 4. Docker Not Available

**Error**: MLflow server won't start

**Solution**:
- Install Docker: https://docs.docker.com/get-docker/
- Or use local MLflow (without Docker):
  ```bash
  mlflow ui --port 5000
  ```

#### 5. Out of Memory During Training

**Error**: CUDA out of memory or system runs out of RAM

**Solution**:
- Reduce batch size in `params.yaml`:
  ```yaml
  training:
    batch_size: 16  # Reduce from 32
  ```
- Reduce model size:
  ```yaml
  model:
    hidden_dim: 32  # Reduce from 64
  ```

#### 6. DVC Remote Configuration Error

**Error**: DVC remote not configured

**Solution**:
```bash
# Remove existing remote
dvc remote remove local_storage

# Add remote again
mkdir -p dvc_storage
dvc remote add -d local_storage ./dvc_storage
```

### Getting Help

If you encounter issues:

1. Check the [README.md](README.md) for common commands
2. Review error messages carefully
3. Ensure all prerequisites are installed
4. Try running `make clean` and then `make setup` again
5. Open an issue on GitHub with:
   - Error message
   - Python version
   - Operating system
   - Steps to reproduce

## Next Steps

After successful setup:

1. **Run the Pipeline**:
   ```bash
   make repro
   ```

2. **Start MLflow** (optional):
   ```bash
   make mlflow-up
   export MLFLOW_TRACKING_URI=http://localhost:5000
   ```

3. **View Results**:
   - Metrics: `cat metrics/metrics.json`
   - Plots: `dvc plots show plots/forecast.csv`
   - MLflow UI: http://localhost:5000

4. **Serve the Model**:
   ```bash
   cd serve
   uvicorn app:app --reload
   ```

## Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

