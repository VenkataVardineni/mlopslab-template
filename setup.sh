#!/bin/bash
# Setup script for MLOps Lab Template
# This script sets up the project environment and dependencies

set -e  # Exit on error

echo "=========================================="
echo "MLOps Lab Template - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Check if Python 3.8+
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
    echo "❌ Error: Python 3.8 or higher is required"
    exit 1
fi
echo "   ✓ Python version is compatible"
echo ""

# Check if pip is available
echo "📦 Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is not installed"
    exit 1
fi
echo "   ✓ pip is available"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip3 install --upgrade pip --quiet
echo "   ✓ pip upgraded"
echo ""

# Install Python dependencies
echo "📥 Installing Python dependencies from requirements-lock.txt..."
if [ ! -f "requirements-lock.txt" ]; then
    echo "❌ Error: requirements-lock.txt not found"
    exit 1
fi

pip3 install -r requirements-lock.txt
echo "   ✓ Python dependencies installed"
echo ""

# Check if DVC is installed
echo "🔍 Checking DVC installation..."
if ! command -v dvc &> /dev/null; then
    echo "⚠️  DVC not found in PATH, installing..."
    pip3 install dvc
    echo "   ✓ DVC installed"
else
    echo "   ✓ DVC is already installed"
fi
echo ""

# Initialize DVC if not already initialized
if [ ! -d ".dvc" ]; then
    echo "🔧 Initializing DVC..."
    dvc init --no-scm
    echo "   ✓ DVC initialized"
else
    echo "   ✓ DVC already initialized"
fi
echo ""

# Setup DVC remote if not configured
if [ ! -d "dvc_storage" ]; then
    echo "📁 Creating DVC storage directory..."
    mkdir -p dvc_storage
    echo "   ✓ DVC storage directory created"
fi

if ! dvc remote list | grep -q "local_storage"; then
    echo "🔗 Configuring DVC remote..."
    dvc remote add -d local_storage ./dvc_storage
    echo "   ✓ DVC remote configured"
else
    echo "   ✓ DVC remote already configured"
fi
echo ""

# Check Docker (optional, for MLflow)
echo "🐳 Checking Docker (optional for MLflow)..."
if command -v docker &> /dev/null; then
    echo "   ✓ Docker is installed"
    if command -v docker-compose &> /dev/null; then
        echo "   ✓ docker-compose is installed"
    else
        echo "   ⚠️  docker-compose not found (MLflow server may not work)"
    fi
else
    echo "   ⚠️  Docker not installed (MLflow server will not work)"
fi
echo ""

# Create necessary directories
echo "📂 Creating project directories..."
directories=(
    "data/raw"
    "data/processed"
    "artifacts/model"
    "artifacts/preprocess"
    "metrics"
    "plots"
    "mlruns"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "   ✓ Created $dir"
    fi
done
echo ""

# Verify installation
echo "✅ Verifying installation..."
echo ""

# Check key packages
packages=("numpy" "pandas" "torch" "mlflow" "dvc" "fastapi" "sklearn" "statsmodels")
missing_packages=()

for package in "${packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "   ✓ $package"
    else
        echo "   ❌ $package (missing)"
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Warning: Some packages are missing. Try running:"
    echo "   pip3 install -r requirements-lock.txt"
    echo ""
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run the pipeline:     make repro"
echo "  2. Start MLflow server:  make mlflow-up"
echo "  3. View MLflow UI:       http://localhost:5000"
echo "  4. Serve the model:      cd serve && uvicorn app:app --reload"
echo ""
echo "For more information, see README.md"
echo ""

