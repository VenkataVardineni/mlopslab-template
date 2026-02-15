"""Stage: Evaluate model and generate metrics + plots."""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

from src.stages.train import Seq2SeqAttention, TimeSeriesDataset


def load_model(model_dir: Path, device: torch.device) -> nn.Module:
    """Load trained model."""
    with open(model_dir / "model_card.json", "r") as f:
        model_card = json.load(f)
    
    # Reconstruct model
    model = Seq2SeqAttention(
        input_dim=model_card["input_dim"],
        hidden_dim=model_card["hyperparameters"]["hidden_dim"],
        num_layers=model_card["hyperparameters"]["num_layers"],
        dropout=model_card["hyperparameters"]["dropout"],
        attention_dim=model_card["hyperparameters"]["attention_dim"],
        horizon=model_card["horizon"],
        quantiles=model_card["quantiles"]
    ).to(device)
    
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    model.eval()
    return model


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device, 
                   quantiles: List[float]) -> Dict[str, np.ndarray]:
    """Evaluate model and return predictions."""
    predictions = []
    targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)  # (batch, num_quantiles, horizon)
    targets = np.concatenate(targets, axis=0)  # (batch, horizon)
    
    return {
        "predictions": predictions,
        "targets": targets
    }


def compute_metrics(targets: np.ndarray, predictions: np.ndarray, quantiles: List[float]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    # predictions: (batch, num_quantiles, horizon)
    # targets: (batch, horizon)
    
    # Use median (p50) for point forecasts
    p50_idx = quantiles.index(0.5)
    pred_p50 = predictions[:, p50_idx, :]
    
    # Flatten for metric computation
    targets_flat = targets.flatten()
    pred_p50_flat = pred_p50.flatten()
    
    # MAE and RMSE
    mae = np.mean(np.abs(targets_flat - pred_p50_flat))
    rmse = np.sqrt(np.mean((targets_flat - pred_p50_flat) ** 2))
    
    # Coverage (for prediction intervals)
    p10_idx = quantiles.index(0.1)
    p90_idx = quantiles.index(0.9)
    pred_p10 = predictions[:, p10_idx, :].flatten()
    pred_p90 = predictions[:, p90_idx, :].flatten()
    
    coverage = np.mean((targets_flat >= pred_p10) & (targets_flat <= pred_p90))
    
    # Mean Interval Width (normalized)
    mean_interval_width = np.mean(pred_p90 - pred_p10)
    mean_target = np.mean(np.abs(targets_flat))
    normalized_width = mean_interval_width / mean_target if mean_target > 0 else 0
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "coverage": float(coverage),
        "mean_interval_width": float(mean_interval_width),
        "normalized_interval_width": float(normalized_width)
    }


def create_forecast_plot_data(df: pd.DataFrame, targets: np.ndarray, predictions: np.ndarray,
                              quantiles: List[float], timestamp_col: str, 
                              context_length: int, horizon: int) -> pd.DataFrame:
    """Create forecast plot data in the required format."""
    # Get timestamps for forecast period
    timestamps = df[timestamp_col].values[context_length:context_length + len(targets)]
    
    # Extract quantile predictions
    p10_idx = quantiles.index(0.1)
    p50_idx = quantiles.index(0.5)
    p90_idx = quantiles.index(0.9)
    
    # For each sample, take the first horizon step (or average across horizon)
    # Here we'll take the first step of each forecast
    pred_p10 = predictions[:, p10_idx, 0]
    pred_p50 = predictions[:, p50_idx, 0]
    pred_p90 = predictions[:, p90_idx, 0]
    actual = targets[:, 0]
    
    # Create DataFrame
    plot_df = pd.DataFrame({
        "timestamp": timestamps[:len(actual)],
        "actual": actual,
        "pred_p50": pred_p50,
        "pred_p10": pred_p10,
        "pred_p90": pred_p90
    })
    
    return plot_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate model and generate metrics + plots")
    parser.add_argument("--features", type=str, default="data/processed/features.parquet", help="Input features path")
    parser.add_argument("--model-dir", type=str, default="artifacts/model", help="Model directory")
    parser.add_argument("--metrics-output", type=str, default="metrics/metrics.json", help="Output metrics path")
    parser.add_argument("--plots-output", type=str, default="plots/forecast.csv", help="Output plots path")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data config")
    args = parser.parse_args()
    
    features_path = Path(args.features)
    model_dir = Path(args.model_dir)
    metrics_output = Path(args.metrics_output)
    plots_output = Path(args.plots_output)
    params_path = Path(args.params)
    config_path = Path(args.config)
    
    if not features_path.exists():
        print(f"Error: Features file not found: {features_path}")
        sys.exit(1)
    
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)
    
    if not params_path.exists():
        print(f"Error: Params file not found: {params_path}")
        sys.exit(1)
    
    # Load params and config
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load data
    print(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    timestamp_col = config["dataset"]["timestamp_col"]
    value_col = config["dataset"]["value_col"]
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in [timestamp_col, value_col]]
    X = df[feature_cols].values
    y = df[value_col].values
    
    # Split train/test
    split_idx = int(len(df) * params["data"]["train_split"])
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    df_test = df.iloc[split_idx:].copy()
    
    print(f"Test size: {len(X_test)}")
    
    # Create test dataset
    context_length = params["model"]["context_length"]
    horizon = params["model"]["horizon"]
    test_dataset = TimeSeriesDataset(X_test, y_test, context_length, horizon)
    test_loader = DataLoader(test_dataset, batch_size=params["training"]["batch_size"], shuffle=False)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_dir}")
    model = load_model(model_dir, device)
    
    # Evaluate
    print("Evaluating model...")
    quantiles = params["quantiles"]
    results = evaluate_model(model, test_loader, device, quantiles)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(results["targets"], results["predictions"], quantiles)
    
    # Save metrics
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Metrics saved to {metrics_output}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Coverage: {metrics['coverage']:.4f}")
    
    # Create plot data
    print("Creating forecast plot data...")
    plot_df = create_forecast_plot_data(
        df_test, results["targets"], results["predictions"],
        quantiles, timestamp_col, context_length, horizon
    )
    
    # Save plot data
    plots_output.parent.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(plots_output, index=False)
    
    print(f"✓ Plot data saved to {plots_output}")
    print(f"  Shape: {plot_df.shape}")


if __name__ == "__main__":
    main()

