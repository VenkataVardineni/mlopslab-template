"""Stage: Prepare features from raw data with lag, rolling, and calendar features."""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_lag_features(df: pd.DataFrame, value_col: str, lags: list) -> pd.DataFrame:
    """Create lag features."""
    df = df.copy()
    for lag in lags:
        df[f"{value_col}_lag_{lag}"] = df[value_col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, value_col: str, windows: list) -> pd.DataFrame:
    """Create rolling window statistics."""
    df = df.copy()
    for window in windows:
        df[f"{value_col}_rolling_mean_{window}"] = df[value_col].rolling(window=window, min_periods=1).mean()
        df[f"{value_col}_rolling_std_{window}"] = df[value_col].rolling(window=window, min_periods=1).std()
        df[f"{value_col}_rolling_min_{window}"] = df[value_col].rolling(window=window, min_periods=1).min()
        df[f"{value_col}_rolling_max_{window}"] = df[value_col].rolling(window=window, min_periods=1).max()
    return df


def create_calendar_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Create calendar-based features."""
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    df["year"] = df[timestamp_col].dt.year
    df["month"] = df[timestamp_col].dt.month
    df["day"] = df[timestamp_col].dt.day
    df["dayofweek"] = df[timestamp_col].dt.dayofweek
    df["dayofyear"] = df[timestamp_col].dt.dayofyear
    df["week"] = df[timestamp_col].dt.isocalendar().week
    df["quarter"] = df[timestamp_col].dt.quarter
    
    # Cyclical encoding for periodic features
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    
    return df


def fit_scaler(df: pd.DataFrame, feature_cols: list, scaler_dir: Path) -> StandardScaler:
    """Fit scaler on training data and save."""
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].values)
    
    # Save scaler parameters
    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": feature_cols
    }
    
    with open(scaler_dir / "scaler.json", "w") as f:
        json.dump(scaler_params, f, indent=2)
    
    print(f"✓ Scaler saved to {scaler_dir / 'scaler.json'}")
    return scaler


def main():
    parser = argparse.ArgumentParser(description="Prepare features from raw data")
    parser.add_argument("--input", type=str, default="data/raw/source.csv", help="Input raw data path")
    parser.add_argument("--output", type=str, default="data/processed/features.parquet", help="Output features path")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data config file")
    parser.add_argument("--scaler-dir", type=str, default="artifacts/preprocess", help="Directory to save scaler")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    config_path = Path(args.config)
    scaler_dir = Path(args.scaler_dir)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config
    config = load_config(config_path)
    timestamp_col = config["dataset"]["timestamp_col"]
    value_col = config["dataset"]["value_col"]
    
    # Load raw data
    print(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    # Create features
    print("Creating lag features...")
    lags = [1, 2, 3, 7, 14, 30]
    df = create_lag_features(df, value_col, lags)
    
    print("Creating rolling features...")
    windows = [7, 14, 30]
    df = create_rolling_features(df, value_col, windows)
    
    print("Creating calendar features...")
    df = create_calendar_features(df, timestamp_col)
    
    # Drop rows with NaN from lag features (first few rows)
    df = df.dropna().reset_index(drop=True)
    
    # Identify feature columns (exclude timestamp and target)
    feature_cols = [col for col in df.columns if col not in [timestamp_col, value_col]]
    
    # Split into train/test (use 80/20 split)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
    
    # Fit scaler on training data only
    print("Fitting scaler on training data...")
    scaler = fit_scaler(df_train, feature_cols, scaler_dir)
    
    # Transform features (but keep original target and timestamp)
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    df_train_scaled[feature_cols] = scaler.transform(df_train[feature_cols].values)
    df_test_scaled[feature_cols] = scaler.transform(df_test[feature_cols].values)
    
    # Combine train and test
    df_processed = pd.concat([df_train_scaled, df_test_scaled], ignore_index=True)
    
    # Save processed features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_parquet(output_path, index=False)
    
    print(f"✓ Processed features saved to {output_path}")
    print(f"  Shape: {df_processed.shape}")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Train/test split: {len(df_train)}/{len(df_test)}")


if __name__ == "__main__":
    main()

