"""Stage: Prepare raw dataset from source."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict, Any


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_or_load_dataset(config: Dict[str, Any], output_path: Path) -> pd.DataFrame:
    """Download or load dataset from URL or local path."""
    dataset_config = config["dataset"]
    
    if "url" in dataset_config and dataset_config["url"]:
        print(f"Downloading dataset from {dataset_config['url']}")
        df = pd.read_csv(dataset_config["url"])
    elif "path" in dataset_config and dataset_config["path"]:
        print(f"Loading dataset from {dataset_config['path']}")
        df = pd.read_csv(dataset_config["path"])
    else:
        raise ValueError("Either 'url' or 'path' must be specified in dataset config")
    
    return df


def validate_dataset(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Validate dataset schema and ordering."""
    validation = config["dataset"]["validation"]
    
    # Check required columns
    required = validation["required_columns"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check minimum rows
    if len(df) < validation["min_rows"]:
        raise ValueError(f"Dataset has {len(df)} rows, minimum required: {validation['min_rows']}")
    
    # Parse and validate timestamp
    timestamp_col = config["dataset"]["timestamp_col"]
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=validation.get("timestamp_format", None), errors="coerce")
        if df[timestamp_col].isna().any():
            raise ValueError(f"Invalid timestamps found in column '{timestamp_col}'")
        
        # Check ordering if requested
        if validation.get("check_ordering", False):
            if not df[timestamp_col].is_monotonic_increasing:
                print("Warning: Timestamps are not in ascending order. Sorting...")
                df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    print(f"✓ Dataset validation passed: {len(df)} rows, {len(df.columns)} columns")
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare raw dataset")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data config file")
    parser.add_argument("--output", type=str, default="data/raw/source.csv", help="Output path for raw data")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    output_path = Path(args.output)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config
    config = load_config(config_path)
    
    # Download or load dataset
    df = download_or_load_dataset(config, output_path)
    
    # Validate
    df = validate_dataset(df, config)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"✓ Raw dataset saved to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()

