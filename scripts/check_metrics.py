#!/usr/bin/env python3
"""Script to check metrics against thresholds and fail CI if exceeded."""

import json
import sys
from pathlib import Path


def main():
    metrics_path = Path("metrics/metrics.json")
    
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found")
        sys.exit(1)
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    # Define thresholds
    thresholds = {
        "mae": 10.0,
        "rmse": 15.0,
        "coverage": 0.7  # Minimum 70% coverage
    }
    
    failed = False
    
    for metric_name, threshold in thresholds.items():
        if metric_name not in metrics:
            print(f"Warning: {metric_name} not found in metrics")
            continue
        
        value = metrics[metric_name]
        
        if metric_name == "coverage":
            # Coverage should be >= threshold
            if value < threshold:
                print(f"❌ {metric_name.upper()}: {value:.4f} < {threshold:.4f} (threshold)")
                failed = True
            else:
                print(f"✓ {metric_name.upper()}: {value:.4f} >= {threshold:.4f} (threshold)")
        else:
            # MAE and RMSE should be <= threshold
            if value > threshold:
                print(f"❌ {metric_name.upper()}: {value:.4f} > {threshold:.4f} (threshold)")
                failed = True
            else:
                print(f"✓ {metric_name.upper()}: {value:.4f} <= {threshold:.4f} (threshold)")
    
    if failed:
        print("\n❌ Metrics regression detected! Build failed.")
        sys.exit(1)
    else:
        print("\n✓ All metrics within acceptable thresholds")
        sys.exit(0)


if __name__ == "__main__":
    main()

