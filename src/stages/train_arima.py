"""Stage: Train ARIMA baseline model."""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import yaml

from src.utils.repro import set_seeds, log_reproducibility_info


def find_best_arima_order(ts: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> tuple:
    """Find best ARIMA order using AIC."""
    best_aic = np.inf
    best_order = None
    
    print("Searching for best ARIMA order...")
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue
    
    print(f"Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
    return best_order


def main():
    parser = argparse.ArgumentParser(description="Train ARIMA baseline model")
    parser.add_argument("--input", type=str, default="data/raw/source.csv", help="Input raw data path")
    parser.add_argument("--output", type=str, default="artifacts/model_arima", help="Output model directory")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data config")
    parser.add_argument("--auto-order", action="store_true", help="Automatically find best ARIMA order")
    parser.add_argument("--p", type=int, default=2, help="ARIMA p parameter (if not auto)")
    parser.add_argument("--d", type=int, default=1, help="ARIMA d parameter (if not auto)")
    parser.add_argument("--q", type=int, default=2, help="ARIMA q parameter (if not auto)")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    params_path = Path(args.params)
    config_path = Path(args.config)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if not params_path.exists():
        print(f"Error: Params file not found: {params_path}")
        sys.exit(1)
    
    # Load params and config
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Set seeds
    seed = params["training"]["seed"]
    set_seeds(seed)
    
    # Load data
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    timestamp_col = config["dataset"]["timestamp_col"]
    value_col = config["dataset"]["value_col"]
    
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    # Split train/test
    split_idx = int(len(df) * params["data"]["train_split"])
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:]
    
    train_ts = train_data[value_col]
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Find or use ARIMA order
    if args.auto_order:
        order = find_best_arima_order(train_ts)
    else:
        order = (args.p, args.d, args.q)
        print(f"Using ARIMA order: {order}")
    
    # Fit ARIMA model
    print("Fitting ARIMA model...")
    model = ARIMA(train_ts, order=order)
    fitted_model = model.fit()
    
    # Forecast
    horizon = params["model"]["horizon"]
    print(f"Forecasting {horizon} steps ahead...")
    forecast = fitted_model.forecast(steps=horizon)
    forecast_ci = fitted_model.get_forecast(steps=horizon).conf_int()
    
    # Save model parameters and predictions
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save ARIMA parameters
    arima_params = {
        "order": order,
        "aic": float(fitted_model.aic),
        "bic": float(fitted_model.bic),
        "horizon": horizon,
        "model_summary": str(fitted_model.summary())
    }
    
    with open(output_path / "arima_params.json", "w") as f:
        json.dump(arima_params, f, indent=2)
    
    # Save predictions
    predictions = pd.DataFrame({
        "forecast": forecast.values,
        "lower": forecast_ci.iloc[:, 0].values,
        "upper": forecast_ci.iloc[:, 1].values
    })
    predictions.to_csv(output_path / "predictions.csv", index=False)
    
    # Save model card
    model_card = {
        "model_type": "ARIMA",
        "order": order,
        "aic": float(fitted_model.aic),
        "bic": float(fitted_model.bic),
        "horizon": horizon,
    }
    
    with open(output_path / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)
    
    # Log reproducibility info
    log_reproducibility_info(output_path, seed=seed, model_type="ARIMA", order=order)
    
    print(f"✓ ARIMA model saved to {output_path}")
    print(f"  AIC: {fitted_model.aic:.2f}")
    print(f"  Predictions saved to {output_path / 'predictions.csv'}")


if __name__ == "__main__":
    main()

