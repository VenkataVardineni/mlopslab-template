"""FastAPI model serving application."""

import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml

from src.stages.train import Seq2SeqAttention
from src.stages.prepare_features import create_lag_features, create_rolling_features, create_calendar_features


app = FastAPI(title="Time Series Forecast API", version="1.0.0")

# Global model and scaler
model = None
model_card = None
scaler_params = None
feature_cols = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ForecastRequest(BaseModel):
    """Request model for forecast endpoint."""
    context: List[float]  # Recent time series values (context_length required)
    timestamp: Optional[str] = None  # Optional timestamp for calendar features


class ForecastResponse(BaseModel):
    """Response model for forecast endpoint."""
    forecast: List[float]  # Median forecast (p50)
    forecast_p10: List[float]  # Lower bound (p10)
    forecast_p90: List[float]  # Upper bound (p90)
    horizon: int


def load_model(model_dir: Path):
    """Load trained model and metadata."""
    global model, model_card, scaler_params, feature_cols
    
    # Load model card
    with open(model_dir / "model_card.json", "r") as f:
        model_card = json.load(f)
    
    # Load scaler
    scaler_path = Path("artifacts/preprocess/scaler.json")
    if scaler_path.exists():
        with open(scaler_path, "r") as f:
            scaler_params = json.load(f)
        feature_cols = scaler_params["feature_names"]
    else:
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
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
    
    print(f"✓ Model loaded from {model_dir}")
    print(f"  Context length: {model_card['context_length']}")
    print(f"  Horizon: {model_card['horizon']}")


def prepare_features(context: List[float], timestamp: Optional[str] = None) -> np.ndarray:
    """Prepare features from context window."""
    context_length = model_card["context_length"]
    
    if len(context) < context_length:
        raise ValueError(f"Context must have at least {context_length} values, got {len(context)}")
    
    # Use last context_length values
    context_window = context[-context_length:]
    
    # Create a temporary DataFrame for feature engineering
    df = pd.DataFrame({
        "y": context_window,
        "ds": pd.date_range(end=pd.Timestamp.now(), periods=len(context_window), freq="D")
    })
    
    # Create features
    df = create_lag_features(df, "y", [1, 2, 3, 7, 14, 30])
    df = create_rolling_features(df, "y", [7, 14, 30])
    df = create_calendar_features(df, "ds")
    
    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)
    
    # Extract features in the correct order
    features = df[feature_cols].values[-1:]  # Take last row
    
    # Apply scaler
    if scaler_params:
        mean = np.array(scaler_params["mean"])
        scale = np.array(scaler_params["scale"])
        features = (features - mean) / scale
    
    return features


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    model_dir = Path("artifacts/model")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    load_model(model_dir)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Time Series Forecast API",
        "version": "1.0.0",
        "model_type": model_card["model_type"] if model_card else "Not loaded",
        "horizon": model_card["horizon"] if model_card else None
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=ForecastResponse)
async def predict(request: ForecastRequest):
    """Generate forecast from context window."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features = prepare_features(request.context, request.timestamp)
        
        # Convert to tensor
        x = torch.FloatTensor(features).to(device)
        
        # Generate forecast
        with torch.no_grad():
            predictions = model(x)  # (batch, num_quantiles, horizon)
        
        # Extract quantiles
        quantiles = model_card["quantiles"]
        p10_idx = quantiles.index(0.1)
        p50_idx = quantiles.index(0.5)
        p90_idx = quantiles.index(0.9)
        
        forecast_p50 = predictions[0, p50_idx, :].cpu().numpy().tolist()
        forecast_p10 = predictions[0, p10_idx, :].cpu().numpy().tolist()
        forecast_p90 = predictions[0, p90_idx, :].cpu().numpy().tolist()
        
        return ForecastResponse(
            forecast=forecast_p50,
            forecast_p10=forecast_p10,
            forecast_p90=forecast_p90,
            horizon=model_card["horizon"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

