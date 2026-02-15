"""Stage: Train Seq2Seq Attention model for quantile forecasting."""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml

from src.utils.repro import set_seeds, log_reproducibility_info
from src.utils.mlflow_utils import setup_mlflow, log_params_from_yaml, log_model_artifacts
import mlflow


class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, context_length: int, horizon: int):
        self.features = features
        self.targets = targets
        self.context_length = context_length
        self.horizon = horizon
        
    def __len__(self):
        return len(self.features) - self.context_length - self.horizon + 1
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.context_length]
        y = self.targets[idx + self.context_length:idx + self.context_length + self.horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class Attention(nn.Module):
    """Attention mechanism for Seq2Seq."""
    
    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.W = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.V = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(self, encoder_outputs):
        # encoder_outputs: (batch, seq_len, hidden_dim)
        energy = self.V(torch.tanh(self.W(encoder_outputs)))  # (batch, seq_len, 1)
        attention_weights = torch.softmax(energy, dim=1)
        context = torch.sum(attention_weights * encoder_outputs, dim=1)  # (batch, hidden_dim)
        return context, attention_weights


class Seq2SeqAttention(nn.Module):
    """Seq2Seq model with attention for quantile regression."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 dropout: float, attention_dim: int, horizon: int, quantiles: List[float]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.quantiles = quantiles
        
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention
        self.attention = Attention(hidden_dim, attention_dim)
        
        # Decoder
        self.decoder = nn.LSTM(1, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output layers for each quantile
        self.quantile_outputs = nn.ModuleList([
            nn.Linear(hidden_dim, horizon) for _ in quantiles
        ])
        
    def forward(self, x):
        # x: (batch, context_length, input_dim)
        batch_size = x.size(0)
        
        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(x)
        
        # Attention
        context, _ = self.attention(encoder_outputs)
        
        # Decoder input (use last value as initial input)
        decoder_input = x[:, -1:, -1:]  # (batch, 1, 1) - last value of target
        
        # Decode
        decoder_outputs = []
        decoder_hidden = (context.unsqueeze(0).repeat(self.num_layers, 1, 1),
                         context.unsqueeze(0).repeat(self.num_layers, 1, 1))
        
        for _ in range(self.horizon):
            decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_out[:, -1, :])
            decoder_input = decoder_out[:, -1:, :1]  # Use prediction as next input
        
        decoder_outputs = torch.stack(decoder_outputs, dim=1)  # (batch, horizon, hidden_dim)
        
        # Generate quantile predictions
        predictions = []
        for quantile_layer in self.quantile_outputs:
            pred = quantile_layer(decoder_outputs[:, -1, :])  # Use last decoder output
            predictions.append(pred)
        
        return torch.stack(predictions, dim=1)  # (batch, num_quantiles, horizon)


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    """Quantile loss (pinball loss)."""
    losses = []
    for i, q in enumerate(quantiles):
        error = target - pred[:, i, :]
        loss = torch.max(q * error, (q - 1) * error)
        losses.append(loss)
    return torch.mean(torch.stack(losses))


def train_epoch(model, dataloader, optimizer, quantiles, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = quantile_loss(pred, y.unsqueeze(1).repeat(1, len(quantiles), 1), quantiles)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train Seq2Seq Attention model")
    parser.add_argument("--input", type=str, default="data/processed/features.parquet", help="Input features path")
    parser.add_argument("--output", type=str, default="artifacts/model", help="Output model directory")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data config")
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
    print(f"Loading features from {input_path}")
    df = pd.read_parquet(input_path)
    timestamp_col = config["dataset"]["timestamp_col"]
    value_col = config["dataset"]["value_col"]
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in [timestamp_col, value_col]]
    X = df[feature_cols].values
    y = df[value_col].values
    
    # Split train/test
    split_idx = int(len(df) * params["data"]["train_split"])
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Create datasets
    context_length = params["model"]["context_length"]
    horizon = params["model"]["horizon"]
    
    train_dataset = TimeSeriesDataset(X_train, y_train, context_length, horizon)
    train_loader = DataLoader(train_dataset, batch_size=params["training"]["batch_size"], shuffle=True)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    quantiles = params["quantiles"]
    model = Seq2SeqAttention(
        input_dim=len(feature_cols),
        hidden_dim=params["model"]["hidden_dim"],
        num_layers=params["model"]["num_layers"],
        dropout=params["model"]["dropout"],
        attention_dim=params["model"]["attention_dim"],
        horizon=horizon,
        quantiles=quantiles
    ).to(device)
    
    # Setup MLflow
    setup_mlflow(experiment_name="mlopslab-training")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=params["training"]["learning_rate"])
    epochs = params["training"]["epochs"]
    
    # Start MLflow run
    with mlflow.start_run(run_name="seq2seq-attention"):
        # Log parameters
        log_params_from_yaml(params_path)
        mlflow.log_param("model_type", "Seq2SeqAttention")
        
        # Train
        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            loss = train_epoch(model, train_loader, optimizer, quantiles, device)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
                mlflow.log_metric("train_loss", loss, step=epoch + 1)
        
        # Log final training loss
        mlflow.log_metric("final_train_loss", loss)
    
    # Save model
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path / "model.pt")
    
    # Save model card
    model_card = {
        "model_type": "Seq2SeqAttention",
        "hyperparameters": params["model"],
        "training_params": params["training"],
        "quantiles": quantiles,
        "input_dim": len(feature_cols),
        "feature_names": feature_cols,
        "context_length": context_length,
        "horizon": horizon,
    }
    
    with open(output_path / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)
    
    # Log reproducibility info
    log_reproducibility_info(output_path, seed=seed, model_type="Seq2SeqAttention")
    
    # Log model artifacts to MLflow
    log_model_artifacts(output_path)
    
    print(f"✓ Model saved to {output_path}")
    print(f"  Model card: {output_path / 'model_card.json'}")
    print(f"  MLflow run ID: {mlflow.active_run().info.run_id if mlflow.active_run() else 'N/A'}")


if __name__ == "__main__":
    main()

