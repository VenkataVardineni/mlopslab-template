"""MLflow utilities for logging experiments."""

import os
import mlflow
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json


def setup_mlflow(tracking_uri: Optional[str] = None, experiment_name: str = "mlopslab") -> None:
    """Setup MLflow tracking.
    
    Args:
        tracking_uri: MLflow tracking URI (defaults to MLFLOW_TRACKING_URI env var or local)
        experiment_name: Name of the MLflow experiment
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    mlflow.set_experiment(experiment_name)


def log_params_from_yaml(params_path: Path) -> None:
    """Log parameters from params.yaml file.
    
    Args:
        params_path: Path to params.yaml file
    """
    if not params_path.exists():
        print(f"Warning: params.yaml not found at {params_path}")
        return
    
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    
    # Flatten nested params
    def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    flat_params = flatten_dict(params)
    mlflow.log_params(flat_params)


def log_metrics_from_json(metrics_path: Path) -> None:
    """Log metrics from metrics.json file.
    
    Args:
        metrics_path: Path to metrics.json file
    """
    if not metrics_path.exists():
        print(f"Warning: metrics.json not found at {metrics_path}")
        return
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    mlflow.log_metrics(metrics)


def log_artifacts_from_path(artifact_path: Path, artifact_name: Optional[str] = None) -> None:
    """Log artifacts (files or directories) to MLflow.
    
    Args:
        artifact_path: Path to artifact file or directory
        artifact_name: Optional name for the artifact in MLflow
    """
    if not artifact_path.exists():
        print(f"Warning: Artifact not found at {artifact_path}")
        return
    
    if artifact_name:
        mlflow.log_artifacts(artifact_path, artifact_name=artifact_name)
    else:
        mlflow.log_artifacts(artifact_path)


def log_model_artifacts(model_dir: Path) -> None:
    """Log model artifacts (model.pt, model_card.json, etc.).
    
    Args:
        model_dir: Directory containing model artifacts
    """
    if not model_dir.exists():
        print(f"Warning: Model directory not found at {model_dir}")
        return
    
    # Log model files
    for file in ["model.pt", "model_card.json", "repro_info.json"]:
        file_path = model_dir / file
        if file_path.exists():
            mlflow.log_artifact(file_path, "model")
    
    # Log entire model directory as artifact
    mlflow.log_artifacts(model_dir, "model")


def log_forecast_plot(plot_path: Path) -> None:
    """Log forecast plot CSV as artifact.
    
    Args:
        plot_path: Path to forecast.csv
    """
    if plot_path.exists():
        mlflow.log_artifact(plot_path, "plots")
    else:
        print(f"Warning: Plot file not found at {plot_path}")

