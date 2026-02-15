"""Reproducibility utilities for setting seeds and logging git commit SHA."""

import os
import random
import subprocess
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_commit_sha() -> Optional[str]:
    """Get the current git commit SHA.
    
    Returns:
        Git commit SHA string, or None if not in a git repo or git is unavailable
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def log_reproducibility_info(output_dir: Path, seed: int = 42, **kwargs) -> None:
    """Log reproducibility information to a JSON file.
    
    Args:
        output_dir: Directory to save the reproducibility info file
        seed: Random seed used
        **kwargs: Additional metadata to log
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    info = {
        "seed": seed,
        "git_commit_sha": get_git_commit_sha(),
        **kwargs
    }
    
    info_path = output_dir / "repro_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"Reproducibility info saved to {info_path}")
    print(f"Git commit SHA: {info['git_commit_sha']}")

