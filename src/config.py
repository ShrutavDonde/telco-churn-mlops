# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"   # local MLflow tracking store
