import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import pandas as pd
from src.data import build_dataset

REQUIRED_COLS = [
    "Customer ID",
    "Churn Value",
]

def test_build_dataset_returns_dataframe():
    df = build_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

def test_required_columns_exist():
    df = build_dataset()
    for c in REQUIRED_COLS:
        assert c in df.columns, f"Missing required column: {c}"

def test_target_is_binary():
    df = build_dataset()
    vals = set(df["Churn Value"].dropna().unique().tolist())
    assert vals.issubset({0, 1}), f"Churn Value has unexpected values: {vals}"
