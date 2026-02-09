# src/data.py
from __future__ import annotations
import pandas as pd
from .config import DATA_DIR

def build_dataset() -> pd.DataFrame:
    """
    Loads the Telco CSVs and returns a single merged dataframe.
    Adjust filenames/merge logic to match your dataset.
    """
    status = pd.read_csv(DATA_DIR / "Telco_customer_churn_status.csv")
    services = pd.read_csv(DATA_DIR / "Telco_customer_churn_services.csv")
    demographics = pd.read_csv(DATA_DIR / "Telco_customer_churn_demographics.csv")

    status = status[['Customer ID', 'Churn Value']]
    services = services[['Customer ID', 'Internet Service', 'Internet Type', 'Phone Service', 'Multiple Lines', 'Online Security', 'Device Protection Plan', 'Premium Tech Support', 'Contract', 'Avg Monthly GB Download', 'Total Long Distance Charges']]
    demographics = demographics[['Customer ID', 'Age', 'Gender', 'Married', 'Dependents']]

    # Merge on Customer ID (left join keeps base table rows)
    findf = pd.merge(services, demographics, how='inner', left_on='Customer ID', right_on='Customer ID')
    findf = pd.merge(findf, status, how='inner', left_on='Customer ID', right_on='Customer ID')

    return findf
