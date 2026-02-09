import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.infer import predict_proba_one

def test_predict_proba_one_returns_probability():
    features = {
        "Internet Service": "Yes",
        "Internet Type": "Fiber Optic",
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Online Security": "No",
        "Device Protection Plan": "No",
        "Premium Tech Support": "No",
        "Contract": "Month-to-Month",
        "Avg Monthly GB Download": 20,
        "Total Long Distance Charges": 30.5,
        "Age": 35,
        "Gender": "Male",
        "Married": "No",
        "Dependents": "No",
    }

    out = predict_proba_one(features)
    assert "churn_probability" in out
    p = out["churn_probability"]
    assert 0.0 <= p <= 1.0
