import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retailpulse import load_data, build_rfm, cohort_retention

ROOT = Path(__file__).resolve().parents[1]

def test_dataset_has_at_least_10000_rows():
    df = load_data(ROOT / "data" / "retail_transactions.csv")
    assert len(df) >= 10000

def test_rfm_has_expected_columns():
    df = load_data(ROOT / "data" / "retail_transactions.csv")
    rfm = build_rfm(df)
    assert {"CustomerID", "Recency", "Frequency", "Monetary"}.issubset(rfm.columns)

def test_cohort_retention_values_are_bounded():
    df = load_data(ROOT / "data" / "retail_transactions.csv")
    retention = cohort_retention(df)
    values = retention.to_numpy()
    valid = values[~pd.isna(values)]
    assert (valid >= 0).all()
    assert (valid <= 1.000001).all()
