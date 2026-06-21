import json
import os
import pathlib
from typing import Any, Dict, List
from unittest.mock import patch
import pandas as pd
import pytest
from customer_churn.config import AppConfig
from customer_churn.validate import validate_data


@pytest.fixture
def mock_raw_df() -> pd.DataFrame:
    """Fixture containing a valid raw dataset mock."""
    data: Dict[str, List[Any]] = {
        "customerID": ["7590-VHVEG", "5575-GNVDE"],
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 0],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [1, 34],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": ["No phone service", "No"],
        "InternetService": ["DSL", "DSL"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "No"],
        "StreamingTV": ["No", "No"],
        "StreamingMovies": ["No", "No"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [29.85, 56.95],
        "TotalCharges": ["29.85", "1889.5"],
        "Churn": ["No", "No"],
    }
    # Multiply lists to satisfy the row count threshold (>1000) for testing
    extended_data = {k: v * 600 for k, v in data.items()}
    return pd.DataFrame(extended_data)


def test_validate_data_success(
    mock_raw_df: pd.DataFrame, tmp_path: pathlib.Path
) -> None:
    """Tests successful validation check run."""
    temp_raw_path = os.path.join(tmp_path, "raw_data.csv")
    mock_raw_df.to_csv(temp_raw_path, index=False)

    report_path = "artifacts/validation_report.json"
    if os.path.exists(report_path):
        os.remove(report_path)

    # Patch AppConfig get to isolate configuration paths
    original_get = AppConfig.get

    def mock_get(self: AppConfig, key_path: str, default: Any = None) -> Any:
        if key_path == "data.raw_path":
            return temp_raw_path
        if key_path == "data.target_col":
            return "Churn"
        return original_get(self, key_path, default)

    with patch.object(AppConfig, "get", mock_get):
        validate_data()

    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        report = json.load(f)

    assert report["validation_passed"] is True
    assert report["checks"]["column_presence"]["status"] == "PASSED"
    assert report["checks"]["row_count"]["status"] == "PASSED"
