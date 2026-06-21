import os
import pathlib
import pickle
from typing import Any, Tuple
from unittest.mock import patch
import pandas as pd
import pytest
from customer_churn.config import AppConfig
from customer_churn.features import engineer_features


@pytest.fixture
def sample_splits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fixture containing small mock train/test splits."""
    train_data = {
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "No"],
        "tenure": [1, 2, 3],
        "PhoneService": ["No", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "Yes"],
        "InternetService": ["DSL", "DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes", "Yes"],
        "OnlineBackup": ["Yes", "No", "No"],
        "DeviceProtection": ["No", "Yes", "Yes"],
        "TechSupport": ["No", "No", "No"],
        "StreamingTV": ["No", "No", "Yes"],
        "StreamingMovies": ["No", "No", "Yes"],
        "Contract": ["Month-to-month", "One year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Electronic check",
        ],
        "MonthlyCharges": [29.85, 56.95, 84.1],
        "TotalCharges": [29.85, 113.9, 252.3],
        "Churn": [0, 1, 0],
    }
    test_data = {
        "gender": ["Male"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["Yes"],
        "tenure": [4],
        "PhoneService": ["Yes"],
        "MultipleLines": ["Yes"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["Yes"],
        "TechSupport": ["Yes"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["Yes"],
        "Contract": ["Two year"],
        "PaperlessBilling": ["No"],
        "PaymentMethod": ["Bank transfer"],
        "MonthlyCharges": [95.0],
        "TotalCharges": [380.0],
        "Churn": [0],
    }
    return pd.DataFrame(train_data), pd.DataFrame(test_data)


def test_engineer_features_success(
    sample_splits: Tuple[pd.DataFrame, pd.DataFrame], tmp_path: pathlib.Path
) -> None:
    """Tests fitting and transformation, pipeline serialization, and file output."""
    train_df, test_df = sample_splits
    temp_train_path = os.path.join(tmp_path, "train.csv")
    temp_test_path = os.path.join(tmp_path, "test.csv")
    train_df.to_csv(temp_train_path, index=False)
    test_df.to_csv(temp_test_path, index=False)

    temp_train_out = os.path.join(tmp_path, "train_features.csv")
    temp_test_out = os.path.join(tmp_path, "test_features.csv")
    temp_prep_path = os.path.join(tmp_path, "preprocessor.pkl")

    original_get = AppConfig.get

    def mock_get(self: AppConfig, key_path: str, default: Any = None) -> Any:
        if key_path == "data.processed_train_path":
            return temp_train_path
        if key_path == "data.processed_test_path":
            return temp_test_path
        if key_path == "data.features_train_path":
            return temp_train_out
        if key_path == "data.features_test_path":
            return temp_test_out
        if key_path == "model.preprocessor_path":
            return temp_prep_path
        if key_path == "data.target_col":
            return "Churn"
        return original_get(self, key_path, default)

    with patch.object(AppConfig, "get", mock_get):
        engineer_features()

    # Assertions
    assert os.path.exists(temp_train_out)
    assert os.path.exists(temp_test_out)
    assert os.path.exists(temp_prep_path)

    # Load outputs and verify shapes
    train_out_df = pd.read_csv(temp_train_out)
    assert "Churn" in train_out_df.columns

    # Check if preprocessor loads correctly
    with open(temp_prep_path, "rb") as f:
        preprocessor = pickle.load(f)
    assert hasattr(preprocessor, "transform")
