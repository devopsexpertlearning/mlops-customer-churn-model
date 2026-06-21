import os
import pathlib
from typing import Any
from unittest.mock import patch
import pandas as pd
import pytest
import yaml
from customer_churn.config import AppConfig
from customer_churn.preprocess import preprocess_data


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Fixture containing a small raw data sample with dirty fields."""
    return pd.DataFrame(
        {
            "customerID": ["1", "2", "3", "4", "5"],
            "gender": ["Female", "Male", "Female", "Male", "Female"],
            "tenure": [1, 2, 0, 4, 5],
            "MonthlyCharges": [29.85, 56.95, 18.9, 84.1, 95.0],
            "TotalCharges": ["29.85", "113.9", " ", "336.4", "475.0"],
            "Churn": ["No", "Yes", "No", "Yes", "No"],
        }
    )


def test_preprocess_data_success(
    sample_raw_data: pd.DataFrame, tmp_path: pathlib.Path
) -> None:
    """Tests that preprocessing cleans, maps target, and splits data correctly."""
    # Write mock raw data
    temp_raw_path = os.path.join(tmp_path, "raw_data.csv")
    sample_raw_data.to_csv(temp_raw_path, index=False)

    # Output paths
    temp_train_path = os.path.join(tmp_path, "train.csv")
    temp_test_path = os.path.join(tmp_path, "test.csv")

    # Mock parameters file
    mock_params = {"train": {"test_size": 0.4, "random_state": 42}}
    temp_params_path = os.path.join(tmp_path, "params.yaml")
    with open(temp_params_path, "w") as f:
        yaml.safe_dump(mock_params, f)

    # Patch AppConfig and open command for params.yaml
    original_get = AppConfig.get

    def mock_get(self: AppConfig, key_path: str, default: Any = None) -> Any:
        if key_path == "data.raw_path":
            return temp_raw_path
        if key_path == "data.processed_train_path":
            return temp_train_path
        if key_path == "data.processed_test_path":
            return temp_test_path
        if key_path == "data.target_col":
            return "Churn"
        return original_get(self, key_path, default)

    original_open = open

    def mock_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if file == "params.yaml":
            return original_open(temp_params_path, mode, *args, **kwargs)
        return original_open(file, mode, *args, **kwargs)

    with (
        patch.object(AppConfig, "get", mock_get),
        patch("customer_churn.preprocess.open", mock_open),
    ):
        preprocess_data()

    # Verify split files exist
    assert os.path.exists(temp_train_path)
    assert os.path.exists(temp_test_path)

    # Load outputs to verify cleaning
    train_df = pd.read_csv(temp_train_path)
    assert "customerID" not in train_df.columns
    assert train_df["TotalCharges"].dtype == float

    # Verify TotalCharges empty space replacement
    combined_total_charges = list(train_df["TotalCharges"].values) + list(
        pd.read_csv(temp_test_path)["TotalCharges"].values
    )
    assert 0.0 in combined_total_charges

    # Target should be converted to numeric [0, 1]
    assert set(train_df["Churn"].unique()).issubset({0, 1})
