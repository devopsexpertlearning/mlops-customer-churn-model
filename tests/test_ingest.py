import os
import pathlib
from typing import Any
from unittest.mock import MagicMock, patch
import pandas as pd
from customer_churn.config import AppConfig
from customer_churn.ingest import ingest_data


@patch("customer_churn.ingest.pd.read_csv")
def test_ingest_data_success(mock_read_csv: MagicMock, tmp_path: pathlib.Path) -> None:
    """Tests a successful run of the data ingestion script in isolation."""
    # Mock return value of pd.read_csv
    mock_df = pd.DataFrame({"customerID": ["123"], "Churn": ["No"]})
    mock_read_csv.return_value = mock_df

    # Create temporary path for raw output
    temp_raw_path = os.path.join(tmp_path, "raw_data.csv")

    # Patch AppConfig get to isolate test from local configs/config.yaml
    original_get = AppConfig.get

    def mock_get(self: AppConfig, key_path: str, default: Any = None) -> Any:
        if key_path == "data.raw_path":
            return temp_raw_path
        if key_path == "data.ingest_url":
            return "https://mock-url.com/telecom-data.csv"
        return original_get(self, key_path, default)

    with patch.object(AppConfig, "get", mock_get):
        ingest_data()

    # Validate output exists and is populated
    assert os.path.exists(temp_raw_path)
    df_out = pd.read_csv(temp_raw_path)
    assert df_out.shape == (1, 2)
    assert df_out.iloc[0]["customerID"] == "123"
