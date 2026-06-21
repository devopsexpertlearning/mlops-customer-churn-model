import os
import pathlib
import pickle
from typing import Any
from unittest.mock import MagicMock, patch
import pandas as pd
from customer_churn.config import AppConfig
from customer_churn.train import train_model


def test_train_model_success(tmp_path: pathlib.Path) -> None:
    """Tests training and serializing the XGBoost model using mocked feature data."""
    # Create sample feature data
    train_features = {
        "tenure": [1.2, -0.5, 0.8, -1.1, 0.1],
        "MonthlyCharges": [0.5, -0.8, 1.2, -1.3, 0.4],
        "TotalCharges": [1.0, -0.6, 1.5, -1.0, 0.2],
        "gender_Male": [1.0, 0.0, 1.0, 0.0, 1.0],
        "Churn": [0, 1, 0, 1, 0],
    }
    train_features_df = pd.DataFrame(train_features)
    temp_features_path = os.path.join(tmp_path, "train_features.csv")
    train_features_df.to_csv(temp_features_path, index=False)

    temp_model_path = os.path.join(tmp_path, "model.pkl")

    original_get = AppConfig.get

    def mock_get(self: AppConfig, key_path: str, default: Any = None) -> Any:
        if key_path == "data.features_train_path":
            return temp_features_path
        if key_path == "model.path":
            return temp_model_path
        if key_path == "data.target_col":
            return "Churn"
        return original_get(self, key_path, default)

    # Mock MLflow context and calls
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id_123"

    with patch.object(AppConfig, "get", mock_get):
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.set_experiment"),
            patch("mlflow.start_run") as mock_start_run,
            patch("mlflow.log_params"),
            patch("mlflow.xgboost.log_model"),
        ):

            # Setup context manager return value
            mock_start_run.return_value.__enter__.return_value = mock_run

            # Run training using temporary artifacts path to avoid local pollution
            temp_run_id_path = os.path.join(tmp_path, "mlflow_run_id.txt")
            with patch("customer_churn.train.RUN_ID_PATH", temp_run_id_path):
                train_model()

    # Assertions
    assert os.path.exists(temp_model_path)
    assert os.path.exists(temp_run_id_path)

    with open(temp_run_id_path, "r") as f_run:
        assert f_run.read().strip() == "test_run_id_123"

    # Verify that the serialized model can be loaded and make predictions
    with open(temp_model_path, "rb") as f:
        model = pickle.load(f)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    X_test = train_features_df.drop(columns=["Churn"])
    preds = model.predict(X_test)
    assert len(preds) == len(train_features_df)
