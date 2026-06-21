import json
import os
import pathlib
from typing import Any
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from customer_churn.config import AppConfig
from customer_churn.evaluate import evaluate_model


def test_evaluate_model_success(tmp_path: pathlib.Path) -> None:
    """Tests loading dependencies, prediction generation, and metric exports."""
    # Create sample test dataframe matching expected raw features
    test_data = {
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
    test_df = pd.DataFrame(test_data)
    temp_test_path = os.path.join(tmp_path, "test.csv")
    test_df.to_csv(temp_test_path, index=False)

    # Write empty placeholder files
    temp_prep_path = os.path.join(tmp_path, "preprocessor.pkl")
    with open(temp_prep_path, "wb") as f:
        f.write(b"")

    temp_model_path = os.path.join(tmp_path, "model.pkl")
    with open(temp_model_path, "wb") as f:
        f.write(b"")

    # Mock preprocessor and model behaviour
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    )

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1, 0])
    mock_model.predict_proba.return_value = np.array(
        [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]
    )

    temp_metrics_path = os.path.join(tmp_path, "metrics.json")
    temp_run_id_path = os.path.join(tmp_path, "mlflow_run_id.txt")
    with open(temp_run_id_path, "w") as f_run:
        f_run.write("test_run_id_123")

    original_get = AppConfig.get

    def mock_get(self: AppConfig, key_path: str, default: Any = None) -> Any:
        if key_path == "data.processed_test_path":
            return temp_test_path
        if key_path == "model.preprocessor_path":
            return temp_prep_path
        if key_path == "model.path":
            return temp_model_path
        if key_path == "data.metrics_path":
            return temp_metrics_path
        if key_path == "data.target_col":
            return "Churn"
        return original_get(self, key_path, default)

    with patch.object(AppConfig, "get", mock_get):
        with patch("pickle.load", side_effect=[mock_preprocessor, mock_model]):
            with (
                patch("mlflow.set_tracking_uri"),
                patch("mlflow.set_experiment"),
                patch("mlflow.start_run"),
                patch("mlflow.log_metrics"),
                patch("mlflow.log_artifact"),
            ):
                with patch("customer_churn.evaluate.RUN_ID_PATH", temp_run_id_path):
                    evaluate_model()

    # Assertions
    assert os.path.exists(temp_metrics_path)

    with open(temp_metrics_path, "r") as f_metrics:
        metrics = json.load(f_metrics)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics

    # Accuracy with predictions [0, 1, 0] vs targets [0, 1, 0] must be 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0
