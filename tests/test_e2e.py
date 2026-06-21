"""End-to-end integration test for the churn prediction workflow."""

import os
import pathlib
import shutil
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from customer_churn.drift_detector import detect_drift
from customer_churn.evaluate import evaluate_model
from customer_churn.features import engineer_features
from customer_churn.ingest import ingest_data
from customer_churn.preprocess import preprocess_data
from customer_churn.register_model import register_and_promote_model
from customer_churn.retrain import check_drift_and_trigger_retraining
from customer_churn.serve import app, load_artifacts
from customer_churn.train import train_model
from customer_churn.validate import validate_data


@pytest.mark.integration
def test_end_to_end_workflow(tmp_path: pathlib.Path) -> None:
    """Runs the entire pipeline sequentially and tests the serving layer."""
    # Define temporary files/directories for this test run
    test_db_path = os.path.join(tmp_path, "test_mlflow.db")
    test_model_path = os.path.join(tmp_path, "model.pkl")
    test_preprocessor_path = os.path.join(tmp_path, "preprocessor.pkl")
    test_drift_json = os.path.join(tmp_path, "drift_report.json")
    test_drift_html = os.path.join(tmp_path, "drift_report.html")

    # Set up environmental patches to isolate MLflow and paths
    env_patches = {
        "MLFLOW_TRACKING_URI": f"sqlite:///{test_db_path}",
        "MLFLOW_S3_IGNORE_TLS": "true",
        "MODEL_PATH": test_model_path,
        "PREPROCESSOR_PATH": test_preprocessor_path,
        # Mock AWS keys so MLflow local client can run SQLite tracking without S3 errors
        "AWS_ACCESS_KEY_ID": "mock",
        "AWS_SECRET_ACCESS_KEY": "mock",
    }

    with patch.dict(os.environ, env_patches):
        # 1. Ingest Data
        print("Starting E2E step: Ingestion")
        ingest_data()
        assert os.path.exists("data/raw/telco_customer_churn.csv")

        # 2. Validate Data
        print("Starting E2E step: Validation")
        validate_data()
        assert os.path.exists("artifacts/validation_report.json")

        # 3. Preprocess Data
        print("Starting E2E step: Preprocessing")
        preprocess_data()
        assert os.path.exists("data/processed/train.csv")
        assert os.path.exists("data/processed/test.csv")

        # 4. Feature Engineering
        print("Starting E2E step: Feature Engineering")
        engineer_features()
        assert os.path.exists("data/processed/train_features.csv")
        assert os.path.exists("data/processed/test_features.csv")

        # If preprocessor was created in default location, copy to test location
        default_preprocessor_path = "models/preprocessor.pkl"
        if os.path.exists(default_preprocessor_path) and not os.path.exists(
            test_preprocessor_path
        ):
            shutil.copy(default_preprocessor_path, test_preprocessor_path)

        # 5. Train Model
        print("Starting E2E step: Model Training")
        train_model()

        # If the model was created in the default location, copy it to our test location
        default_model_path = "models/model.pkl"
        if os.path.exists(default_model_path) and not os.path.exists(test_model_path):
            shutil.copy(default_model_path, test_model_path)

        assert os.path.exists(test_model_path)

        # 6. Evaluate Model
        print("Starting E2E step: Model Evaluation")
        evaluate_model()
        assert os.path.exists("artifacts/metrics.json")
        assert os.path.exists("artifacts/mlflow_run_id.txt")

        # 7. Register Model (uses SQLITE test db)
        print("Starting E2E step: Model Registration")
        register_and_promote_model()
        assert os.path.exists("artifacts/registration_status.json")

        # 8. Test Serving API Layer
        print("Starting E2E step: FastAPI Serving verification")
        load_artifacts()

        client = TestClient(app)

        # Verify /health endpoint
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"

        # Verify /predict endpoint
        sample_payload = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.35,
            "TotalCharges": 840.50,
        }
        predict_resp = client.post("/predict", json=sample_payload)
        assert predict_resp.status_code == 200
        assert "prediction" in predict_resp.json()

        # 9. Verify Drift Detection
        print("Starting E2E step: Drift Detection")
        drift_detected, share = detect_drift(
            reference_path="data/processed/train.csv",
            current_path="data/processed/test.csv",
            html_report_path=test_drift_html,
            json_report_path=test_drift_json,
        )
        assert os.path.exists(test_drift_html)
        assert os.path.exists(test_drift_json)

        # 10. Verify Retraining triggering check
        print("Starting E2E step: Retraining checks")
        with patch("customer_churn.retrain.subprocess.run") as mock_sub_run:
            mock_sub_run.return_value = MagicMock(returncode=0, stdout="success")
            triggered = check_drift_and_trigger_retraining(
                json_report_path=test_drift_json,
                drift_threshold=0.5,
            )
            assert triggered is False
            mock_sub_run.assert_not_called()
