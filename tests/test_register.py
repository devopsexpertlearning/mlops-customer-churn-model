import json
import os
import pathlib
from typing import Any
from unittest.mock import MagicMock, patch
from mlflow.exceptions import MlflowException
from customer_churn.config import AppConfig
from customer_churn.register_model import register_and_promote_model


def test_register_and_promote_first_time(tmp_path: pathlib.Path) -> None:
    """Test model registration and promotion.

    Case: no previous production model exists.
    """
    # Write mock run ID file
    run_id_file = os.path.join(tmp_path, "mlflow_run_id.txt")
    with open(run_id_file, "w") as f:
        f.write("run_123")

    # Write mock metrics file
    metrics_file = os.path.join(tmp_path, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump({"f1_score": 0.85}, f)

    status_file = os.path.join(tmp_path, "registration_status.json")

    # Mock AppConfig config paths/values
    original_get = AppConfig.get

    def mock_get(self: AppConfig, key_path: str, default: Any = None) -> Any:
        if key_path == "mlflow.tracking_uri":
            return "http://localhost:5000"
        if key_path == "model.name":
            return "xgboost-churn-classifier"
        return original_get(self, key_path, default)

    # Mock MLflow
    mock_version = MagicMock()
    mock_version.version = "1"

    mock_client = MagicMock()
    # Mock status check loop
    v_details = MagicMock()
    v_details.status = "READY"
    mock_client.get_model_version.return_value = v_details
    # Mock no existing production model alias
    mock_client.get_model_version_by_alias.side_effect = MlflowException(
        "Model or alias not found"
    )

    with patch.object(AppConfig, "get", mock_get):
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.register_model", return_value=mock_version) as mock_register,
            patch(
                "customer_churn.register_model.MlflowClient", return_value=mock_client
            ),
            patch("customer_churn.register_model.RUN_ID_PATH", run_id_file),
            patch("customer_churn.register_model.METRICS_PATH", metrics_file),
            patch("customer_churn.register_model.STATUS_PATH", status_file),
        ):
            register_and_promote_model()

            # Assertions
            mock_register.assert_called_once_with(
                model_uri="runs:/run_123/model", name="xgboost-churn-classifier"
            )
            mock_client.get_model_version_by_alias.assert_called_once_with(
                "xgboost-churn-classifier", "production"
            )
            mock_client.set_registered_model_alias.assert_called_once_with(
                "xgboost-churn-classifier", "production", "1"
            )

            # Check status file content
            assert os.path.exists(status_file)
            with open(status_file, "r") as f_status:
                status_info = json.load(f_status)
                assert status_info["status"] == "promoted"
                assert status_info["candidate_run_id"] == "run_123"
                assert status_info["candidate_f1_score"] == 0.85
                assert status_info["previous_production_version"] is None
                assert status_info["previous_production_f1_score"] is None
                assert status_info["active_production_version"] == 1


def test_register_and_promote_better_score(tmp_path: pathlib.Path) -> None:
    """Test model registration and promotion.

    Case: candidate F1-score is strictly better than production.
    """
    # Write mock run ID file
    run_id_file = os.path.join(tmp_path, "mlflow_run_id.txt")
    with open(run_id_file, "w") as f:
        f.write("run_456")

    # Write mock metrics file
    metrics_file = os.path.join(tmp_path, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump({"f1_score": 0.88}, f)

    status_file = os.path.join(tmp_path, "registration_status.json")

    # Mock AppConfig config paths/values
    original_get = AppConfig.get

    def mock_get(self: AppConfig, key_path: str, default: Any = None) -> Any:
        if key_path == "mlflow.tracking_uri":
            return "http://localhost:5000"
        if key_path == "model.name":
            return "xgboost-churn-classifier"
        return original_get(self, key_path, default)

    # Mock MLflow
    mock_version = MagicMock()
    mock_version.version = "2"

    mock_client = MagicMock()
    # Mock status check loop
    v_details = MagicMock()
    v_details.status = "READY"
    mock_client.get_model_version.return_value = v_details

    # Mock existing production model version 1 with F1=0.85
    prod_mv = MagicMock()
    prod_mv.version = "1"
    prod_mv.run_id = "run_123"
    mock_client.get_model_version_by_alias.return_value = prod_mv

    prod_run = MagicMock()
    prod_run.data.metrics = {"f1_score": 0.85}
    mock_client.get_run.return_value = prod_run

    with patch.object(AppConfig, "get", mock_get):
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.register_model", return_value=mock_version) as mock_register,
            patch(
                "customer_churn.register_model.MlflowClient", return_value=mock_client
            ),
            patch("customer_churn.register_model.RUN_ID_PATH", run_id_file),
            patch("customer_churn.register_model.METRICS_PATH", metrics_file),
            patch("customer_churn.register_model.STATUS_PATH", status_file),
        ):
            register_and_promote_model()

            # Assertions
            mock_register.assert_called_once_with(
                model_uri="runs:/run_456/model", name="xgboost-churn-classifier"
            )
            mock_client.get_model_version_by_alias.assert_called_once_with(
                "xgboost-churn-classifier", "production"
            )
            mock_client.set_registered_model_alias.assert_called_once_with(
                "xgboost-churn-classifier", "production", "2"
            )

            # Check status file content
            assert os.path.exists(status_file)
            with open(status_file, "r") as f_status:
                status_info = json.load(f_status)
                assert status_info["status"] == "promoted"
                assert status_info["candidate_run_id"] == "run_456"
                assert status_info["candidate_f1_score"] == 0.88
                assert status_info["previous_production_version"] == 1
                assert status_info["previous_production_f1_score"] == 0.85
                assert status_info["active_production_version"] == 2


def test_register_and_promote_worse_score(tmp_path: pathlib.Path) -> None:
    """Test model registration.

    Case: promotion is skipped due to lower/equal F1-score.
    """
    # Write mock run ID file
    run_id_file = os.path.join(tmp_path, "mlflow_run_id.txt")
    with open(run_id_file, "w") as f:
        f.write("run_789")

    # Write mock metrics file (F1 is lower than existing production)
    metrics_file = os.path.join(tmp_path, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump({"f1_score": 0.82}, f)

    status_file = os.path.join(tmp_path, "registration_status.json")

    # Mock AppConfig config paths/values
    original_get = AppConfig.get

    def mock_get(self: AppConfig, key_path: str, default: Any = None) -> Any:
        if key_path == "mlflow.tracking_uri":
            return "http://localhost:5000"
        if key_path == "model.name":
            return "xgboost-churn-classifier"
        return original_get(self, key_path, default)

    # Mock MLflow
    mock_version = MagicMock()
    mock_version.version = "3"

    mock_client = MagicMock()
    # Mock status check loop
    v_details = MagicMock()
    v_details.status = "READY"
    mock_client.get_model_version.return_value = v_details

    # Mock existing production model version 2 with F1=0.88
    prod_mv = MagicMock()
    prod_mv.version = "2"
    prod_mv.run_id = "run_456"
    mock_client.get_model_version_by_alias.return_value = prod_mv

    prod_run = MagicMock()
    prod_run.data.metrics = {"f1_score": 0.88}
    mock_client.get_run.return_value = prod_run

    with patch.object(AppConfig, "get", mock_get):
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.register_model", return_value=mock_version) as mock_register,
            patch(
                "customer_churn.register_model.MlflowClient", return_value=mock_client
            ),
            patch("customer_churn.register_model.RUN_ID_PATH", run_id_file),
            patch("customer_churn.register_model.METRICS_PATH", metrics_file),
            patch("customer_churn.register_model.STATUS_PATH", status_file),
        ):
            register_and_promote_model()

            # Assertions
            mock_register.assert_called_once_with(
                model_uri="runs:/run_789/model", name="xgboost-churn-classifier"
            )
            mock_client.get_model_version_by_alias.assert_called_once_with(
                "xgboost-churn-classifier", "production"
            )
            mock_client.set_registered_model_alias.assert_not_called()

            # Check status file content
            assert os.path.exists(status_file)
            with open(status_file, "r") as f_status:
                status_info = json.load(f_status)
                assert status_info["status"] == "registered_but_not_promoted"
                assert status_info["candidate_run_id"] == "run_789"
                assert status_info["candidate_f1_score"] == 0.82
                assert status_info["previous_production_version"] == 2
                assert status_info["previous_production_f1_score"] == 0.88
                assert status_info["active_production_version"] == 2
