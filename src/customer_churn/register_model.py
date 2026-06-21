import json
import os
import time
from typing import Any, Dict, Optional
import mlflow
from mlflow.client import MlflowClient
from mlflow.exceptions import MlflowException
from customer_churn.config import AppConfig
from customer_churn.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("register_model")

RUN_ID_PATH = "artifacts/mlflow_run_id.txt"
METRICS_PATH = "artifacts/metrics.json"
STATUS_PATH = "artifacts/registration_status.json"


def register_and_promote_model() -> None:
    """Registers the candidate model.

    Promotes it to 'production' if its F1-score is better.
    """
    logger.info("Starting model registration and promotion stage...")
    try:
        config = AppConfig()
        tracking_uri = config.get("mlflow.tracking_uri", "http://localhost:5000")
        model_name = config.get("model.name", "xgboost-churn-classifier")

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)

        # 1. Read active run ID
        if not os.path.exists(RUN_ID_PATH):
            raise FileNotFoundError(f"Active run ID file not found at {RUN_ID_PATH}")
        with open(RUN_ID_PATH, "r") as f:
            run_id = f.read().strip()
        logger.info(f"Loaded active run ID: {run_id}")

        # 2. Read local metrics
        if not os.path.exists(METRICS_PATH):
            raise FileNotFoundError(f"Metrics file not found at {METRICS_PATH}")
        with open(METRICS_PATH, "r") as f:
            metrics: Dict[str, Any] = json.load(f)
        new_f1 = float(metrics.get("f1_score", 0.0))
        logger.info(f"Candidate model F1-score: {new_f1:.4f}")

        # 3. Register the candidate model version
        model_uri = f"runs:/{run_id}/model"
        logger.info(
            f"Registering model version from {model_uri} " f"under name '{model_name}'"
        )
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

        # Poll until the model version is ready
        logger.info(
            f"Waiting for model version {model_version.version} " "to become READY..."
        )
        for i in range(10):
            v_details = client.get_model_version(model_name, model_version.version)
            if v_details.status == "READY":
                logger.info(f"Model version {model_version.version} is READY.")
                break
            time.sleep(1)
        else:
            raise TimeoutError(
                f"Model version {model_version.version} "
                "did not reach READY status in time."
            )

        # 4. Check current production model version
        prod_version: Optional[str] = None
        prod_f1: Optional[float] = None
        try:
            prod_mv = client.get_model_version_by_alias(model_name, "production")
            prod_version = prod_mv.version
            logger.info(
                f"Found existing production model version: {prod_version} "
                f"(run ID: {prod_mv.run_id})"
            )
            # Retrieve metrics for the existing production run
            if not prod_mv.run_id:
                raise ValueError(
                    f"Production model version {prod_version} has no run ID."
                )
            prod_run = client.get_run(prod_mv.run_id)
            prod_f1 = float(prod_run.data.metrics.get("f1_score", 0.0))
            logger.info(f"Existing production model F1-score: {prod_f1:.4f}")
        except MlflowException as e:
            # MlflowException is raised if model or alias is not found
            logger.info(
                "No existing production model or alias 'production' "
                f"found. Details: {e}"
            )

        # 5. Decide promotion based on F1-score comparison
        promote = False
        if prod_f1 is None:
            logger.info("Promoting model because no production model currently exists.")
            promote = True
        elif new_f1 > prod_f1:
            logger.info(
                f"Promoting model: candidate F1 ({new_f1:.4f}) "
                f"> production F1 ({prod_f1:.4f})"
            )
            promote = True
        else:
            logger.info(
                f"Candidate F1 ({new_f1:.4f}) <= production "
                f"F1 ({prod_f1:.4f}). Promotion skipped."
            )

        status_info: Dict[str, Any] = {
            "model_name": model_name,
            "candidate_version": int(model_version.version),
            "candidate_run_id": run_id,
            "candidate_f1_score": new_f1,
            "previous_production_version": (
                int(prod_version) if prod_version is not None else None
            ),
            "previous_production_f1_score": prod_f1,
        }

        if promote:
            logger.info(
                f"Setting 'production' alias on model version {model_version.version}"
            )
            client.set_registered_model_alias(
                model_name, "production", model_version.version
            )
            status_info["status"] = "promoted"
            status_info["active_production_version"] = int(model_version.version)
        else:
            status_info["status"] = "registered_but_not_promoted"
            status_info["active_production_version"] = (
                int(prod_version) if prod_version is not None else None
            )

        # 6. Save registration status report
        os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
        with open(STATUS_PATH, "w") as f_status:
            json.dump(status_info, f_status, indent=4)
        logger.info(f"Registration status report written to {STATUS_PATH}")

    except Exception as e:
        logger.exception(f"Error during model registration: {e}")
        raise


if __name__ == "__main__":
    register_and_promote_model()
