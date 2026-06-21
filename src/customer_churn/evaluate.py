import json
import os
import pickle
import pandas as pd
import mlflow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from customer_churn.config import AppConfig
from customer_churn.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("evaluate")
RUN_ID_PATH = "artifacts/mlflow_run_id.txt"


def evaluate_model() -> None:
    """Evaluates the trained model on test data, logs to MLflow, and exports metrics."""
    logger.info("Starting model evaluation stage...")
    try:
        config = AppConfig()
        test_path = config.get("data.processed_test_path")
        preprocessor_path = config.get("model.preprocessor_path")
        model_path = config.get("model.path")
        metrics_path = config.get("data.metrics_path")
        target_col = config.get("data.target_col")

        # Configure MLflow
        tracking_uri = config.get("mlflow.tracking_uri", "http://localhost:5000")
        experiment_name = config.get(
            "mlflow.experiment_name", "customer-churn-prediction"
        )

        logger.info(f"Setting MLflow tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Setting MLflow experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)

        logger.info(f"Loading test split from {test_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test split file not found: {test_path}")
        test_df = pd.read_csv(test_path)

        if target_col not in test_df.columns:
            raise KeyError(f"Target column '{target_col}' not found in test dataset")

        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        logger.info(f"Loading preprocessor from {preprocessor_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)

        logger.info(f"Loading trained model from {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model file not found: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logger.info("Transforming test features using preprocessor")
        X_test_transformed = preprocessor.transform(X_test)

        logger.info("Generating predictions and probability scores")
        preds = model.predict(X_test_transformed)
        probs = model.predict_proba(X_test_transformed)[:, 1]

        # Calculate metrics
        accuracy = float(accuracy_score(y_test, preds))
        precision = float(precision_score(y_test, preds, zero_division=0))
        recall = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, probs))

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
        }

        logger.info(f"Evaluation metrics computed: {metrics}")

        # Ensure metrics parent directory exists
        metrics_dir = os.path.dirname(metrics_path)
        if metrics_dir and not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=True)

        logger.info(f"Saving metrics report locally to {metrics_path}")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Check if active MLflow run ID exists from train stage to resume
        run_id = None
        if os.path.exists(RUN_ID_PATH):
            with open(RUN_ID_PATH, "r") as f_run:
                run_id = f_run.read().strip()
                logger.info(f"Found active MLflow run ID to resume: {run_id}")

        with mlflow.start_run(run_id=run_id):
            logger.info("Logging metrics to MLflow")
            mlflow.log_metrics(metrics)
            logger.info("Logging metrics report JSON as MLflow artifact")
            mlflow.log_artifact(metrics_path, artifact_path="metrics")

        logger.info("Model evaluation stage completed successfully!")

    except Exception as e:
        logger.exception(f"Error occurred during model evaluation: {e}")
        raise


if __name__ == "__main__":
    evaluate_model()
