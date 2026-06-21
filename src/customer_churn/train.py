import os
import pickle
import yaml
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from customer_churn.config import AppConfig
from customer_churn.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("train")
RUN_ID_PATH = "artifacts/mlflow_run_id.txt"


def train_model() -> None:
    """Trains XGBoost using parameters from params.yaml and logs to MLflow."""
    logger.info("Starting model training stage...")
    try:
        config = AppConfig()
        train_features_path = config.get("data.features_train_path")
        model_path = config.get("model.path")
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

        logger.info(f"Loading training features from {train_features_path}")
        if not os.path.exists(train_features_path):
            raise FileNotFoundError(
                f"Training features file not found: {train_features_path}"
            )

        train_df = pd.read_csv(train_features_path)

        if target_col not in train_df.columns:
            raise KeyError(
                f"Target column '{target_col}' not found in training features"
            )

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        logger.info("Loading training parameters from params.yaml")
        if not os.path.exists("params.yaml"):
            raise FileNotFoundError("params.yaml file not found")

        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        train_params = params.get("train", {})
        learning_rate = float(
            config.get("train.learning_rate", train_params.get("learning_rate", 0.1))
        )
        max_depth = int(config.get("train.max_depth", train_params.get("max_depth", 6)))
        n_estimators = int(
            config.get("train.n_estimators", train_params.get("n_estimators", 100))
        )
        random_state = int(
            config.get("train.random_state", train_params.get("random_state", 42))
        )

        logger.info(
            f"Initializing XGBClassifier with learning_rate={learning_rate}, "
            f"max_depth={max_depth}, n_estimators={n_estimators}, "
            f"random_state={random_state}"
        )

        model = XGBClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=random_state,
            eval_metric="logloss",
        )

        actual_params = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "random_state": random_state,
        }

        with mlflow.start_run() as run:
            logger.info(f"Started MLflow run: {run.info.run_id}")
            logger.info("Logging hyperparameters to MLflow")
            mlflow.log_params(actual_params)

            logger.info("Fitting model on training features...")
            model.fit(X_train, y_train)

            # Ensure model directory exists
            model_dir = os.path.dirname(model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)

            logger.info(f"Saving serialized model locally to {model_path}")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            logger.info("Logging model to MLflow model registry/artifacts")
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                input_example=X_train.head(5),
            )

            # Save the active run ID to artifacts to share with evaluate stage
            os.makedirs(os.path.dirname(RUN_ID_PATH), exist_ok=True)
            logger.info(f"Saving active MLflow run ID to {RUN_ID_PATH}")
            with open(RUN_ID_PATH, "w") as f_run:
                f_run.write(run.info.run_id)

        logger.info("Model training stage completed successfully!")

    except Exception as e:
        logger.exception(f"Error occurred during model training: {e}")
        raise


if __name__ == "__main__":
    train_model()
