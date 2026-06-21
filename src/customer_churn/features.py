import os
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from customer_churn.config import AppConfig
from customer_churn.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("features")

# Define columns explicitly
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def engineer_features() -> None:
    """Fits categorical encoding and scaling to train data and transforms splits."""
    logger.info("Starting feature engineering stage...")
    try:
        config = AppConfig()
        train_path = config.get("data.processed_train_path")
        test_path = config.get("data.processed_test_path")
        train_out_path = config.get("data.features_train_path")
        test_out_path = config.get("data.features_test_path")
        preprocessor_path = config.get("model.preprocessor_path")

        logger.info(f"Loading train split from {train_path}")
        train_df = pd.read_csv(train_path)
        logger.info(f"Loading test split from {test_path}")
        test_df = pd.read_csv(test_path)

        target_col = config.get("data.target_col")

        # Split features and target to prevent leakage
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        logger.info("Initializing preprocessor column transformer")
        # Define pipeline: Standard scaler for numeric, OneHot for categorical
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERICAL_COLS),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL_COLS,
                ),
            ],
            remainder="passthrough",
        )

        logger.info("Fitting preprocessor pipeline on training features")
        X_train_transformed = preprocessor.fit_transform(X_train)
        logger.info("Transforming testing features using fitted preprocessor")
        X_test_transformed = preprocessor.transform(X_test)

        # Retrieve feature names out
        cat_features = (
            preprocessor.named_transformers_["cat"]
            .get_feature_names_out(CATEGORICAL_COLS)
            .tolist()
        )
        feature_names = NUMERICAL_COLS + cat_features

        # Reconstruct DataFrames
        train_features_df = pd.DataFrame(X_train_transformed, columns=feature_names)
        train_features_df[target_col] = y_train.values

        test_features_df = pd.DataFrame(X_test_transformed, columns=feature_names)
        test_features_df[target_col] = y_test.values

        # Ensure model directory exists for preprocessor serialization
        models_dir = os.path.dirname(preprocessor_path)
        if models_dir and not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)

        logger.info(f"Serializing fitted preprocessor pipeline to {preprocessor_path}")
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)

        # Save engineered outputs
        logger.info(
            "Saving engineered train features of shape "
            f"{train_features_df.shape} to {train_out_path}"
        )
        train_features_df.to_csv(train_out_path, index=False)

        logger.info(
            "Saving engineered test features of shape "
            f"{test_features_df.shape} to {test_out_path}"
        )
        test_features_df.to_csv(test_out_path, index=False)

        logger.info("Feature engineering stage completed successfully!")

    except Exception as e:
        logger.exception(f"Error occurred during feature engineering: {e}")
        raise


if __name__ == "__main__":
    engineer_features()
