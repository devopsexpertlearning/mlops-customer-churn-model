import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from customer_churn.config import AppConfig
from customer_churn.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("preprocess")


def preprocess_data() -> None:
    """Preprocesses raw data: cleans TotalCharges, splits train/test, and saves."""
    logger.info("Starting data preprocessing stage...")
    try:
        config = AppConfig()
        raw_path = config.get("data.raw_path")
        train_path = config.get("data.processed_train_path")
        test_path = config.get("data.processed_test_path")

        # Load parameters
        if not os.path.exists("params.yaml"):
            raise FileNotFoundError("params.yaml file not found")

        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        test_size = params["train"]["test_size"]
        random_state = params["train"]["random_state"]

        logger.info(f"Loading raw data from {raw_path}")
        df = pd.read_csv(raw_path)

        # 1. Drop customerID
        if "customerID" in df.columns:
            logger.info("Dropping customerID column")
            df = df.drop(columns=["customerID"])

        # 2. Clean TotalCharges (convert empty spaces to 0 and cast to float)
        if "TotalCharges" in df.columns:
            logger.info("Cleaning TotalCharges column")
            # Replace empty strings or whitespaces with None
            df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", None, regex=True)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            # Fill NaN values with 0.0 (new customers with tenure = 0)
            df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

        # 3. Map target column Churn to numeric (Yes -> 1, No -> 0)
        target_col = config.get("data.target_col")
        if target_col in df.columns:
            logger.info(f"Mapping target column '{target_col}' to binary labels")
            df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

        # 4. Train-test split (stratified by target variable)
        logger.info(
            f"Splitting data with test_size={test_size}, "
            f"random_state={random_state}"
        )
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_col] if target_col in df.columns else None,
        )

        # Ensure processed directory exists
        for path in [train_path, test_path]:
            processed_dir = os.path.dirname(path)
            if processed_dir and not os.path.exists(processed_dir):
                os.makedirs(processed_dir, exist_ok=True)

        logger.info(f"Saving training set of shape {train_df.shape} to {train_path}")
        train_df.to_csv(train_path, index=False)

        logger.info(f"Saving testing set of shape {test_df.shape} to {test_path}")
        test_df.to_csv(test_path, index=False)

        logger.info("Preprocessing completed successfully!")

    except Exception as e:
        logger.exception(f"Error occurred during data preprocessing: {e}")
        raise


if __name__ == "__main__":
    preprocess_data()
