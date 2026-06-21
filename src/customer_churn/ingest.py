import os
import pandas as pd
from customer_churn.config import AppConfig
from customer_churn.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("ingest")


def ingest_data() -> None:
    """Ingests Telco Customer Churn dataset and saves it as raw data."""
    logger.info("Starting data ingestion process...")
    try:
        config = AppConfig()
        url = config.get("data.ingest_url")
        raw_path = config.get("data.raw_path")

        logger.info(f"Downloading raw data from URL: {url}")
        df = pd.read_csv(url)

        logger.info(f"Successfully downloaded dataset of shape: {df.shape}")

        raw_dir = os.path.dirname(raw_path)
        if raw_dir and not os.path.exists(raw_dir):
            logger.info(f"Creating directory: {raw_dir}")
            os.makedirs(raw_dir, exist_ok=True)

        logger.info(f"Saving raw data to {raw_path}")
        df.to_csv(raw_path, index=False)
        logger.info("Data ingestion completed successfully!")

    except Exception as e:
        logger.exception(f"Error occurred during data ingestion: {e}")
        raise


if __name__ == "__main__":
    ingest_data()
