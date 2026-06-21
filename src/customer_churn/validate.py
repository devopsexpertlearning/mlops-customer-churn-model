import json
import os
import pandas as pd
from typing import Any, Dict, List
from customer_churn.config import AppConfig
from customer_churn.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("validate")

# Define expected column names for schema verification
EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def validate_data() -> None:
    """Validates the raw customer churn dataset and writes a JSON report."""
    logger.info("Starting data validation...")
    try:
        config = AppConfig()
        raw_path = config.get("data.raw_path")
        report_path = config.get(
            "data.validation_report_path", "artifacts/validation_report.json"
        )

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file not found at {raw_path}")

        df = pd.read_csv(raw_path)
        logger.info(f"Loaded raw data of shape {df.shape} for validation")

        report: Dict[str, Any] = {
            "file_path": raw_path,
            "shape": list(df.shape),
            "validation_passed": True,
            "checks": {},
        }

        # Check 1: Columns presence
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        report["checks"]["column_presence"] = {
            "status": "PASSED" if not missing_cols else "FAILED",
            "missing_columns": missing_cols,
        }
        if missing_cols:
            report["validation_passed"] = False
            logger.error(f"Validation FAILED: Missing expected columns: {missing_cols}")

        # Check 2: Row count threshold (e.g. at least 1000 rows for training)
        min_rows = 1000
        row_count_status = "PASSED" if len(df) >= min_rows else "FAILED"
        report["checks"]["row_count"] = {
            "status": row_count_status,
            "rows": len(df),
            "min_required": min_rows,
        }
        if row_count_status == "FAILED":
            report["validation_passed"] = False
            logger.error(
                f"Validation FAILED: Row count {len(df)} is below minimum "
                f"threshold {min_rows}"
            )

        # Check 3: Missing values count
        missing_counts = df.isnull().sum().to_dict()
        missing_counts = {k: int(v) for k, v in missing_counts.items()}
        report["checks"]["missing_values"] = {
            "status": "PASSED",
            "counts": missing_counts,
        }

        # Check 4: Column ranges / domain constraints
        invalid_charges = (
            int((df["MonthlyCharges"] < 0).sum())
            if "MonthlyCharges" in df.columns
            else 0
        )
        invalid_tenure = int((df["tenure"] < 0).sum()) if "tenure" in df.columns else 0

        range_passed = invalid_charges == 0 and invalid_tenure == 0
        report["checks"]["value_ranges"] = {
            "status": "PASSED" if range_passed else "FAILED",
            "invalid_monthly_charges_count": invalid_charges,
            "invalid_tenure_count": invalid_tenure,
        }
        if not range_passed:
            report["validation_passed"] = False
            logger.error("Validation FAILED: Value range violations discovered.")

        # Check 5: Target column values
        target_col = config.get("data.target_col")
        target_passed = False
        target_values: List[str] = []
        if target_col in df.columns:
            target_values = [str(x) for x in df[target_col].unique()]
            target_passed = any(val in target_values for val in ["Yes", "No"])

        report["checks"]["target_column"] = {
            "status": "PASSED" if target_passed else "FAILED",
            "unique_values": target_values,
        }
        if not target_passed:
            report["validation_passed"] = False
            logger.error(
                f"Validation FAILED: Target column '{target_col}' has invalid "
                f"distribution: {target_values}"
            )

        # Save validation report
        report_dir = os.path.dirname(report_path)
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir, exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        if report["validation_passed"]:
            logger.info("Data validation completed successfully and passed all checks.")
        else:
            logger.warning("Data validation completed but some checks FAILED.")

    except Exception as e:
        logger.exception(f"Error occurred during data validation: {e}")
        raise


if __name__ == "__main__":
    validate_data()
