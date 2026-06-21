"""Automated retraining trigger script based on data drift analysis."""

import argparse
import json
import os
import subprocess
import sys

from customer_churn.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("retrain")


def check_drift_and_trigger_retraining(
    json_report_path: str = "artifacts/drift_report.json",
    drift_threshold: float = 0.5,
    force_retrain: bool = False,
) -> bool:
    """Reads the Evidently JSON report, checks if drift exceeds threshold,
    and triggers retraining.

    Returns:
        bool: True if retraining was triggered, False otherwise.
    """
    if force_retrain:
        logger.info("Force retrain flag set. Triggering DVC pipeline retraining...")
        trigger_dvc_repro()
        return True

    logger.info(f"Reading drift report from {json_report_path}")
    if not os.path.exists(json_report_path):
        logger.warning(
            f"Drift report not found at {json_report_path}. " "No retraining triggered."
        )
        return False

    try:
        with open(json_report_path, "r") as f:
            report_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read or parse JSON drift report: {e}")
        return False

    drift_detected = False
    share_of_drifted_features = 0.0

    try:
        metrics = report_data.get("metrics", [])
        for m in metrics:
            if m.get("metric") == "DatasetDriftMetric":
                result = m.get("result", {})
                drift_detected = bool(result.get("dataset_drift", False))
                share_of_drifted_features = float(
                    result.get("share_of_drifted_columns", 0.0)
                )
                break
    except Exception as e:
        logger.error(f"Error parsing drift metrics from JSON report: {e}")
        return False

    logger.info(
        f"Drift check results: drift_detected={drift_detected}, "
        f"share={share_of_drifted_features:.2%} (threshold={drift_threshold:.2%})"
    )

    # Retrain if dataset_drift is True OR if the drifted share exceeds our threshold
    if drift_detected or share_of_drifted_features >= drift_threshold:
        logger.info(
            f"Drift threshold exceeded ({share_of_drifted_features:.2%} "
            f">= {drift_threshold:.2%}) or dataset drift flag is active. "
            "Triggering DVC pipeline retraining..."
        )
        trigger_dvc_repro()
        return True

    logger.info("No significant drift detected. Retraining skipped.")
    return False


def trigger_dvc_repro() -> None:
    """Executes 'dvc repro' to run the retraining pipeline."""
    try:
        logger.info("Running: dvc repro")
        # Run dvc repro and pipe output to stdout/stderr
        result = subprocess.run(
            ["dvc", "repro"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("DVC retraining pipeline completed successfully.")
        logger.debug(f"DVC output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC retraining pipeline failed with exit code {e.returncode}")
        logger.error(f"DVC stderr: {e.stderr}")
        raise RuntimeError("Retraining pipeline execution failed.") from e


def main() -> None:
    """CLI entrypoint for retraining scheduler."""
    parser = argparse.ArgumentParser(
        description="Automated retraining pipeline runner."
    )
    parser.add_argument(
        "--report",
        default="artifacts/drift_report.json",
        help="Path to Evidently JSON drift report",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Drift ratio threshold to trigger retraining",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force run retraining pipeline bypass drift check",
    )

    args = parser.parse_args()

    try:
        triggered = check_drift_and_trigger_retraining(
            json_report_path=args.report,
            drift_threshold=args.threshold,
            force_retrain=args.force,
        )
        if triggered:
            print("INFO: Retraining successfully triggered and executed.")
        else:
            print("INFO: Retraining not needed.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Retraining process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
