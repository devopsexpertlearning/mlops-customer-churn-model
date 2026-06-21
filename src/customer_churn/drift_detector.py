"""Data drift detection module using Evidently AI."""

import argparse
import os
import sys
from typing import Tuple

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

from customer_churn.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("drift_detector")


def detect_drift(
    reference_path: str,
    current_path: str,
    html_report_path: str = "artifacts/drift_report.html",
    json_report_path: str = "artifacts/drift_report.json",
) -> Tuple[bool, float]:
    """Compares current data against reference data to identify feature drift.

    Returns:
        Tuple[bool, float]: (drift_detected, share_of_drifted_features)
    """
    logger.info(f"Loading reference data from {reference_path}")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    reference_df = pd.read_csv(reference_path)

    logger.info(f"Loading current data from {current_path}")
    if not os.path.exists(current_path):
        raise FileNotFoundError(f"Current file not found: {current_path}")
    current_df = pd.read_csv(current_path)

    # Exclude non-feature columns if they exist (like Churn or customerID)
    cols_to_exclude = ["customerID", "Churn"]
    features_to_compare = [
        col
        for col in reference_df.columns
        if col in current_df.columns and col not in cols_to_exclude
    ]

    ref_features = reference_df[features_to_compare]
    cur_features = current_df[features_to_compare]

    logger.info(
        f"Computing data drift on {len(features_to_compare)} features "
        f"using reference ({ref_features.shape[0]} rows) and "
        f"current ({cur_features.shape[0]} rows) datasets"
    )

    # Instantiate and run Evidently report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_features, current_data=cur_features)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(html_report_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_report_path), exist_ok=True)

    logger.info(f"Saving HTML report to {html_report_path}")
    report.save_html(html_report_path)

    logger.info(f"Saving JSON report to {json_report_path}")
    report.save_json(json_report_path)

    dict_report = report.as_dict()

    drift_detected = False
    share_of_drifted_features = 0.0

    try:
        metrics = dict_report.get("metrics", [])
        for m in metrics:
            if m.get("metric") == "DatasetDriftMetric":
                result = m.get("result", {})
                drift_detected = bool(result.get("dataset_drift", False))
                share_of_drifted_features = float(
                    result.get("share_of_drifted_columns", 0.0)
                )
                break
    except Exception as e:
        logger.error(f"Error parsing drift metrics from Evidently report: {e}")

    logger.info(
        f"Drift Analysis Complete. "
        f"Drift Detected: {drift_detected} "
        f"(Drifted Columns Share: {share_of_drifted_features:.2%})"
    )

    return drift_detected, share_of_drifted_features


def main() -> None:
    """CLI entrypoint for drift detection."""
    parser = argparse.ArgumentParser(
        description="Run Evidently AI Data Drift detection."
    )
    parser.add_argument(
        "--reference",
        default="data/processed/train.csv",
        help="Path to reference (training) dataset CSV",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current (inference logs / production) dataset CSV",
    )
    parser.add_argument(
        "--html-report",
        default="artifacts/drift_report.html",
        help="Path to output HTML report",
    )
    parser.add_argument(
        "--json-report",
        default="artifacts/drift_report.json",
        help="Path to output JSON report",
    )

    args = parser.parse_args()

    try:
        drifted, share = detect_drift(
            reference_path=args.reference,
            current_path=args.current,
            html_report_path=args.html_report,
            json_report_path=args.json_report,
        )
        if drifted:
            print("ALERT: Data drift detected!")
            sys.exit(1)
        else:
            print("SUCCESS: No data drift detected.")
            sys.exit(0)
    except Exception as e:
        logger.exception(f"Drift detection run failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
