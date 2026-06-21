"""Tests for the automated retraining trigger module."""

import json
import os
import pathlib
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from customer_churn.retrain import check_drift_and_trigger_retraining


def _create_mock_drift_report(
    tmp_path: pathlib.Path, drift_detected: bool, share: float
) -> str:
    """Helper to write a mock JSON drift report."""
    report_data = {
        "metrics": [
            {
                "metric": "DatasetDriftMetric",
                "result": {
                    "dataset_drift": drift_detected,
                    "share_of_drifted_columns": share,
                },
            }
        ]
    }
    report_path = os.path.join(tmp_path, "drift_report.json")
    with open(report_path, "w") as f:
        json.dump(report_data, f)
    return report_path


@patch("customer_churn.retrain.subprocess.run")
def test_retrain_triggered_due_to_drift(
    mock_run: MagicMock, tmp_path: pathlib.Path
) -> None:
    """Verifies retraining is triggered when drift is detected."""
    report_path = _create_mock_drift_report(tmp_path, drift_detected=True, share=0.6)

    # Mock successful execution of dvc repro
    mock_run.return_value = MagicMock(returncode=0, stdout="success")

    triggered = check_drift_and_trigger_retraining(
        json_report_path=report_path, drift_threshold=0.5
    )

    assert triggered is True
    mock_run.assert_called_once_with(
        ["dvc", "repro"], check=True, capture_output=True, text=True
    )


@patch("customer_churn.retrain.subprocess.run")
def test_retrain_skipped_no_drift(mock_run: MagicMock, tmp_path: pathlib.Path) -> None:
    """Verifies retraining is skipped when drift is below threshold."""
    report_path = _create_mock_drift_report(tmp_path, drift_detected=False, share=0.2)

    triggered = check_drift_and_trigger_retraining(
        json_report_path=report_path, drift_threshold=0.5
    )

    assert triggered is False
    mock_run.assert_not_called()


@patch("customer_churn.retrain.subprocess.run")
def test_retrain_force_flag(mock_run: MagicMock, tmp_path: pathlib.Path) -> None:
    """Verifies retraining is triggered when force is set, ignoring the report."""
    mock_run.return_value = MagicMock(returncode=0, stdout="success")

    triggered = check_drift_and_trigger_retraining(
        json_report_path="non_existent.json", drift_threshold=0.5, force_retrain=True
    )

    assert triggered is True
    mock_run.assert_called_once_with(
        ["dvc", "repro"], check=True, capture_output=True, text=True
    )


@patch("customer_churn.retrain.subprocess.run")
def test_retrain_handles_dvc_failure(
    mock_run: MagicMock, tmp_path: pathlib.Path
) -> None:
    """Verifies retraining handler raises exception on subprocess failure."""
    report_path = _create_mock_drift_report(tmp_path, drift_detected=True, share=0.7)

    mock_run.side_effect = subprocess.CalledProcessError(
        1, ["dvc", "repro"], stderr="error"
    )

    with pytest.raises(RuntimeError, match="Retraining pipeline execution failed."):
        check_drift_and_trigger_retraining(
            json_report_path=report_path, drift_threshold=0.5
        )
