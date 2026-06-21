"""Tests for the data drift detection module."""

import os
import pathlib

import numpy as np
import pandas as pd

from customer_churn.drift_detector import detect_drift

MOCK_COLUMNS = [
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


def _create_mock_dataset(size: int = 50, drift: bool = False) -> pd.DataFrame:
    """Generates mock customer churn dataset."""
    rng = np.random.default_rng(42)

    data = {
        "gender": rng.choice(["Male", "Female"], size=size),
        "SeniorCitizen": rng.choice([0, 1], size=size),
        "Partner": rng.choice(["Yes", "No"], size=size),
        "Dependents": rng.choice(["Yes", "No"], size=size),
        "tenure": rng.integers(1, 72, size=size),
        "PhoneService": rng.choice(["Yes", "No"], size=size),
        "MultipleLines": rng.choice(["No", "Yes", "No phone service"], size=size),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=size),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], size=size),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=size),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size=size),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], size=size),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=size),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=size),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=size),
        "PaperlessBilling": rng.choice(["Yes", "No"], size=size),
        "PaymentMethod": rng.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            size=size,
        ),
        "MonthlyCharges": rng.uniform(18.0, 118.0, size=size),
        "TotalCharges": rng.uniform(18.0, 8000.0, size=size),
        "Churn": rng.choice([0, 1], size=size),
    }

    if drift:
        # Intentionally shift tenure and charges values to trigger drift
        data["tenure"] = data["tenure"] * 10
        data["MonthlyCharges"] = data["MonthlyCharges"] + 200.0
        data["TotalCharges"] = data["TotalCharges"] * 5

        # Shift categorical distributions to ensure dataset drift threshold (>50%)
        # is met.
        data["gender"] = np.array(["Other"] * size)
        data["Partner"] = np.array(["Unknown"] * size)
        data["Dependents"] = np.array(["Unknown"] * size)
        data["PhoneService"] = np.array(["No phone service"] * size)
        data["MultipleLines"] = np.array(["Yes"] * size)
        data["InternetService"] = np.array(["Satellite"] * size)
        data["Contract"] = np.array(["Three year"] * size)
        data["PaymentMethod"] = np.array(["Bitcoin"] * size)

    return pd.DataFrame(data)


def test_drift_detector_no_drift(tmp_path: pathlib.Path) -> None:
    """Verifies that no drift is reported for identical distributions."""
    ref_path = os.path.join(tmp_path, "ref.csv")
    cur_path = os.path.join(tmp_path, "cur.csv")

    html_path = os.path.join(tmp_path, "report.html")
    json_path = os.path.join(tmp_path, "report.json")

    # Generate identical distributions
    ref_df = _create_mock_dataset(size=100, drift=False)
    cur_df = _create_mock_dataset(size=100, drift=False)

    ref_df.to_csv(ref_path, index=False)
    cur_df.to_csv(cur_path, index=False)

    drift_detected, share = detect_drift(
        reference_path=ref_path,
        current_path=cur_path,
        html_report_path=html_path,
        json_report_path=json_path,
    )

    assert drift_detected is False
    assert share < 0.5
    assert os.path.exists(html_path)
    assert os.path.exists(json_path)


def test_drift_detector_with_drift(tmp_path: pathlib.Path) -> None:
    """Verifies that data drift is correctly detected and reported."""
    ref_path = os.path.join(tmp_path, "ref.csv")
    cur_path = os.path.join(tmp_path, "cur_shifted.csv")

    html_path = os.path.join(tmp_path, "report.html")
    json_path = os.path.join(tmp_path, "report.json")

    # Generate drifted distribution for current
    ref_df = _create_mock_dataset(size=100, drift=False)
    cur_df = _create_mock_dataset(size=100, drift=True)

    ref_df.to_csv(ref_path, index=False)
    cur_df.to_csv(cur_path, index=False)

    drift_detected, share = detect_drift(
        reference_path=ref_path,
        current_path=cur_path,
        html_report_path=html_path,
        json_report_path=json_path,
    )

    assert drift_detected is True
    assert share >= 0.5
    assert os.path.exists(html_path)
    assert os.path.exists(json_path)
