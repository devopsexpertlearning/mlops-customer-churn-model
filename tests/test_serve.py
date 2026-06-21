"""Tests for the FastAPI serving module."""

import os
import pathlib
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from customer_churn.serve import app, load_artifacts

SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 840.50,
}

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


def _build_test_artifacts(tmp_path: pathlib.Path) -> tuple:
    """Creates mock model and preprocessor artifacts for testing."""
    # Build a small training dataset to fit the preprocessor
    train_data = pd.DataFrame(
        [
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 10,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.0,
                "TotalCharges": 700.0,
            },
            {
                "gender": "Female",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "Yes",
                "tenure": 50,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 45.0,
                "TotalCharges": 2250.0,
            },
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ]
    )
    preprocessor.fit(train_data)

    # Create a mock model with predict and predict_proba using a real scikit-learn model
    from sklearn.tree import DecisionTreeClassifier

    mock_model = DecisionTreeClassifier(random_state=42)
    mock_model.fit(preprocessor.transform(train_data), np.array([1, 0]))

    model_path = os.path.join(tmp_path, "model.pkl")
    preprocessor_path = os.path.join(tmp_path, "preprocessor.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(mock_model, f)
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)

    return model_path, preprocessor_path


def test_predict_endpoint(tmp_path: pathlib.Path) -> None:
    """Tests the /predict endpoint with valid input."""
    model_path, preprocessor_path = _build_test_artifacts(tmp_path)

    with (
        patch.dict(
            os.environ,
            {
                "MODEL_PATH": model_path,
                "PREPROCESSOR_PATH": preprocessor_path,
            },
        ),
    ):
        load_artifacts()

    client = TestClient(app)
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200

    body = response.json()
    assert "prediction" in body
    assert "churn_probability" in body
    assert "label" in body
    assert body["prediction"] in [0, 1]
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["label"] in ["Churn", "No Churn"]


def test_predict_batch_endpoint(
    tmp_path: pathlib.Path,
) -> None:
    """Tests the /predict/batch endpoint."""
    model_path, preprocessor_path = _build_test_artifacts(tmp_path)

    with patch.dict(
        os.environ,
        {
            "MODEL_PATH": model_path,
            "PREPROCESSOR_PATH": preprocessor_path,
        },
    ):
        load_artifacts()

    client = TestClient(app)
    batch_request = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
    response = client.post("/predict/batch", json=batch_request)
    assert response.status_code == 200

    body = response.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 2


def test_health_endpoint(tmp_path: pathlib.Path) -> None:
    """Tests the /health endpoint."""
    model_path, preprocessor_path = _build_test_artifacts(tmp_path)

    with patch.dict(
        os.environ,
        {
            "MODEL_PATH": model_path,
            "PREPROCESSOR_PATH": preprocessor_path,
        },
    ):
        load_artifacts()

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["preprocessor_loaded"] is True


def test_metrics_endpoint(tmp_path: pathlib.Path) -> None:
    """Tests the /metrics endpoint returns Prometheus format."""
    model_path, preprocessor_path = _build_test_artifacts(tmp_path)

    with patch.dict(
        os.environ,
        {
            "MODEL_PATH": model_path,
            "PREPROCESSOR_PATH": preprocessor_path,
        },
    ):
        load_artifacts()

    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "churn_predictions_total" in response.text


def test_predict_invalid_input() -> None:
    """Tests that the /predict endpoint rejects invalid input."""
    client = TestClient(app)
    response = client.post("/predict", json={"invalid": "data"})
    assert response.status_code == 422
