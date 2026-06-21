"""FastAPI serving module for the Customer Churn Prediction model.

Exposes endpoints for:
- POST /predict — Real-time churn prediction
- GET /health — Liveness/readiness check
- GET /metrics — Prometheus-compatible metrics
"""

import os
import pickle
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel, Field

from customer_churn.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("serve")

# --- Prometheus Metrics ---
PREDICTION_COUNTER = Counter(
    "churn_predictions_total",
    "Total number of churn predictions served",
    ["predicted_class"],
)
PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Latency of churn prediction requests in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
PREDICTION_ERRORS = Counter(
    "churn_prediction_errors_total",
    "Total number of prediction errors",
)

# --- Global model state ---
MODEL: Any = None
PREPROCESSOR: Any = None


# --- Pydantic Schemas ---
class CustomerFeatures(BaseModel):
    """Input schema for a single customer prediction request."""

    gender: str = Field(..., examples=["Male"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[70.35])
    TotalCharges: float = Field(..., ge=0, examples=[840.50])


class PredictionResponse(BaseModel):
    """Output schema for a churn prediction response."""

    prediction: int = Field(..., description="0 = No Churn, 1 = Churn")
    churn_probability: float = Field(
        ..., description="Probability of churn (0.0 to 1.0)"
    )
    label: str = Field(..., description="Human-readable label: 'Churn' or 'No Churn'")


class BatchPredictionRequest(BaseModel):
    """Input schema for batch prediction requests."""

    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    """Output schema for batch prediction responses."""

    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    model_loaded: bool
    preprocessor_loaded: bool


def load_artifacts() -> None:
    """Loads the trained model and preprocessor from disk."""
    global MODEL, PREPROCESSOR

    model_path = os.environ.get("MODEL_PATH", "models/model.pkl")
    preprocessor_path = os.environ.get("PREPROCESSOR_PATH", "models/preprocessor.pkl")

    logger.info(f"Loading model from {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        MODEL = pickle.load(f)

    logger.info(f"Loading preprocessor from {preprocessor_path}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    with open(preprocessor_path, "rb") as f:
        PREPROCESSOR = pickle.load(f)

    logger.info("Model and preprocessor loaded successfully")


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncGenerator[None, None]:
    """Loads model artifacts on startup."""
    load_artifacts()
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Real-time inference API for predicting customer churn "
        "using a trained XGBoost model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _predict_single(
    features: CustomerFeatures,
) -> PredictionResponse:
    """Runs prediction for a single customer."""
    feature_dict = features.model_dump()
    input_df = pd.DataFrame([feature_dict])

    transformed = PREPROCESSOR.transform(input_df)
    prediction = int(MODEL.predict(transformed)[0])
    probability = float(MODEL.predict_proba(transformed)[0][1])
    label = "Churn" if prediction == 1 else "No Churn"

    return PredictionResponse(
        prediction=prediction,
        churn_probability=round(probability, 4),
        label=label,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict churn for a single customer",
    tags=["Prediction"],
)
async def predict(features: CustomerFeatures, request: Request) -> PredictionResponse:
    """Accepts customer features and returns churn prediction."""
    start_time = time.perf_counter()
    try:
        if MODEL is None or PREPROCESSOR is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service unavailable.",
            )

        result = _predict_single(features)

        # Record Prometheus metrics
        PREDICTION_COUNTER.labels(predicted_class=str(result.prediction)).inc()
        elapsed = time.perf_counter() - start_time
        PREDICTION_LATENCY.observe(elapsed)

        logger.info(
            f"Prediction: {result.label} "
            f"(prob={result.churn_probability:.4f}, "
            f"latency={elapsed:.4f}s)"
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict churn for multiple customers",
    tags=["Prediction"],
)
async def predict_batch(
    batch: BatchPredictionRequest, request: Request
) -> BatchPredictionResponse:
    """Accepts a batch of customer features and returns predictions."""
    start_time = time.perf_counter()
    try:
        if MODEL is None or PREPROCESSOR is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service unavailable.",
            )

        results = []
        for customer in batch.customers:
            result = _predict_single(customer)
            PREDICTION_COUNTER.labels(predicted_class=str(result.prediction)).inc()
            results.append(result)

        elapsed = time.perf_counter() - start_time
        PREDICTION_LATENCY.observe(elapsed)

        logger.info(
            f"Batch prediction: {len(results)} customers " f"(latency={elapsed:.4f}s)"
        )
        return BatchPredictionResponse(predictions=results)

    except HTTPException:
        raise
    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.exception(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness check",
    tags=["Operations"],
)
async def health() -> HealthResponse:
    """Returns service health status."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        preprocessor_loaded=PREPROCESSOR is not None,
    )


@app.get(
    "/metrics",
    summary="Prometheus metrics endpoint",
    tags=["Operations"],
)
async def metrics() -> PlainTextResponse:
    """Exposes Prometheus-compatible metrics."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
