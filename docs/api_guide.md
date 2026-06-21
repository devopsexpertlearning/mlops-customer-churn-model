# FastAPI Serve API Guide

This document describes the API endpoints, schemas, request bodies, and responses for the Customer Churn FastAPI serving server.

## 1. Local Server Initialization
The serving container can be started locally via the Docker Compose command:
```bash
docker compose up -d api
```
The server will bind to port `8000`. The interactive Swagger UI documentation is available at:
`http://localhost:8000/docs`

## 2. API Endpoints

### 2.1 GET /health
Queries the service health check and verifies whether the model and preprocessor files loaded successfully.

* **Response Schema (JSON)**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "preprocessor_loaded": true
  }
  ```

* **Example CURL**:
  ```bash
  curl -s http://localhost:8000/health
  ```

---

### 2.2 POST /predict
Predicts customer churn in real-time for a single customer profile.

* **Request Body (JSON)**:
  ```json
  {
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
    "TotalCharges": 840.50
  }
  ```

* **Response Body (JSON)**:
  ```json
  {
    "prediction": 0,
    "churn_probability": 0.4845,
    "label": "No Churn"
  }
  ```

* **Example CURL**:
  ```bash
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
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
      "TotalCharges": 840.50
    }'
  ```

---

### 2.3 POST /predict/batch
Predicts customer churn in batch for multiple customer profiles simultaneously.

* **Request Body (JSON)**:
  ```json
  {
    "customers": [
      {
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
        "TotalCharges": 840.50
      },
      {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 25.30,
        "TotalCharges": 25.30
      }
    ]
  }
  ```

* **Response Body (JSON)**:
  ```json
  {
    "predictions": [
      {
        "prediction": 0,
        "churn_probability": 0.4845,
        "label": "No Churn"
      },
      {
        "prediction": 1,
        "churn_probability": 0.8226,
        "label": "Churn"
      }
    ]
  }
  ```

* **Example CURL**:
  ```bash
  curl -s -X POST http://localhost:8000/predict/batch \
    -H "Content-Type: application/json" \
    -d '{
      "customers": [
        {"gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 840.50},
        {"gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 25.30, "TotalCharges": 25.30}
      ]
    }'
  ```

---

### 2.4 GET /metrics
Exposes Prometheus-compatible statistics for container and serving pipeline health tracking.

* **Exposed Metrics**:
  - `churn_predictions_total`: Counter tracking total requests broken down by prediction class (`predicted_class="0"` or `predicted_class="1"`).
  - `churn_prediction_latency_seconds`: Histogram logging inference speeds in seconds.
  - `churn_prediction_errors_total`: Counter tracking server errors.

* **Example CURL**:
  ```bash
  curl -s http://localhost:8000/metrics
  ```
