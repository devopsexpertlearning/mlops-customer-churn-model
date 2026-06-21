# Tests Directory

## 1. Purpose
This directory contains the unit, integration, and end-to-end testing suites for all the MLOps scripts, configuration parameters, validation logic, and server APIs.

## 2. Architecture and Workflow
The testing architecture follows a strict 1-to-1 correspondence with the source package modules:
```text
  [src/customer_churn/config.py]      ─────────>  [tests/test_config.py]
  [src/customer_churn/logger.py]      ─────────>  [tests/test_logger.py]
  [src/customer_churn/ingest.py]      ─────────>  [tests/test_ingest.py]
  [src/customer_churn/validate.py]    ─────────>  [tests/test_validate.py]
  [src/customer_churn/preprocess.py]  ─────────>  [tests/test_preprocess.py]
  [src/customer_churn/features.py]    ─────────>  [tests/test_features.py]
  [src/customer_churn/train.py]       ─────────>  [tests/test_train.py]
  [src/customer_churn/evaluate.py]    ─────────>  [tests/test_evaluate.py]
```
Tests are written using `pytest` and executed inside the project's Python 3.12 virtual environment.

## 3. Files Contained Inside
* [test_config.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/tests/test_config.py): Validates YAML config loading, path resolving, missing config error handling, and environment overrides type casting.
* [test_logger.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/tests/test_logger.py): Validates structured JSON logging initialization and console fallbacks.
* [test_ingest.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/tests/test_ingest.py): Validates programmatic raw dataset downloading, folder structures setup, and data storage.
* [test_validate.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/tests/test_validate.py): Validates schema integrity checks, volume constraint checks, range validations, and outputs artifacts correctly.
* [test_preprocess.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/tests/test_preprocess.py): Validates TotalCharges numeric conversions, customerID stripping, stratified split partitions, and YAML parameters compliance.
* [test_features.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/tests/test_features.py): Validates standard feature scaling, categorical one-hot encoding columns mappings, and preprocessor pipeline object serialization.
* [test_train.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/tests/test_train.py): Validates XGBClassifier fitting, configuration loading, parameter usage, local model serialization, and MLflow logging integrations (using mock patches).
* [test_evaluate.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/tests/test_evaluate.py): Validates data loading, preprocessor mapping, model prediction generation, JSON metrics formatting, and MLflow run resumption/logging (using mock patches).
* [test_placeholder.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/tests/test_placeholder.py): Placeholder test verifying testing pipeline integrity.

## 4. Interaction with Other Modules
* **Source Code**: Imports package modules under `src/customer_churn/`.
* **CI Pipelines**: Triggered automatically on PRs or pushes via `.github/workflows/ci.yaml` to block buggy or malformed contributions.

## 5. Usage Instructions
To execute the testing suite, run `pytest` from the virtual environment:
```bash
pytest tests/
```

## 6. Examples
### Run specific test files
You can run a single test module to speed up local validation:
```bash
pytest tests/test_config.py
```
