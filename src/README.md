# Source Directory

## 1. Purpose
Contains the core Python package modules implementing the end-to-end MLOps workflow from ingestion to deployment.

## 2. Architecture and Workflow
The codebase follows a modular design pattern, dividing steps into distinct, self-contained scripts. 
At runtime, configuration settings and logging levels are loaded globally first:
```text
  [Configs] 
      │
      ▼
  [config.py / logger.py] ──> Initializes configs and logging
      │
      ▼
  [Data Pipeline Modules] (ingest -> validate -> preprocess -> train -> evaluate)
```

## 3. Files Contained Inside
* [customer_churn/config.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/config.py): Dynamic loader of YAML configuration files supporting environment overrides and type casting.
* [customer_churn/logger.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/logger.py): Configures structured JSON logs for observability in production.
* [customer_churn/ingest.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/ingest.py): Programmatically downloads the Telco Customer Churn dataset from a configured raw repository URL.
* [customer_churn/validate.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/validate.py): Performs schema compliance, range checking, record thresholds verification, and target distribution checking, writing structured results to artifacts.
* [customer_churn/preprocess.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/preprocess.py): Drops non-relevant columns, converts data types (TotalCharges cleaning), maps classification labels, and creates stratified train/test partitions using pipeline parameters.
* [customer_churn/features.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/features.py): Performs feature scaling (numerical columns) and one-hot encoding (categorical columns) using a Scikit-Learn `ColumnTransformer`, outputting feature files and the serialized pipeline.
* [customer_churn/train.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/train.py): Trains an XGBoost classifier using parameters from `params.yaml` and serialized training features, logs parameters and model registry artifacts to MLflow, and saves `models/model.pkl` and `artifacts/mlflow_run_id.txt`.
* [customer_churn/evaluate.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/evaluate.py): Evaluates the trained model against the test dataset, resumes the active MLflow run to log computed classification performance metrics and report artifacts, and exports `artifacts/metrics.json`.

*(Placeholder modules for upcoming phases: serve.py)*

## 4. Interaction with Other Modules
* **Configurations**: Interacts directly with files inside the `configs/` directory.
* **Testing Suite**: Extensively tested by scripts under `tests/` (e.g. `test_config.py`, `test_ingest.py`, `test_validate.py`, `test_preprocess.py`, `test_features.py`, `test_train.py`, `test_evaluate.py`).
* **Pipelines**: Stages are automated using `dvc.yaml` at the root folder.

## 5. Usage Instructions
To initialize configurations and logging in any module, import the setup helpers:
```python
from customer_churn.config import AppConfig
from customer_churn.logger import setup_logging, get_logger

# 1. Setup logging
setup_logging()
logger = get_logger(__name__)

# 2. Setup config
config = AppConfig()
project_name = config.get("project.name")
logger.info(f"Initialized project: {project_name}")
```

## 6. Examples
### Running the Config Loader directly
You can load and query configurations programmatically inside the virtual environment:
```bash
python -c "from customer_churn.config import AppConfig; print(AppConfig().get('project.name'))"
# Output: mlops-customer-churn
```
