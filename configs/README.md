# Configurations Directory

## 1. Purpose
This directory houses configuration files (YAML, JSON) for pipeline parameters, model training hyper-parameters, structured logging settings, and infrastructure options.

## 2. Architecture and Workflow
We use a centralized configuration architecture. Instead of hardcoding paths, targets, or ports in code, all parameters are loaded from YAML/JSON files:
1. **Application Parameters**: Managed in `config.yaml`.
2. **Logging Rules**: Configured in `logging_config.json`.
3. **Pipeline Hyper-parameters**: Tracked in `params.yaml` at the root, which DVC uses to track pipeline dependencies.

```text
 [config.yaml] ────────┐
                       ├──> Loaded by [src/customer_churn/config.py]
 [logging_config.json] ┘
```

## 3. Files Contained Inside
* [config.yaml](file:///home/ubuntu/mlops/mlops-customer-churn-model/configs/config.yaml): Defines raw/processed data paths, target column, model file path, and port numbers.
* [logging_config.json](file:///home/ubuntu/mlops/mlops-customer-churn-model/configs/logging_config.json): Defines logging handlers (console, rolling files) and log formatters (structured JSON format using `pythonjsonlogger`).

## 4. Interaction with Other Modules
* **Config Loader** ([config.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/config.py)): Loads and resolves these config files.
* **Logging Setup** ([logger.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/logger.py)): Configures python logging levels and formats dynamically based on `logging_config.json`.

## 5. Usage Instructions
To adjust parameters, edit the YAML/JSON files directly.
For example, to change the Prometheus port:
```yaml
monitoring:
  prometheus_port: 9090
```

## 6. Examples
### config.yaml Snippet
```yaml
project:
  name: "mlops-customer-churn"
data:
  raw_path: "data/raw/telco_customer_churn.csv"
```
