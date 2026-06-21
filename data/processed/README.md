# Processed Data Directory

## 1. Purpose
This directory stores datasets that have been cleaned, partitioned, and prepared for feature engineering and model training stages.

## 2. Architecture and Workflow
The preprocessing module `src/customer_churn/preprocess.py` cleans raw fields (e.g. converting `TotalCharges` to float, replacing spaces with 0.0), drops the irrelevant identifier column `customerID`, maps the target `Churn` to numeric labels [0, 1], and splits the data into train/test subsets using parameters from `params.yaml`.

```text
 [data/raw/telco_customer_churn.csv] ──> [preprocess.py] ──┐
                                                            ├──> [data/processed/train.csv]
                                                            └──> [data/processed/test.csv]
```

## 3. Files Contained Inside
* [train.csv](file:///home/ubuntu/mlops/mlops-customer-churn-model/data/processed/train.csv): Preprocessed training dataset (80% split). **This file is git-ignored.**
* [test.csv](file:///home/ubuntu/mlops/mlops-customer-churn-model/data/processed/test.csv): Preprocessed testing dataset (20% split). **This file is git-ignored.**

## 4. Interaction with Other Modules
* **Preprocessing Module** ([preprocess.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/preprocess.py)): Generates the files in this directory.
* **Feature Engineering Module** ([features.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/features.py)): Will load these files in Phase 7 to perform encoding and scaling.
* **Model Training Module** ([train.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/train.py)): Loads engineered outputs derived from this directory.

## 5. Usage Instructions
To re-run the preprocessing pipeline and regenerate train/test splits:
```bash
dvc repro preprocess
```

## 6. Examples
### Python code to verify splits shape
```python
import pandas as pd
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
print(f"Train Shape: {train.shape}, Test Shape: {test.shape}")
```
