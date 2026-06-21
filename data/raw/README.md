# Raw Data Directory

## 1. Purpose
This directory stores the untouched, raw dataset retrieved from public registries, acting as the single source of truth for the rest of the MLOps pipeline.

## 2. Architecture and Workflow
The ingestion script `src/customer_churn/ingest.py` downloads the raw dataset and stores it here as `telco_customer_churn.csv`.
Once saved, DVC tracks this file via hash mapping to prevent committing large CSV files to Git.

```text
 [Ingection Module] ──> [data/raw/telco_customer_churn.csv] ──> Tracked by DVC
```

## 3. Files Contained Inside
* [telco_customer_churn.csv](file:///home/ubuntu/mlops/mlops-customer-churn-model/data/raw/telco_customer_churn.csv): The raw Telco Customer Churn dataset. **This file is git-ignored.**

### Dataset Schema
The dataset contains 7,043 rows and 21 columns:
* `customerID`: Unique customer identifier.
* `gender`: Customer gender (Male, Female).
* `SeniorCitizen`: Indicates if customer is a senior citizen (1, 0).
* `Partner`: Indicates if customer has a partner (Yes, No).
* `Dependents`: Indicates if customer has dependents (Yes, No).
* `tenure`: Number of months the customer has stayed with the company.
* `PhoneService`: Indicates if customer has phone service (Yes, No).
* `MultipleLines`: Indicates if customer has multiple phone lines (Yes, No, No phone service).
* `InternetService`: Customer's internet service provider (DSL, Fiber optic, No).
* `OnlineSecurity`: Indicates if customer has online security (Yes, No, No internet service).
* `OnlineBackup`: Indicates if customer has online backup (Yes, No, No internet service).
* `DeviceProtection`: Indicates if customer has device protection (Yes, No, No internet service).
* `TechSupport`: Indicates if customer has tech support (Yes, No, No internet service).
* `StreamingTV`: Indicates if customer has streaming TV (Yes, No, No internet service).
* `StreamingMovies`: Indicates if customer has streaming movies (Yes, No, No internet service).
* `Contract`: The contract term of the customer (Month-to-month, One year, Two year).
* `PaperlessBilling`: Indicates if customer has paperless billing (Yes, No).
* `PaymentMethod`: The customer's payment method (Electronic check, Mailed check, Bank transfer, Credit card).
* `MonthlyCharges`: The amount charged to the customer monthly.
* `TotalCharges`: The total amount charged to the customer.
* `Churn`: Target column indicating whether the customer churned or not (Yes, No).

## 4. Interaction with Other Modules
* **Data Validation** ([validate.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/validate.py)): Reads the raw dataset from this folder to perform schema and range checks.
* **Data Preprocessing** ([preprocess.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/preprocess.py)): Loads data from this folder to clean, split, and encode features.

## 5. Usage Instructions
To ingest and update the dataset in this directory, run:
```bash
dvc repro ingest
```

## 6. Examples
### Python code to view raw data schema
```python
import pandas as pd
df = pd.read_csv("data/raw/telco_customer_churn.csv")
print(df.info())
```
