# Models Directory

## 1. Purpose
This directory stores serialized model files, neural network weights, and preprocessing pipeline binary artifacts used during model training and serving.

## 2. Architecture and Workflow
Model tracking follows a decoupled pattern. The preprocessing pipeline is saved separately as a `ColumnTransformer` binary here. 
At serving time, incoming requests are passed through this preprocessor first, and the resulting feature vectors are fed to the model:

```text
 [Raw JSON Payload] ──> [preprocessor.pkl] ──> [model.pkl] ──> [Predictive Output]
```

## 3. Files Contained Inside
* [preprocessor.pkl](file:///home/ubuntu/mlops/mlops-customer-churn-model/models/preprocessor.pkl): Serialized Scikit-Learn `ColumnTransformer` object fitted on the training split. **This file is git-ignored.**
* [model.pkl](file:///home/ubuntu/mlops/mlops-customer-churn-model/models/model.pkl): Serialized XGBoost `XGBClassifier` model binary fitted on the engineered training features. **This file is git-ignored.**

## 4. Interaction with Other Modules
* **Feature Engineering** ([features.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/features.py)): Generates and serializes the preprocessor pipeline.
* **Model Training** ([train.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/train.py)): Fits the XGBoost classifier and serializes the resulting model binary to `models/model.pkl`.
* **FastAPI Serving** ([serve.py](file:///home/ubuntu/mlops/mlops-customer-churn-model/src/customer_churn/serve.py)): Loads both `preprocessor.pkl` and `model.pkl` to execute real-time model inference.

## 5. Usage Instructions
To serialize models and preprocessors using the DVC pipeline:
```bash
dvc repro train
```

## 6. Examples
### Python code to load the preprocessor and model
```python
import pickle

# Load the column preprocessor
with open("models/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)
print(preprocessor)

# Load the trained XGBoost model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
print(model)
```
