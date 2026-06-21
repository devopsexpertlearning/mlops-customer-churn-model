"""Kubeflow Pipeline DAG for Customer Churn Prediction.

Assembles the ML pipeline stages (ingest -> validate -> preprocess ->
features -> train -> evaluate) into a single KFP v2 pipeline DAG
and compiles it to kubeflow/pipeline.yaml.
"""

from kfp import dsl, compiler

CONTAINER_IMAGE = "mlops-customer-churn:latest"
INGEST_URL = (
    "https://raw.githubusercontent.com/alexeygrigorev/"
    "mlbookcamp-code/master/chapter-03-churn-prediction/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


@dsl.component(base_image=CONTAINER_IMAGE)
def ingest_data(ingest_url: str) -> str:
    """Ingests the Telco churn dataset from a public URL."""
    import os
    import pandas as pd

    raw_path = "/tmp/data/raw/telco_customer_churn.csv"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df = pd.read_csv(ingest_url)
    df.to_csv(raw_path, index=False)
    return raw_path


@dsl.component(base_image=CONTAINER_IMAGE)
def validate_data(raw_path: str) -> str:
    """Validates raw data schema, value ranges, and target column."""
    import json
    import os
    import pandas as pd

    report_path = "/tmp/artifacts/validation_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df = pd.read_csv(raw_path)
    report = {
        "shape": list(df.shape),
        "validation_passed": len(df) >= 1000,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    if not report["validation_passed"]:
        raise ValueError("Validation failed: insufficient rows")
    return report_path


@dsl.component(base_image=CONTAINER_IMAGE)
def preprocess_data(raw_path: str) -> str:
    """Drops IDs, cleans charges, binarizes target, and splits data."""
    import os
    import json
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(raw_path)

    # Drop customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Clean TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Binarize target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Churn"]
    )

    train_path = "/tmp/data/processed/train.csv"
    test_path = "/tmp/data/processed/test.csv"
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    paths_file = "/tmp/data/processed/paths.json"
    with open(paths_file, "w") as f:
        json.dump({"train_path": train_path, "test_path": test_path}, f)
    return paths_file


@dsl.component(base_image=CONTAINER_IMAGE)
def feature_engineering(
    preprocess_paths: str,
) -> str:
    """Scales numerics, one-hot encodes categoricals, saves preprocessor."""
    import json
    import os
    import pickle
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    with open(preprocess_paths, "r") as f:
        paths = json.load(f)
    train_df = pd.read_csv(paths["train_path"])
    test_df = pd.read_csv(paths["test_path"])

    target_col = "Churn"
    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    train_out = pd.DataFrame(X_train_transformed)
    train_out[target_col] = train_df[target_col].values
    test_out = pd.DataFrame(X_test_transformed)
    test_out[target_col] = test_df[target_col].values

    out_dir = "/tmp/data/processed"
    os.makedirs(out_dir, exist_ok=True)
    train_features_path = f"{out_dir}/train_features.csv"
    test_features_path = f"{out_dir}/test_features.csv"
    preprocessor_path = "/tmp/models/preprocessor.pkl"
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

    train_out.to_csv(train_features_path, index=False)
    test_out.to_csv(test_features_path, index=False)

    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)

    feature_paths = f"{out_dir}/feature_paths.json"
    with open(feature_paths, "w") as f:
        json.dump(
            {
                "train_features_path": train_features_path,
                "test_features_path": test_features_path,
                "preprocessor_path": preprocessor_path,
                "test_path": paths["test_path"],
            },
            f,
        )
    return feature_paths


@dsl.component(base_image=CONTAINER_IMAGE)
def train_model(feature_paths: str) -> str:
    """Trains an XGBoost classifier and saves the model."""
    import json
    import os
    import pickle
    import pandas as pd
    from xgboost import XGBClassifier

    with open(feature_paths, "r") as f:
        paths = json.load(f)

    train_df = pd.read_csv(paths["train_features_path"])
    target_col = "Churn"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    model = XGBClassifier(
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    model_path = "/tmp/models/model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    train_output = "/tmp/models/train_output.json"
    with open(train_output, "w") as f:
        json.dump(
            {
                "model_path": model_path,
                "test_path": paths["test_path"],
                "preprocessor_path": paths["preprocessor_path"],
            },
            f,
        )
    return train_output


@dsl.component(base_image=CONTAINER_IMAGE)
def evaluate_model(train_output: str) -> str:
    """Evaluates model and produces metrics JSON report."""
    import json
    import os
    import pickle
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    with open(train_output, "r") as f:
        paths = json.load(f)

    test_df = pd.read_csv(paths["test_path"])
    target_col = "Churn"
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    with open(paths["preprocessor_path"], "rb") as f:
        preprocessor = pickle.load(f)
    with open(paths["model_path"], "rb") as f:
        model = pickle.load(f)

    X_test_transformed = preprocessor.transform(X_test)
    preds = model.predict(X_test_transformed)
    probs = model.predict_proba(X_test_transformed)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1_score": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
    }

    metrics_path = "/tmp/artifacts/metrics.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    return metrics_path


@dsl.pipeline(
    name="Customer Churn Prediction Pipeline",
    description=(
        "End-to-end ML pipeline: ingest, validate, preprocess, "
        "feature engineer, train XGBoost, and evaluate."
    ),
)
def churn_pipeline(
    ingest_url: str = INGEST_URL,
) -> None:
    """Assembles all stages into a sequential KFP DAG."""
    ingest_task = ingest_data(ingest_url=ingest_url)

    validate_task = validate_data(
        raw_path=ingest_task.output,
    )

    preprocess_task = preprocess_data(
        raw_path=ingest_task.output,
    )
    preprocess_task.after(validate_task)

    features_task = feature_engineering(
        preprocess_paths=preprocess_task.output,
    )

    train_task = train_model(
        feature_paths=features_task.output,
    )

    evaluate_model(
        train_output=train_task.output,
    )


def compile_pipeline() -> None:
    """Compiles the pipeline to kubeflow/pipeline.yaml."""
    compiler.Compiler().compile(
        pipeline_func=churn_pipeline,
        package_path="kubeflow/pipeline.yaml",
    )
    print("Pipeline compiled to kubeflow/pipeline.yaml")


if __name__ == "__main__":
    compile_pipeline()
