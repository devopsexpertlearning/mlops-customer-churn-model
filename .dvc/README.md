# DVC (Data Version Control) Directory

## 1. Purpose
This directory stores DVC configuration files, cache mappings, and local state. It establishes standard data versioning pipelines and storage configurations for the customer churn platform.

## 2. Architecture and Workflow
DVC allows us to track large data files (like CSVs, model binaries) without committing them to Git. 
DVC works on a dual-layer tracking model:
1. **Metadata Tracking**: Git commits small `.dvc` pointer files that track actual file hashes.
2. **Data Tracking**: DVC pushes/pulls the actual data files to/from S3-compatible remote storage based on those hash pointers.

```text
  [Git Repository]             [DVC Cache / Local Storage]
   (Tracks metadata)           (Tracks large artifacts & datasets)
     ├── data.csv.dvc  ───────>    s3://dvc-cache/<hash-key>
```

## 3. Files Contained Inside
* [config](file:///home/ubuntu/mlops/mlops-customer-churn-model/.dvc/config): Central DVC settings (e.g. S3 remote endpoints, default remote choice).
* [config.local](file:///home/ubuntu/mlops/mlops-customer-churn-model/.dvc/config.local): Machine-local overrides (e.g. Access keys, credentials). **This file is git-ignored for security.**
* `.gitignore`: Configured by DVC to ignore internal folders (`cache/`, `tmp/`).

## 4. Interaction with Other Modules
* **Ingestion/Preprocessing**: Read raw/processed datasets versioned under `data/` via `.dvc` files.
* **Model Training**: Saves versioned model binaries to `models/` tracked under DVC references.
* **Pipelines**: Integrated with `dvc.yaml` at the root folder to execute stages.

## 5. Usage Instructions
To fetch versioned data:
```bash
dvc pull
```

To commit new dataset changes:
```bash
dvc add data/raw/telco_customer_churn.csv
dvc push
```

## 6. Examples
### Merged Remote Configuration Example
DVC automatically merges `.dvc/config` (committed) and `.dvc/config.local` (local secrets) at runtime:
```bash
# Verify merged remote configs
dvc remote list
# Output: s3-remote       s3://dvc-cache  (default)
```
