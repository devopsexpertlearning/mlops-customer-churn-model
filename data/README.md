# Data Directory

## Purpose
This directory contains raw, processed, and external datasets used for model training, evaluation, and drift detection. Data files in this directory are tracked by DVC (and ignored by Git).

## Directory Structure
- `raw/`: Untouched datasets as ingested from external sources.
- `processed/`: Cleansed, preprocessed, and feature-engineered datasets.
- `external/`: External configurations or reference datasets (e.g., benchmark statistics).

## Interaction
- Fed into data pipelines located in `src/` and managed under `dvc.yaml`.
