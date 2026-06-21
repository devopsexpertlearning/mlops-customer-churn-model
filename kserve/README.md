# KServe Serving Directory

## Purpose
This directory contains configurations for deploying models as serverless InferenceServices using KServe on Kubernetes.

## Key Files
- `inferenceservice.yaml`: Resource definition configuration for the KServe Predictor.
- `transformer/`: Preprocessing/postprocessing pipeline wrapper to run alongside the KServe XGBoost predictor.
