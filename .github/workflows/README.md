# GitHub Actions Workflows

## Purpose
This directory holds configuration files for GitHub Actions. It defines tasks for linting, testing, building Docker images, and deploying to Kubernetes.

## Key Files
- `ci.yaml`: Triggered on pull requests to run formatters (Black), linters (Flake8, MyPy), and unit tests (PyTest).
- `cd.yaml`: Triggered on main branch commits/tags to build Docker images and release Helm packages.

## Integration
- Automates the MLOps pipeline testing and deployment process.
