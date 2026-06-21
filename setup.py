from setuptools import find_packages, setup

setup(
    name="mlops-customer-churn",
    version="0.1.0",
    description="Enterprise-grade Customer Churn Prediction MLOps Platform",
    author="Senior MLOps Architect",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.12",
    install_requires=[
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.28.0",
        "gunicorn>=21.2.0",
        "PyYAML>=6.0.0",

        "prometheus-client>=0.20.0",
        "python-json-logger>=2.0.7",
    ],
    extras_require={
        "train": [
            "mlflow>=2.11.0",
            "dvc>=3.48.0",
            "evidently>=0.4.15",
            "psycopg2-binary>=2.9.9",
            "sqlalchemy>=2.0.27",
            "boto3>=1.34.0",
            "kfp>=2.7.0",
        ],
        "dev": [
            "pytest>=8.0.0",
            "black>=24.2.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
    },
)

