import os
import pathlib
from kfp import compiler
from kubeflow.pipeline import churn_pipeline


def test_pipeline_compiles(tmp_path: pathlib.Path) -> None:
    """Verifies that the churn pipeline compiles to a valid YAML file."""
    output_path = os.path.join(tmp_path, "pipeline.yaml")
    compiler.Compiler().compile(
        pipeline_func=churn_pipeline,
        package_path=output_path,
    )
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

    # Verify the compiled YAML contains expected pipeline name
    with open(output_path, "r") as f:
        content = f.read()
    assert "customer-churn-prediction-pipeline" in content


def test_pipeline_has_expected_tasks(tmp_path: pathlib.Path) -> None:
    """Verifies all expected task names appear in the compiled DAG."""
    output_path = os.path.join(tmp_path, "pipeline.yaml")
    compiler.Compiler().compile(
        pipeline_func=churn_pipeline,
        package_path=output_path,
    )
    with open(output_path, "r") as f:
        content = f.read()

    expected_tasks = [
        "ingest-data",
        "validate-data",
        "preprocess-data",
        "feature-engineering",
        "train-model",
        "evaluate-model",
    ]
    for task_name in expected_tasks:
        assert (
            task_name in content
        ), f"Expected task '{task_name}' not found in compiled pipeline"
