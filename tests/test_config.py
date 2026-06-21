import pytest
from customer_churn.config import AppConfig, ConfigError


def test_load_valid_config() -> None:
    """Tests loading the default configuration file."""
    config = AppConfig(config_path="configs/config.yaml")
    assert config.get("project.name") == "mlops-customer-churn"
    assert config.get("data.target_col") == "Churn"


def test_missing_config_raises_error() -> None:
    """Tests that loading a non-existent config raises ConfigError."""
    with pytest.raises(ConfigError):
        AppConfig(config_path="configs/non_existent_config.yaml")


def test_get_invalid_key_raises_error() -> None:
    """Tests that querying a non-existent key path raises ConfigError."""
    config = AppConfig(config_path="configs/config.yaml")
    with pytest.raises(ConfigError):
        config.get("non.existent.key")


def test_get_invalid_key_with_default() -> None:
    """Tests that querying a non-existent key path with default returns default."""
    config = AppConfig(config_path="configs/config.yaml")
    val = config.get("non.existent.key", default="fallback")
    assert val == "fallback"


def test_environment_override_casting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that environment variables override config and type casting."""
    config = AppConfig(config_path="configs/config.yaml")

    # Override string value
    monkeypatch.setenv("PROJECT_NAME", "override-project-name")
    assert config.get("project.name") == "override-project-name"

    # Override numeric value
    monkeypatch.setenv("MONITORING_PROMETHEUS_PORT", "9090")
    assert config.get("monitoring.prometheus_port") == 9090
    assert isinstance(config.get("monitoring.prometheus_port"), int)
