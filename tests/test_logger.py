import logging
from customer_churn.logger import get_logger, setup_logging


def test_setup_logging_success() -> None:
    """Tests setting up logging with valid config JSON."""
    setup_logging(config_path="configs/logging_config.json")
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    logger.info("Test logging setup success")


def test_setup_logging_fallback() -> None:
    """Tests setup_logging fallback behaviour when path does not exist."""
    setup_logging(config_path="configs/non_existent_logger_config.json")
    logger = get_logger("test_fallback_logger")
    assert isinstance(logger, logging.Logger)
