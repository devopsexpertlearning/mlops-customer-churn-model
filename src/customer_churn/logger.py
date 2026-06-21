import json
import logging
import logging.config
import os
from typing import Any, Dict

DEFAULT_LOGGING_CONFIG_PATH = "configs/logging_config.json"


def setup_logging(config_path: str = DEFAULT_LOGGING_CONFIG_PATH) -> None:
    """Initializes logging configurations using the JSON file.

    Creates target directories for log files automatically if configured.
    """
    if not os.path.exists(config_path):
        # Fallback to simple console logging if config file not found
        logging.basicConfig(level=logging.INFO)
        logging.warning(
            f"Logging config not found at {config_path}. "
            "Initialized fallback console logger."
        )
        return

    try:
        with open(config_path, "r") as f:
            config: Dict[str, Any] = json.load(f)

        # Automatically create directories for rotating files
        handlers = config.get("handlers", {})
        for handler_name, handler_conf in handlers.items():
            if "filename" in handler_conf:
                log_file = handler_conf["filename"]
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

        logging.config.dictConfig(config)
    except Exception as e:
        # Fallback to simple logging if parsing config fails
        logging.basicConfig(level=logging.INFO)
        logging.error(
            f"Failed to load logging configuration from {config_path}: {e}. "
            "Fallback enabled."
        )


def get_logger(name: str) -> logging.Logger:
    """Retrieves standard python logger with name parameter."""
    return logging.getLogger(name)
