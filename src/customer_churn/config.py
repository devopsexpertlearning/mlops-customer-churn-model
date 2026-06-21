import os
import yaml
from typing import Any, Dict


class ConfigError(Exception):
    """Custom exception raised for configuration-related errors."""

    pass


class AppConfig:
    """Loads, validates, and queries configuration from configs/config.yaml."""

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config_path = config_path
        self._config_data: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        """Loads configuration from YAML file with error handling."""
        if not os.path.exists(self.config_path):
            raise ConfigError(f"Configuration file not found at {self.config_path}")

        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)
                if data is None:
                    raise ConfigError(
                        f"Configuration file at {self.config_path} is empty"
                    )
                if not isinstance(data, dict):
                    raise ConfigError(
                        f"Configuration file at {self.config_path} "
                        "is invalid (not a dictionary)"
                    )
                self._config_data = data
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML config: {e}") from e
        except Exception as e:
            raise ConfigError(f"Unexpected error loading config: {e}") from e

    def get(self, key_path: str, default: Any = None) -> Any:
        """Gets config value using dot notation, supporting environment overrides.

        Example: data.target_col can be overridden by env variable
        DATA_TARGET_COL.
        """
        # 1. Resolve from YAML config data
        keys = key_path.split(".")
        current: Any = self._config_data
        key_found = True

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                key_found = False
                break

        # 2. Check environment variable override
        env_key = key_path.replace(".", "_").upper()
        if env_key in os.environ:
            env_val = os.environ[env_key]
            if key_found and current is not None:
                # Cast the environment variable to match the type in configs
                target_type = type(current)
                if target_type is bool:
                    return env_val.lower() in ("true", "1", "yes", "on")
                try:
                    return target_type(env_val)
                except (ValueError, TypeError):
                    return env_val
            return env_val

        if not key_found:
            if default is not None:
                return default
            raise ConfigError(f"Configuration key '{key_path}' not found")

        return current
