"""Configuration management for qbacktester."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from rich.logging import RichHandler


class Config:
    """Configuration class for qbacktester."""

    def __init__(self) -> None:
        """Initialize configuration with default values."""
        self._config: Dict[str, Any] = {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None,
            },
            "data": {
                "cache_dir": "data",
                "max_retries": 3,
                "timeout": 30,
            },
            "backtesting": {
                "default_initial_cash": 100_000,
                "default_fee_bps": 1.0,
                "default_slippage_bps": 0.5,
                "trading_days_per_year": 252,
            },
            "plotting": {
                "style": "seaborn-v0_8",
                "figure_size": (12, 8),
                "dpi": 300,
                "save_format": "png",
            },
            "optimization": {
                "max_workers": None,  # Auto-detect
                "chunk_size": 100,
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split(".")
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key."""
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        env_mappings = {
            "QBACKTESTER_LOG_LEVEL": "logging.level",
            "QBACKTESTER_CACHE_DIR": "data.cache_dir",
            "QBACKTESTER_MAX_RETRIES": "data.max_retries",
            "QBACKTESTER_TIMEOUT": "data.timeout",
            "QBACKTESTER_DEFAULT_CASH": "backtesting.default_initial_cash",
            "QBACKTESTER_DEFAULT_FEE": "backtesting.default_fee_bps",
            "QBACKTESTER_DEFAULT_SLIPPAGE": "backtesting.default_slippage_bps",
            "QBACKTESTER_MAX_WORKERS": "optimization.max_workers",
        }

        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Try to convert to appropriate type
                if config_key.endswith(("_bps", "_cash", "_retries", "_timeout", "_workers")):
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                elif config_key.endswith("level"):
                    # Keep as string for logging level
                    pass
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                
                self.set(config_key, value)

    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = self.get("logging.level", "INFO")
        log_format = self.get("logging.format")
        log_file = self.get("logging.file")

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[]
        )

        # Get root logger
        root_logger = logging.getLogger()
        
        # Clear existing handlers
        root_logger.handlers.clear()

        # Add console handler with Rich formatting
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=True,
            markup=True
        )
        console_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_formatter = logging.Formatter(log_format)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        # Set specific logger levels
        logging.getLogger("yfinance").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    def get_cache_dir(self) -> Path:
        """Get cache directory path."""
        cache_dir = Path(self.get("data.cache_dir", "data"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def get_reports_dir(self) -> Path:
        """Get reports directory path."""
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        return reports_dir


# Global configuration instance
config = Config()

# Update from environment variables on import
config.update_from_env()

# Setup logging
config.setup_logging()

# Create logger for this module
logger = logging.getLogger(__name__)

