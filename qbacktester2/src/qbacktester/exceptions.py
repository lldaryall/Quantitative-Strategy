"""Custom exceptions for qbacktester."""

from typing import Any, Dict, Optional


class QBacktesterError(Exception):
    """Base exception for all qbacktester errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Optional additional details about the error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class DataError(QBacktesterError):
    """Exception raised for data-related errors."""

    pass


class ValidationError(QBacktesterError):
    """Exception raised for validation errors."""

    pass


class BacktestError(QBacktesterError):
    """Exception raised for backtesting errors."""

    pass


class StrategyError(QBacktesterError):
    """Exception raised for strategy-related errors."""

    pass


class OptimizationError(QBacktesterError):
    """Exception raised for optimization errors."""

    pass


class PlottingError(QBacktesterError):
    """Exception raised for plotting errors."""

    pass


class ConfigurationError(QBacktesterError):
    """Exception raised for configuration errors."""

    pass


class NetworkError(QBacktesterError):
    """Exception raised for network-related errors."""

    pass


class CacheError(QBacktesterError):
    """Exception raised for cache-related errors."""

    pass

