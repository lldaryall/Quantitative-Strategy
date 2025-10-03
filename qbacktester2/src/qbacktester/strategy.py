"""Strategy classes and signal generation for qbacktester."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from .indicators import crossover, sma


@dataclass
class StrategyParams:
    """
    Parameters for trading strategy configuration.

    Attributes:
        symbol: Trading symbol (e.g., 'AAPL', 'MSFT')
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        fast_window: Fast moving average window
        slow_window: Slow moving average window
        initial_cash: Starting capital (default: 100,000)
        fee_bps: Round-trip cost in basis points (default: 1.0)
        slippage_bps: Slippage cost in basis points (default: 0.5)
    """

    symbol: str
    start: str
    end: str
    fast_window: int
    slow_window: int
    initial_cash: float = 100_000
    fee_bps: float = 1.0
    slippage_bps: float = 0.5

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.fast_window <= 0:
            raise ValueError(f"fast_window must be positive, got {self.fast_window}")
        if self.slow_window <= 0:
            raise ValueError(f"slow_window must be positive, got {self.slow_window}")
        if self.fast_window >= self.slow_window:
            raise ValueError(
                f"fast_window ({self.fast_window}) must be less than slow_window ({self.slow_window})"
            )
        if self.initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {self.initial_cash}")
        if self.fee_bps < 0:
            raise ValueError(f"fee_bps must be non-negative, got {self.fee_bps}")
        if self.slippage_bps < 0:
            raise ValueError(
                f"slippage_bps must be non-negative, got {self.slippage_bps}"
            )


def generate_signals(
    price_df: pd.DataFrame, fast_window: int, slow_window: int
) -> pd.DataFrame:
    """
    Generate trading signals based on SMA crossover strategy.

    This function implements a vectorized SMA crossover strategy that:
    - Calculates fast and slow SMAs
    - Detects crossovers using the indicators.crossover function
    - Generates signals with 1-day delay to avoid look-ahead bias
    - Only allows long positions (1) or flat (0), no shorting

    Args:
        price_df: DataFrame with OHLCV data, must have 'Close' column
        fast_window: Fast SMA window period
        slow_window: Slow SMA window period

    Returns:
        DataFrame with additional columns:
        - 'sma_fast': Fast SMA values
        - 'sma_slow': Slow SMA values
        - 'crossover': Crossover detection (-1, 0, 1)
        - 'signal': Trading signal (0 or 1) with 1-day delay

    Raises:
        ValueError: If windows are invalid or 'Close' column missing
    """
    if "Close" not in price_df.columns:
        raise ValueError("price_df must contain 'Close' column")

    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("Both fast_window and slow_window must be positive")

    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window")

    # Create a copy to avoid modifying original DataFrame
    result_df = price_df.copy()

    # Calculate SMAs using the indicators module
    result_df["sma_fast"] = sma(price_df, window=fast_window, col="Close")
    result_df["sma_slow"] = sma(price_df, window=slow_window, col="Close")

    # Detect crossovers using the indicators module
    result_df["crossover"] = crossover(result_df["sma_fast"], result_df["sma_slow"])

    # Generate signals with 1-day delay to avoid look-ahead bias
    # Only enter long positions (1) on bullish crossovers, otherwise flat (0)
    # Shift by 1 day to ensure no look-ahead bias
    bullish_crossovers = (
        (result_df["crossover"] == 1).shift(1).fillna(False).astype(bool)
    )
    result_df["signal"] = bullish_crossovers.astype(int)

    return result_df


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    All trading strategies should inherit from this class and implement
    the generate_signal method.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the strategy.

        Args:
            name: Name of the strategy
        """
        self.name = name

    @abstractmethod
    def generate_signal(
        self, data: pd.Series, portfolio_value: float, positions: Dict[str, float]
    ) -> Optional[Dict[str, Union[str, float]]]:
        """
        Generate trading signal based on current market data.

        Args:
            data: Current market data (OHLCV)
            portfolio_value: Current portfolio value
            positions: Current positions

        Returns:
            Trading signal dictionary or None for no action
        """
        pass


class BuyAndHoldStrategy(Strategy):
    """Simple buy and hold strategy."""

    def __init__(self) -> None:
        super().__init__("Buy and Hold")
        self.bought = False

    def generate_signal(
        self, data: pd.Series, portfolio_value: float, positions: Dict[str, float]
    ) -> Optional[Dict[str, Union[str, float]]]:
        """Generate buy signal on first day, then hold."""
        if not self.bought:
            self.bought = True
            return {
                "action": "buy",
                "quantity": portfolio_value / data["close"],
                "price": data["close"],
            }
        return None


class MovingAverageCrossoverStrategy(Strategy):
    """Moving average crossover strategy."""

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        """
        Initialize the strategy.

        Args:
            short_window: Short moving average window
            long_window: Long moving average window
        """
        super().__init__("Moving Average Crossover")
        self.short_window = short_window
        self.long_window = long_window
        self.prices = []

    def generate_signal(
        self, data: pd.Series, portfolio_value: float, positions: Dict[str, float]
    ) -> Optional[Dict[str, Union[str, float]]]:
        """Generate signal based on moving average crossover."""
        self.prices.append(data["close"])

        if len(self.prices) < self.long_window:
            return None

        # Calculate moving averages
        short_ma = np.mean(self.prices[-self.short_window :])
        long_ma = np.mean(self.prices[-self.long_window :])

        # Previous values for crossover detection
        if len(self.prices) > self.long_window:
            prev_short_ma = np.mean(self.prices[-self.short_window - 1 : -1])
            prev_long_ma = np.mean(self.prices[-self.long_window - 1 : -1])

            # Golden cross (buy signal)
            if prev_short_ma <= prev_long_ma and short_ma > long_ma:
                return {
                    "action": "buy",
                    "quantity": portfolio_value / data["close"],
                    "price": data["close"],
                }
            # Death cross (sell signal)
            elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
                return {
                    "action": "sell",
                    "quantity": positions.get("quantity", 0),
                    "price": data["close"],
                }

        return None
