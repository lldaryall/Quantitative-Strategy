"""Tests for the backtester module."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from qbacktester import (
    Backtester,
    BuyAndHoldStrategy,
    DataLoader,
    MovingAverageCrossoverStrategy,
    StrategyParams,
)


class TestBacktester:
    """Test cases for the Backtester class."""

    def test_initialization(self):
        """Test backtester initialization with new API."""
        # Create sample data for new API
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            {"Close": [100, 101, 102, 103, 104], "Open": [100, 101, 102, 103, 104]},
            index=dates,
        )
        signals = pd.Series([0, 1, 0, 1, 0], index=dates)
        params = StrategyParams(
            "TEST", "2020-01-01", "2020-01-05", 5, 10, initial_cash=50000
        )

        backtester = Backtester(price_df, signals, params)
        assert backtester.params.initial_cash == 50000

    def test_run_with_buy_hold_strategy(self):
        """Test running backtest with new API."""
        # Create sample data
        dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
        price_df = pd.DataFrame(
            {
                "Close": [100 + i for i in range(10)],
                "Open": [100 + i for i in range(10)],
            },
            index=dates,
        )
        signals = pd.Series([0, 1, 1, 0, 0, 1, 1, 0, 0, 0], index=dates)
        params = StrategyParams(
            "TEST", "2020-01-01", "2020-01-10", 5, 10, initial_cash=100000
        )

        backtester = Backtester(price_df, signals, params)
        result = backtester.run()

        assert isinstance(result, pd.DataFrame)
        assert "total_equity" in result.columns
        assert "holdings_value" in result.columns
        assert "cash" in result.columns


class TestDataLoader:
    """Test cases for the DataLoader class."""

    def test_initialization(self):
        """Test data loader initialization."""
        loader = DataLoader()
        assert loader is not None

    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        dates = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
        data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [102, 103, 104, 105, 106],
                "volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=dates,
        )

        loader = DataLoader()
        assert loader.validate_data(data) is True

    def test_validate_data_invalid_columns(self):
        """Test data validation with missing columns."""
        dates = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
        data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                # Missing low, close, volume columns
            },
            index=dates,
        )

        loader = DataLoader()
        assert loader.validate_data(data) is False


class TestStrategies:
    """Test cases for strategy classes."""

    def test_buy_and_hold_strategy(self):
        """Test buy and hold strategy."""
        strategy = BuyAndHoldStrategy()
        assert strategy.name == "Buy and Hold"
        assert strategy.bought is False

        # Test first signal (should be buy)
        data = pd.Series({"close": 100})
        signal = strategy.generate_signal(data, 10000, {})
        assert signal is not None
        assert signal["action"] == "buy"
        assert strategy.bought is True

        # Test second signal (should be None)
        signal = strategy.generate_signal(data, 10000, {})
        assert signal is None

    def test_moving_average_crossover_strategy(self):
        """Test moving average crossover strategy."""
        strategy = MovingAverageCrossoverStrategy(short_window=2, long_window=4)
        assert strategy.name == "Moving Average Crossover"
        assert strategy.short_window == 2
        assert strategy.long_window == 4

        # Test with insufficient data
        data = pd.Series({"close": 100})
        signal = strategy.generate_signal(data, 10000, {})
        assert signal is None

        # Test with sufficient data for crossover
        prices = [100, 101, 102, 103, 104, 105]  # Upward trend
        for price in prices:
            data = pd.Series({"close": price})
            signal = strategy.generate_signal(data, 10000, {})
            # Should generate buy signal when short MA crosses above long MA
