"""Tests for the indicators module."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from qbacktester.indicators import (
    bollinger_bands,
    crossover,
    ema,
    macd,
    rsi,
    sma,
    stochastic,
)


class TestSMA:
    """Test cases for Simple Moving Average function."""

    def test_sma_basic(self):
        """Test basic SMA calculation."""
        # Create test data
        dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"Close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]}, index=dates
        )

        result = sma(df, window=3, col="Close")

        # Check first few values
        assert result.iloc[0] == 100.0  # First value should be the same
        assert result.iloc[1] == 100.5  # (100 + 101) / 2
        assert result.iloc[2] == 101.0  # (100 + 101 + 102) / 3
        assert result.iloc[3] == 102.0  # (101 + 102 + 103) / 3
        assert len(result) == 10

    def test_sma_with_nans(self):
        """Test SMA with NaN values in data."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        df = pd.DataFrame({"Close": [100, np.nan, 102, 103, 104]}, index=dates)

        result = sma(df, window=3, col="Close")

        # Should handle NaNs gracefully
        assert not result.isnull().all()
        assert len(result) == 5

    def test_sma_short_series(self):
        """Test SMA with series shorter than window."""
        dates = pd.date_range(start="2020-01-01", periods=2, freq="D")
        df = pd.DataFrame({"Close": [100, 101]}, index=dates)

        result = sma(df, window=5, col="Close")

        # Should return series with NaN values
        assert len(result) == 2
        assert result.iloc[0] == 100.0  # First value
        assert result.iloc[1] == 100.5  # Average of two values

    def test_sma_invalid_window(self):
        """Test SMA with invalid window."""
        df = pd.DataFrame({"Close": [100, 101, 102]})

        with pytest.raises(ValueError, match="Window must be positive"):
            sma(df, window=0, col="Close")

        with pytest.raises(ValueError, match="Window must be positive"):
            sma(df, window=-1, col="Close")

    def test_sma_missing_column(self):
        """Test SMA with missing column."""
        df = pd.DataFrame({"Open": [100, 101, 102]})

        with pytest.raises(ValueError, match="Column 'Close' not found"):
            sma(df, window=3, col="Close")

    def test_sma_empty_dataframe(self):
        """Test SMA with empty DataFrame."""
        df = pd.DataFrame(columns=["Close"])

        result = sma(df, window=3, col="Close")
        assert len(result) == 0


class TestEMA:
    """Test cases for Exponential Moving Average function."""

    def test_ema_basic(self):
        """Test basic EMA calculation."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates)

        result = ema(df, span=3, col="Close")

        # EMA should be calculated
        assert len(result) == 5
        assert not result.isnull().all()
        # First value should be the same as input
        assert result.iloc[0] == 100.0

    def test_ema_with_nans(self):
        """Test EMA with NaN values."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        df = pd.DataFrame({"Close": [100, np.nan, 102, 103, 104]}, index=dates)

        result = ema(df, span=3, col="Close")

        # Should handle NaNs
        assert len(result) == 5
        assert not result.isnull().all()

    def test_ema_short_series(self):
        """Test EMA with series shorter than span."""
        dates = pd.date_range(start="2020-01-01", periods=2, freq="D")
        df = pd.DataFrame({"Close": [100, 101]}, index=dates)

        result = ema(df, span=5, col="Close")

        # Should still calculate EMA
        assert len(result) == 2
        assert result.iloc[0] == 100.0

    def test_ema_invalid_span(self):
        """Test EMA with invalid span."""
        df = pd.DataFrame({"Close": [100, 101, 102]})

        with pytest.raises(ValueError, match="Span must be positive"):
            ema(df, span=0, col="Close")

    def test_ema_missing_column(self):
        """Test EMA with missing column."""
        df = pd.DataFrame({"Open": [100, 101, 102]})

        with pytest.raises(ValueError, match="Column 'Close' not found"):
            ema(df, span=3, col="Close")


class TestCrossover:
    """Test cases for crossover function."""

    def test_crossover_basic(self):
        """Test basic crossover detection."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        fast = pd.Series([100, 101, 102, 101, 100], index=dates)
        slow = pd.Series([99, 100, 101, 102, 103], index=dates)

        result = crossover(fast, slow)

        # Check crossover points
        assert result.iloc[0] == 0  # First value should be 0
        assert result.iloc[1] == 0  # No crossover (both diff and diff_prev > 0)
        assert result.iloc[2] == 0  # No crossover (both diff and diff_prev > 0)
        assert result.iloc[3] == -1  # Fast crosses below slow
        assert result.iloc[4] == 0  # No crossover

    def test_crossover_no_crossovers(self):
        """Test crossover with no crossovers."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        fast = pd.Series([100, 101, 102, 103, 104], index=dates)
        slow = pd.Series([99, 100, 101, 102, 103], index=dates)

        result = crossover(fast, slow)

        # Should be all zeros except first
        assert (result == 0).all()

    def test_crossover_identical_series(self):
        """Test crossover with identical series."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        series = pd.Series([100, 101, 102, 103, 104], index=dates)

        result = crossover(series, series)

        # Should be all zeros
        assert (result == 0).all()

    def test_crossover_with_nans(self):
        """Test crossover with NaN values."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        fast = pd.Series([100, np.nan, 102, 101, 100], index=dates)
        slow = pd.Series([99, 100, 101, 102, 103], index=dates)

        result = crossover(fast, slow)

        # Should handle NaNs gracefully
        assert len(result) == 5
        assert result.iloc[0] == 0  # First value should be 0

    def test_crossover_short_series(self):
        """Test crossover with short series."""
        dates = pd.date_range(start="2020-01-01", periods=2, freq="D")
        fast = pd.Series([100, 101], index=dates)
        slow = pd.Series([99, 100], index=dates)

        result = crossover(fast, slow)

        # Should handle short series
        assert len(result) == 2
        assert result.iloc[0] == 0  # First value should be 0
        assert result.iloc[1] == 0  # No crossover (both diff and diff_prev > 0)

    def test_crossover_different_indices(self):
        """Test crossover with different indices."""
        dates1 = pd.date_range(start="2020-01-01", periods=3, freq="D")
        dates2 = pd.date_range(start="2020-01-02", periods=3, freq="D")
        fast = pd.Series([100, 101, 102], index=dates1)
        slow = pd.Series([99, 100, 101], index=dates2)

        with pytest.raises(
            ValueError, match="Fast and slow series must have the same index"
        ):
            crossover(fast, slow)

    def test_crossover_empty_series(self):
        """Test crossover with empty series."""
        dates = pd.date_range(start="2020-01-01", periods=0, freq="D")
        fast = pd.Series([], index=dates)
        slow = pd.Series([], index=dates)

        with pytest.raises(ValueError, match="Input series cannot be empty"):
            crossover(fast, slow)

    def test_crossover_equal_windows(self):
        """Test crossover when series are equal (edge case)."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        series = pd.Series([100, 101, 102, 103, 104], index=dates)

        result = crossover(series, series)

        # Should be all zeros
        assert (result == 0).all()


class TestRSI:
    """Test cases for RSI function."""

    def test_rsi_basic(self):
        """Test basic RSI calculation."""
        dates = pd.date_range(start="2020-01-01", periods=20, freq="D")
        # Create trending data
        prices = [100 + i for i in range(20)]
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = rsi(df, window=14, col="Close")

        # RSI should be calculated
        assert len(result) == 20
        assert not result.isnull().all()
        # RSI should be between 0 and 100 (excluding NaN values)
        valid_rsi = result.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_invalid_window(self):
        """Test RSI with invalid window."""
        df = pd.DataFrame({"Close": [100, 101, 102]})

        with pytest.raises(ValueError, match="Window must be positive"):
            rsi(df, window=0, col="Close")

    def test_rsi_missing_column(self):
        """Test RSI with missing column."""
        df = pd.DataFrame({"Open": [100, 101, 102]})

        with pytest.raises(ValueError, match="Column 'Close' not found"):
            rsi(df, window=14, col="Close")


class TestMACD:
    """Test cases for MACD function."""

    def test_macd_basic(self):
        """Test basic MACD calculation."""
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        prices = [100 + i for i in range(30)]
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = macd(df, fast=12, slow=26, signal=9, col="Close")

        # Should return DataFrame with three columns
        assert isinstance(result, pd.DataFrame)
        assert "MACD" in result.columns
        assert "Signal" in result.columns
        assert "Histogram" in result.columns
        assert len(result) == 30

    def test_macd_invalid_periods(self):
        """Test MACD with invalid periods."""
        df = pd.DataFrame({"Close": [100, 101, 102]})

        with pytest.raises(ValueError, match="All periods must be positive"):
            macd(df, fast=0, slow=26, signal=9, col="Close")

    def test_macd_missing_column(self):
        """Test MACD with missing column."""
        df = pd.DataFrame({"Open": [100, 101, 102]})

        with pytest.raises(ValueError, match="Column 'Close' not found"):
            macd(df, col="Close")


class TestBollingerBands:
    """Test cases for Bollinger Bands function."""

    def test_bollinger_bands_basic(self):
        """Test basic Bollinger Bands calculation."""
        dates = pd.date_range(start="2020-01-01", periods=25, freq="D")
        prices = [100 + np.sin(i * 0.1) * 5 for i in range(25)]
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = bollinger_bands(df, window=20, std_dev=2.0, col="Close")

        # Should return DataFrame with three columns
        assert isinstance(result, pd.DataFrame)
        assert "Upper" in result.columns
        assert "Middle" in result.columns
        assert "Lower" in result.columns
        assert len(result) == 25

        # Upper should be higher than middle, middle higher than lower (excluding NaN values)
        valid_data = result.dropna()
        assert (valid_data["Upper"] >= valid_data["Middle"]).all()
        assert (valid_data["Middle"] >= valid_data["Lower"]).all()

    def test_bollinger_bands_invalid_window(self):
        """Test Bollinger Bands with invalid window."""
        df = pd.DataFrame({"Close": [100, 101, 102]})

        with pytest.raises(ValueError, match="Window must be positive"):
            bollinger_bands(df, window=0, col="Close")

    def test_bollinger_bands_missing_column(self):
        """Test Bollinger Bands with missing column."""
        df = pd.DataFrame({"Open": [100, 101, 102]})

        with pytest.raises(ValueError, match="Column 'Close' not found"):
            bollinger_bands(df, col="Close")


class TestStochastic:
    """Test cases for Stochastic Oscillator function."""

    def test_stochastic_basic(self):
        """Test basic Stochastic calculation."""
        dates = pd.date_range(start="2020-01-01", periods=20, freq="D")
        df = pd.DataFrame(
            {
                "High": [105 + i for i in range(20)],
                "Low": [95 + i for i in range(20)],
                "Close": [100 + i for i in range(20)],
            },
            index=dates,
        )

        result = stochastic(df, k_window=14, d_window=3, col="Close")

        # Should return DataFrame with two columns
        assert isinstance(result, pd.DataFrame)
        assert "K" in result.columns
        assert "D" in result.columns
        assert len(result) == 20

        # K and D should be between 0 and 100
        assert (result["K"] >= 0).all()
        assert (result["K"] <= 100).all()
        assert (result["D"] >= 0).all()
        assert (result["D"] <= 100).all()

    def test_stochastic_invalid_windows(self):
        """Test Stochastic with invalid windows."""
        df = pd.DataFrame({"High": [105], "Low": [95], "Close": [100]})

        with pytest.raises(ValueError, match="All windows must be positive"):
            stochastic(df, k_window=0, d_window=3, col="Close")

    def test_stochastic_missing_columns(self):
        """Test Stochastic with missing columns."""
        df = pd.DataFrame({"Close": [100, 101, 102]})

        with pytest.raises(ValueError, match="Missing required columns"):
            stochastic(df, col="Close")


class TestEdgeCases:
    """Test edge cases across all indicators."""

    def test_indicators_with_single_value(self):
        """Test indicators with single value data."""
        df = pd.DataFrame(
            {"Close": [100]}, index=pd.date_range("2020-01-01", periods=1)
        )

        # SMA should work
        sma_result = sma(df, window=3, col="Close")
        assert len(sma_result) == 1
        assert sma_result.iloc[0] == 100.0

        # EMA should work
        ema_result = ema(df, span=3, col="Close")
        assert len(ema_result) == 1
        assert ema_result.iloc[0] == 100.0

    def test_indicators_with_all_nans(self):
        """Test indicators with all NaN values."""
        df = pd.DataFrame(
            {"Close": [np.nan] * 5}, index=pd.date_range("2020-01-01", periods=5)
        )

        # SMA should handle NaNs
        sma_result = sma(df, window=3, col="Close")
        assert len(sma_result) == 5

        # EMA should handle NaNs
        ema_result = ema(df, span=3, col="Close")
        assert len(ema_result) == 5

    def test_crossover_edge_cases(self):
        """Test crossover with various edge cases."""
        dates = pd.date_range(start="2020-01-01", periods=3, freq="D")

        # Test with equal values
        fast = pd.Series([100, 100, 100], index=dates)
        slow = pd.Series([100, 100, 100], index=dates)
        result = crossover(fast, slow)
        assert (result == 0).all()

        # Test with alternating values
        fast = pd.Series([100, 101, 100], index=dates)
        slow = pd.Series([100, 100, 101], index=dates)
        result = crossover(fast, slow)
        assert result.iloc[0] == 0  # First should be 0
        assert result.iloc[1] == 1  # Fast crosses above
        assert result.iloc[2] == -1  # Fast crosses below
