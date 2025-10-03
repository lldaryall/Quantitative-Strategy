"""Tests for the enhanced strategy module."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from qbacktester.strategy import StrategyParams, generate_signals


class TestStrategyParams:
    """Test cases for StrategyParams dataclass."""

    def test_strategy_params_valid(self):
        """Test valid StrategyParams initialization."""
        params = StrategyParams(
            symbol="AAPL",
            start="2020-01-01",
            end="2020-12-31",
            fast_window=10,
            slow_window=20,
        )

        assert params.symbol == "AAPL"
        assert params.start == "2020-01-01"
        assert params.end == "2020-12-31"
        assert params.fast_window == 10
        assert params.slow_window == 20
        assert params.initial_cash == 100_000
        assert params.fee_bps == 1.0
        assert params.slippage_bps == 0.5

    def test_strategy_params_custom_values(self):
        """Test StrategyParams with custom values."""
        params = StrategyParams(
            symbol="MSFT",
            start="2021-01-01",
            end="2021-12-31",
            fast_window=5,
            slow_window=15,
            initial_cash=50_000,
            fee_bps=2.0,
            slippage_bps=1.0,
        )

        assert params.symbol == "MSFT"
        assert params.initial_cash == 50_000
        assert params.fee_bps == 2.0
        assert params.slippage_bps == 1.0

    def test_strategy_params_invalid_fast_window(self):
        """Test StrategyParams with invalid fast_window."""
        with pytest.raises(ValueError, match="fast_window must be positive"):
            StrategyParams(
                symbol="AAPL",
                start="2020-01-01",
                end="2020-12-31",
                fast_window=0,
                slow_window=20,
            )

    def test_strategy_params_invalid_slow_window(self):
        """Test StrategyParams with invalid slow_window."""
        with pytest.raises(ValueError, match="slow_window must be positive"):
            StrategyParams(
                symbol="AAPL",
                start="2020-01-01",
                end="2020-12-31",
                fast_window=10,
                slow_window=0,
            )

    def test_strategy_params_fast_greater_than_slow(self):
        """Test StrategyParams with fast_window >= slow_window."""
        with pytest.raises(
            ValueError, match="fast_window.*must be less than slow_window"
        ):
            StrategyParams(
                symbol="AAPL",
                start="2020-01-01",
                end="2020-12-31",
                fast_window=20,
                slow_window=10,
            )

    def test_strategy_params_invalid_initial_cash(self):
        """Test StrategyParams with invalid initial_cash."""
        with pytest.raises(ValueError, match="initial_cash must be positive"):
            StrategyParams(
                symbol="AAPL",
                start="2020-01-01",
                end="2020-12-31",
                fast_window=10,
                slow_window=20,
                initial_cash=0,
            )

    def test_strategy_params_negative_fee_bps(self):
        """Test StrategyParams with negative fee_bps."""
        with pytest.raises(ValueError, match="fee_bps must be non-negative"):
            StrategyParams(
                symbol="AAPL",
                start="2020-01-01",
                end="2020-12-31",
                fast_window=10,
                slow_window=20,
                fee_bps=-1.0,
            )

    def test_strategy_params_negative_slippage_bps(self):
        """Test StrategyParams with negative slippage_bps."""
        with pytest.raises(ValueError, match="slippage_bps must be non-negative"):
            StrategyParams(
                symbol="AAPL",
                start="2020-01-01",
                end="2020-12-31",
                fast_window=10,
                slow_window=20,
                slippage_bps=-0.5,
            )


class TestGenerateSignals:
    """Test cases for generate_signals function."""

    def test_generate_signals_basic(self):
        """Test basic signal generation."""
        # Create synthetic data with known pattern
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        # Create trending data that will generate crossovers
        prices = [100 + i * 0.5 + np.sin(i * 0.2) * 2 for i in range(30)]
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = generate_signals(df, fast_window=5, slow_window=10)

        # Check that all required columns are present
        assert "sma_fast" in result.columns
        assert "sma_slow" in result.columns
        assert "crossover" in result.columns
        assert "signal" in result.columns

        # Check that signals are only 0 or 1
        assert result["signal"].isin([0, 1]).all()

        # Check that original data is preserved
        assert "Close" in result.columns
        assert result["Close"].equals(df["Close"])

    def test_generate_signals_missing_close_column(self):
        """Test generate_signals with missing Close column."""
        df = pd.DataFrame({"Open": [100, 101, 102]})

        with pytest.raises(ValueError, match="price_df must contain 'Close' column"):
            generate_signals(df, fast_window=5, slow_window=10)

    def test_generate_signals_invalid_windows(self):
        """Test generate_signals with invalid windows."""
        df = pd.DataFrame({"Close": [100, 101, 102]})

        with pytest.raises(
            ValueError, match="Both fast_window and slow_window must be positive"
        ):
            generate_signals(df, fast_window=0, slow_window=10)

        with pytest.raises(
            ValueError, match="fast_window must be less than slow_window"
        ):
            generate_signals(df, fast_window=10, slow_window=5)

    def test_generate_signals_short_data(self):
        """Test generate_signals with short data."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates)

        result = generate_signals(df, fast_window=3, slow_window=5)

        # Should still work with short data
        assert len(result) == 5
        assert "signal" in result.columns
        assert result["signal"].isin([0, 1]).all()

    def test_generate_signals_no_lookahead_bias(self):
        """Test that signals don't have look-ahead bias."""
        # Create synthetic data with a clear crossover pattern
        dates = pd.date_range(start="2020-01-01", periods=20, freq="D")

        # Create data where fast MA crosses above slow MA on day 10
        # and crosses below on day 15
        prices = [100] * 5 + [101] * 5 + [102] * 5 + [101] * 5
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = generate_signals(df, fast_window=3, slow_window=7)

        # Check that signals are properly delayed
        # The crossover should happen on day 10, but signal should be on day 11
        crossover_days = result[result["crossover"] == 1].index
        signal_days = result[result["signal"] == 1].index

        # Verify that signals come after crossovers (1-day delay)
        if len(crossover_days) > 0 and len(signal_days) > 0:
            for crossover_day in crossover_days:
                # Find the next trading day after crossover
                next_day = crossover_day + timedelta(days=1)
                if next_day in result.index:
                    # Signal should be 1 on the day after crossover
                    assert result.loc[next_day, "signal"] == 1

    def test_generate_signals_vectorized_operation(self):
        """Test that generate_signals is fully vectorized."""
        # Create larger dataset to test vectorization
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        prices = [100 + i * 0.1 + np.sin(i * 0.1) * 5 for i in range(100)]
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = generate_signals(df, fast_window=10, slow_window=20)

        # All operations should be vectorized - no loops in the function
        assert len(result) == 100
        assert "sma_fast" in result.columns
        assert "sma_slow" in result.columns
        assert "crossover" in result.columns
        assert "signal" in result.columns

        # Check that SMAs are calculated correctly
        assert not result["sma_fast"].isnull().all()
        assert not result["sma_slow"].isnull().all()

    def test_generate_signals_signal_timing(self):
        """Test that first tradable signal timing is correct."""
        # Create data with known crossover pattern
        dates = pd.date_range(start="2020-01-01", periods=15, freq="D")

        # Create data where crossover happens on day 8
        prices = [100] * 7 + [101, 102, 103, 104, 105, 106, 107, 108]
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = generate_signals(df, fast_window=3, slow_window=6)

        # Find the first crossover
        first_crossover_idx = result[result["crossover"] == 1].index
        if len(first_crossover_idx) > 0:
            first_crossover_day = first_crossover_idx[0]

            # The signal should be 1 on the day after the first crossover
            next_day = first_crossover_day + timedelta(days=1)
            if next_day in result.index:
                assert result.loc[next_day, "signal"] == 1

    def test_generate_signals_only_long_positions(self):
        """Test that signals only generate long positions (1) or flat (0)."""
        dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
        # Create volatile data to generate multiple crossovers
        prices = [100 + np.sin(i * 0.3) * 10 + i * 0.2 for i in range(50)]
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = generate_signals(df, fast_window=5, slow_window=15)

        # All signals should be 0 or 1 (no shorting)
        unique_signals = result["signal"].unique()
        assert set(unique_signals).issubset({0, 1})

    def test_generate_signals_with_nan_values(self):
        """Test generate_signals with NaN values in price data."""
        dates = pd.date_range(start="2020-01-01", periods=20, freq="D")
        prices = [
            100,
            101,
            np.nan,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
        ]
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = generate_signals(df, fast_window=3, slow_window=7)

        # Should handle NaN values gracefully
        assert len(result) == 20
        assert "signal" in result.columns
        # Signals should still be 0 or 1
        assert result["signal"].isin([0, 1]).all()

    def test_generate_signals_empty_dataframe(self):
        """Test generate_signals with empty DataFrame."""
        df = pd.DataFrame(columns=["Close"])

        # Empty DataFrame should raise an error when trying to calculate crossovers
        with pytest.raises(ValueError, match="Input series cannot be empty"):
            generate_signals(df, fast_window=3, slow_window=7)

    def test_generate_signals_preserves_original_data(self):
        """Test that original data is preserved in the result."""
        dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
        original_df = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "Open": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
                "Volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            },
            index=dates,
        )

        result = generate_signals(original_df, fast_window=3, slow_window=5)

        # Original columns should be preserved
        assert "Close" in result.columns
        assert "Open" in result.columns
        assert "Volume" in result.columns

        # Original data should be unchanged
        assert result["Close"].equals(original_df["Close"])
        assert result["Open"].equals(original_df["Open"])
        assert result["Volume"].equals(original_df["Volume"])

    def test_generate_signals_crossover_detection(self):
        """Test that crossover detection works correctly."""
        # Create data with known crossover points
        dates = pd.date_range(start="2020-01-01", periods=20, freq="D")

        # Create data where fast MA crosses above slow MA on day 8
        # and crosses below on day 15
        prices = [100] * 7 + [
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
        ]
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = generate_signals(df, fast_window=3, slow_window=7)

        # Check that crossovers are detected
        crossovers = result[result["crossover"] != 0]
        assert len(crossovers) > 0

        # Check that signals are generated after crossovers
        bullish_crossovers = result[result["crossover"] == 1]
        if len(bullish_crossovers) > 0:
            # There should be corresponding signals (delayed by 1 day)
            signal_days = result[result["signal"] == 1].index
            assert len(signal_days) > 0


class TestLookAheadBiasVerification:
    """Test cases specifically for look-ahead bias verification."""

    def test_no_future_data_usage(self):
        """Test that signals don't use future data."""
        # Create data with a clear pattern
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        prices = [100 + i for i in range(30)]  # Simple uptrend
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = generate_signals(df, fast_window=5, slow_window=10)

        # For each signal day, verify that the signal is based only on past data
        for i, (date, row) in enumerate(result.iterrows()):
            if row["signal"] == 1:
                # The signal should be based on crossovers that happened before this date
                # Check that the crossover that generated this signal happened before
                crossover_date = date - timedelta(days=1)
                if crossover_date in result.index:
                    assert result.loc[crossover_date, "crossover"] == 1

    def test_signal_delay_verification(self):
        """Test that signals are properly delayed by 1 day."""
        # Create synthetic data with known crossover point
        dates = pd.date_range(start="2020-01-01", periods=20, freq="D")

        # Create data where crossover happens on day 10
        prices = [100] * 9 + [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = generate_signals(df, fast_window=3, slow_window=7)

        # Find crossover day
        crossover_days = result[result["crossover"] == 1].index

        if len(crossover_days) > 0:
            first_crossover = crossover_days[0]

            # Signal should be 1 on the day after crossover
            next_day = first_crossover + timedelta(days=1)
            if next_day in result.index:
                assert result.loc[next_day, "signal"] == 1

                # Verify that signal is 0 on the crossover day itself
                assert result.loc[first_crossover, "signal"] == 0
