"""Tests for the metrics module."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from qbacktester.metrics import (
    avg_win_loss,
    cagr,
    calmar,
    cvar,
    daily_returns,
    hit_rate,
    max_drawdown,
    sharpe,
    sortino,
    var,
    volatility,
)


class TestDailyReturns:
    """Test cases for daily_returns function."""

    def test_daily_returns_basic(self):
        """Test basic daily returns calculation."""
        equity = pd.Series(
            [100, 101, 102, 100, 98],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        returns = daily_returns(equity)

        expected = pd.Series([0.0, 0.01, 0.0099, -0.0196, -0.02], index=equity.index)

        pd.testing.assert_series_equal(returns, expected, atol=1e-4)

    def test_daily_returns_constant_series(self):
        """Test daily returns with constant equity (no changes)."""
        equity = pd.Series(
            [100, 100, 100, 100], index=pd.date_range("2020-01-01", periods=4, freq="D")
        )
        returns = daily_returns(equity)

        expected = pd.Series([0.0, 0.0, 0.0, 0.0], index=equity.index)
        pd.testing.assert_series_equal(returns, expected)

    def test_daily_returns_empty_series(self):
        """Test daily returns with empty series."""
        equity = pd.Series([], dtype=float)

        with pytest.raises(ValueError, match="equity_curve cannot be empty"):
            daily_returns(equity)

    def test_daily_returns_negative_values(self):
        """Test daily returns with negative values."""
        equity = pd.Series(
            [100, 101, -50, 102], index=pd.date_range("2020-01-01", periods=4, freq="D")
        )

        with pytest.raises(
            ValueError, match="equity_curve must contain only positive values"
        ):
            daily_returns(equity)

    def test_daily_returns_zero_values(self):
        """Test daily returns with zero values."""
        equity = pd.Series(
            [100, 101, 0, 102], index=pd.date_range("2020-01-01", periods=4, freq="D")
        )

        with pytest.raises(
            ValueError, match="equity_curve must contain only positive values"
        ):
            daily_returns(equity)

    def test_daily_returns_single_value(self):
        """Test daily returns with single value."""
        equity = pd.Series(
            [100], index=pd.date_range("2020-01-01", periods=1, freq="D")
        )
        returns = daily_returns(equity)

        expected = pd.Series([0.0], index=equity.index)
        pd.testing.assert_series_equal(returns, expected)


class TestCAGR:
    """Test cases for cagr function."""

    def test_cagr_basic(self):
        """Test basic CAGR calculation."""
        # 20% return over 1 year (252 days)
        equity = pd.Series(
            [100] + [100 * (1.2 ** (i / 252)) for i in range(1, 253)],
            index=pd.date_range("2020-01-01", periods=253, freq="D"),
        )

        cagr_value = cagr(equity)
        assert abs(cagr_value - 0.20) < 0.01  # Should be close to 20%

    def test_cagr_constant_series(self):
        """Test CAGR with constant equity (no growth)."""
        equity = pd.Series(
            [100, 100, 100, 100], index=pd.date_range("2020-01-01", periods=4, freq="D")
        )

        cagr_value = cagr(equity)
        assert cagr_value == 0.0

    def test_cagr_negative_returns(self):
        """Test CAGR with negative returns."""
        # -10% return over 1 year
        equity = pd.Series(
            [100] + [100 * (0.9 ** (i / 252)) for i in range(1, 253)],
            index=pd.date_range("2020-01-01", periods=253, freq="D"),
        )

        cagr_value = cagr(equity)
        assert abs(cagr_value - (-0.10)) < 0.01  # Should be close to -10%

    def test_cagr_empty_series(self):
        """Test CAGR with empty series."""
        equity = pd.Series([], dtype=float)

        with pytest.raises(ValueError, match="equity_curve cannot be empty"):
            cagr(equity)

    def test_cagr_single_value(self):
        """Test CAGR with single value."""
        equity = pd.Series(
            [100], index=pd.date_range("2020-01-01", periods=1, freq="D")
        )

        cagr_value = cagr(equity)
        assert cagr_value == 0.0

    def test_cagr_custom_periods(self):
        """Test CAGR with custom periods per year."""
        # 20% return over 1 year (12 months)
        equity = pd.Series(
            [100] + [100 * (1.2 ** (i / 12)) for i in range(1, 13)],
            index=pd.date_range("2020-01-01", periods=13, freq="ME"),
        )

        cagr_value = cagr(equity, periods_per_year=12)
        assert abs(cagr_value - 0.20) < 0.01


class TestSharpe:
    """Test cases for sharpe function."""

    def test_sharpe_basic(self):
        """Test basic Sharpe ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe_ratio = sharpe(returns)

        # Should be positive for this series
        assert sharpe_ratio > 0

    def test_sharpe_constant_returns(self):
        """Test Sharpe ratio with constant returns."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        sharpe_ratio = sharpe(returns)

        # Should be infinity (zero volatility)
        assert sharpe_ratio == np.inf

    def test_sharpe_zero_returns(self):
        """Test Sharpe ratio with zero returns."""
        returns = pd.Series([0.0, 0.0, 0.0, 0.0])
        sharpe_ratio = sharpe(returns)

        assert sharpe_ratio == 0.0

    def test_sharpe_negative_returns(self):
        """Test Sharpe ratio with negative returns."""
        returns = pd.Series([-0.01, -0.02, -0.01, -0.03])
        sharpe_ratio = sharpe(returns)

        # Should be negative
        assert sharpe_ratio < 0

    def test_sharpe_with_risk_free(self):
        """Test Sharpe ratio with risk-free rate."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        sharpe_ratio = sharpe(returns, risk_free=0.05)

        # Should be lower than without risk-free rate
        sharpe_no_rf = sharpe(returns, risk_free=0.0)
        assert sharpe_ratio < sharpe_no_rf

    def test_sharpe_empty_series(self):
        """Test Sharpe ratio with empty series."""
        returns = pd.Series([], dtype=float)

        with pytest.raises(ValueError, match="returns cannot be empty"):
            sharpe(returns)

    def test_sharpe_with_nans(self):
        """Test Sharpe ratio with NaN values."""
        returns = pd.Series([0.01, np.nan, 0.02, np.nan, -0.01])
        sharpe_ratio = sharpe(returns)

        # Should handle NaNs gracefully
        assert not np.isnan(sharpe_ratio)


class TestMaxDrawdown:
    """Test cases for max_drawdown function."""

    def test_max_drawdown_basic(self):
        """Test basic max drawdown calculation."""
        equity = pd.Series(
            [100, 110, 105, 120, 115, 130, 125, 140],
            index=pd.date_range("2020-01-01", periods=8, freq="D"),
        )

        max_dd, peak_date, trough_date = max_drawdown(equity)

        # Max drawdown should be from 110 to 105 = 4.55%
        expected_dd = (110 - 105) / 110
        assert abs(max_dd - expected_dd) < 0.001
        assert peak_date == pd.Timestamp("2020-01-02")
        assert trough_date == pd.Timestamp("2020-01-03")

    def test_max_drawdown_constant_series(self):
        """Test max drawdown with constant equity."""
        equity = pd.Series(
            [100, 100, 100, 100], index=pd.date_range("2020-01-01", periods=4, freq="D")
        )

        max_dd, peak_date, trough_date = max_drawdown(equity)

        assert max_dd == 0.0
        assert peak_date == equity.index[0]
        assert trough_date == equity.index[0]

    def test_max_drawdown_monotonic_increase(self):
        """Test max drawdown with monotonic increase."""
        equity = pd.Series(
            [100, 101, 102, 103, 104],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )

        max_dd, peak_date, trough_date = max_drawdown(equity)

        assert max_dd == 0.0
        assert peak_date == equity.index[0]
        assert trough_date == equity.index[0]

    def test_max_drawdown_monotonic_decrease(self):
        """Test max drawdown with monotonic decrease."""
        equity = pd.Series(
            [100, 99, 98, 97, 96],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )

        max_dd, peak_date, trough_date = max_drawdown(equity)

        # Max drawdown should be from 100 to 96 = 4%
        assert abs(max_dd - 0.04) < 0.001
        assert peak_date == equity.index[0]
        assert trough_date == equity.index[-1]

    def test_max_drawdown_empty_series(self):
        """Test max drawdown with empty series."""
        equity = pd.Series([], dtype=float)

        with pytest.raises(ValueError, match="equity_curve cannot be empty"):
            max_drawdown(equity)

    def test_max_drawdown_single_value(self):
        """Test max drawdown with single value."""
        equity = pd.Series(
            [100], index=pd.date_range("2020-01-01", periods=1, freq="D")
        )

        max_dd, peak_date, trough_date = max_drawdown(equity)

        assert max_dd == 0.0
        assert peak_date == equity.index[0]
        assert trough_date == equity.index[0]


class TestCalmar:
    """Test cases for calmar function."""

    def test_calmar_basic(self):
        """Test basic Calmar ratio calculation."""
        # Create equity curve with known CAGR and max drawdown
        equity = pd.Series(
            [100, 110, 105, 120, 115, 130, 125, 140],
            index=pd.date_range("2020-01-01", periods=8, freq="D"),
        )

        calmar_ratio = calmar(equity)

        # Should be positive
        assert calmar_ratio > 0

    def test_calmar_constant_series(self):
        """Test Calmar ratio with constant equity."""
        equity = pd.Series(
            [100, 100, 100, 100], index=pd.date_range("2020-01-01", periods=4, freq="D")
        )

        calmar_ratio = calmar(equity)

        # Should be infinity (zero max drawdown)
        assert calmar_ratio == np.inf

    def test_calmar_negative_cagr(self):
        """Test Calmar ratio with negative CAGR."""
        equity = pd.Series(
            [100, 99, 98, 97, 96],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )

        calmar_ratio = calmar(equity)

        # Should be negative
        assert calmar_ratio < 0

    def test_calmar_empty_series(self):
        """Test Calmar ratio with empty series."""
        equity = pd.Series([], dtype=float)

        with pytest.raises(ValueError, match="equity_curve cannot be empty"):
            calmar(equity)


class TestHitRate:
    """Test cases for hit_rate function."""

    def test_hit_rate_basic(self):
        """Test basic hit rate calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        hit_rate_value = hit_rate(returns)

        # 2 out of 4 non-first returns are positive = 50%
        assert abs(hit_rate_value - 0.5) < 0.001

    def test_hit_rate_all_positive(self):
        """Test hit rate with all positive returns."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.01])
        hit_rate_value = hit_rate(returns)

        assert hit_rate_value == 1.0

    def test_hit_rate_all_negative(self):
        """Test hit rate with all negative returns."""
        returns = pd.Series([-0.01, -0.02, -0.03, -0.01])
        hit_rate_value = hit_rate(returns)

        assert hit_rate_value == 0.0

    def test_hit_rate_with_threshold(self):
        """Test hit rate with custom threshold."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.01])
        hit_rate_value = hit_rate(returns, threshold=0.015)

        # 2 out of 3 non-first returns above 1.5% = 66.67%
        assert abs(hit_rate_value - 2 / 3) < 0.001

    def test_hit_rate_empty_series(self):
        """Test hit rate with empty series."""
        returns = pd.Series([], dtype=float)

        with pytest.raises(ValueError, match="returns cannot be empty"):
            hit_rate(returns)

    def test_hit_rate_with_nans(self):
        """Test hit rate with NaN values."""
        returns = pd.Series([0.01, np.nan, 0.02, np.nan, -0.01])
        hit_rate_value = hit_rate(returns)

        # 1 out of 2 non-NaN non-first returns are positive = 50%
        assert abs(hit_rate_value - 0.5) < 0.001


class TestAvgWinLoss:
    """Test cases for avg_win_loss function."""

    def test_avg_win_loss_basic(self):
        """Test basic average win/loss calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        avg_win, avg_loss = avg_win_loss(returns)

        # Average win: (0.01 + 0.03 + 0.02) / 3 = 0.02
        # Average loss: (0.02 + 0.01) / 2 = 0.015
        assert abs(avg_win - 0.02) < 0.001
        assert abs(avg_loss - 0.015) < 0.001

    def test_avg_win_loss_all_positive(self):
        """Test average win/loss with all positive returns."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.01])
        avg_win, avg_loss = avg_win_loss(returns)

        # Average win: (0.01 + 0.02 + 0.03 + 0.01) / 4 = 0.0175
        # Average loss: 0 (no losses)
        assert abs(avg_win - 0.0175) < 0.001
        assert avg_loss == 0.0

    def test_avg_win_loss_all_negative(self):
        """Test average win/loss with all negative returns."""
        returns = pd.Series([-0.01, -0.02, -0.03, -0.01])
        avg_win, avg_loss = avg_win_loss(returns)

        # Average win: 0 (no wins)
        # Average loss: (0.01 + 0.02 + 0.03 + 0.01) / 4 = 0.0175
        assert avg_win == 0.0
        assert abs(avg_loss - 0.0175) < 0.001

    def test_avg_win_loss_empty_series(self):
        """Test average win/loss with empty series."""
        returns = pd.Series([], dtype=float)

        with pytest.raises(ValueError, match="returns cannot be empty"):
            avg_win_loss(returns)

    def test_avg_win_loss_with_nans(self):
        """Test average win/loss with NaN values."""
        returns = pd.Series([0.01, np.nan, -0.02, np.nan, 0.03])
        avg_win, avg_loss = avg_win_loss(returns)

        # Average win: (0.01 + 0.03) / 2 = 0.02
        # Average loss: 0.02 / 1 = 0.02
        assert abs(avg_win - 0.02) < 0.001
        assert abs(avg_loss - 0.02) < 0.001


class TestAdditionalMetrics:
    """Test cases for additional metrics functions."""

    def test_volatility_basic(self):
        """Test basic volatility calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        vol = volatility(returns)

        # Should be positive
        assert vol > 0

    def test_volatility_constant_returns(self):
        """Test volatility with constant returns."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        vol = volatility(returns)

        assert vol == 0.0

    def test_sortino_basic(self):
        """Test basic Sortino ratio calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        sortino_ratio = sortino(returns)

        # Should be positive
        assert sortino_ratio > 0

    def test_var_basic(self):
        """Test basic VaR calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.05, 0.01])
        var_value = var(returns, confidence_level=0.05)

        # Should be positive (representing loss)
        assert var_value > 0

    def test_cvar_basic(self):
        """Test basic CVaR calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.05, 0.01])
        cvar_value = cvar(returns, confidence_level=0.05)

        # Should be positive (representing expected loss)
        assert cvar_value > 0


class TestRealisticData:
    """Test cases with realistic noisy data."""

    def test_realistic_equity_curve(self):
        """Test metrics with realistic equity curve."""
        # Create realistic equity curve with noise
        np.random.seed(42)
        n_days = 252  # 1 year
        base_return = 0.0008  # ~20% annual return
        volatility = 0.02  # 2% daily volatility

        returns = np.random.normal(base_return, volatility, n_days)
        equity = 100 * np.cumprod(1 + returns)
        equity_series = pd.Series(
            equity, index=pd.date_range("2020-01-01", periods=n_days, freq="D")
        )

        # Test all metrics
        daily_ret = daily_returns(equity_series)
        cagr_value = cagr(equity_series)
        sharpe_ratio = sharpe(daily_ret)
        max_dd, peak_date, trough_date = max_drawdown(equity_series)
        calmar_ratio = calmar(equity_series)
        hit_rate_value = hit_rate(daily_ret)
        avg_win, avg_loss = avg_win_loss(daily_ret)

        # Basic sanity checks
        assert len(daily_ret) == n_days
        assert daily_ret.iloc[0] == 0.0  # First return should be 0
        assert cagr_value > 0  # Should be positive for this series
        assert sharpe_ratio > 0  # Should be positive
        assert max_dd >= 0  # Drawdown should be non-negative
        assert calmar_ratio > 0  # Should be positive
        assert 0 <= hit_rate_value <= 1  # Hit rate should be between 0 and 1
        assert avg_win >= 0  # Average win should be non-negative
        assert avg_loss >= 0  # Average loss should be non-negative

    def test_negative_only_series(self):
        """Test metrics with series that only has negative returns."""
        returns = pd.Series([-0.01, -0.02, -0.01, -0.03, -0.01])

        # Test metrics that should handle negative returns
        hit_rate_value = hit_rate(returns)
        avg_win, avg_loss = avg_win_loss(returns)
        sharpe_ratio = sharpe(returns)

        assert hit_rate_value == 0.0  # No positive returns
        assert avg_win == 0.0  # No wins
        assert avg_loss > 0  # Should have losses
        assert sharpe_ratio < 0  # Should be negative

    def test_constant_series_comprehensive(self):
        """Test all metrics with constant series."""
        equity = pd.Series(
            [100, 100, 100, 100, 100],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        returns = daily_returns(equity)

        # Test all metrics
        cagr_value = cagr(equity)
        sharpe_ratio = sharpe(returns)
        max_dd, peak_date, trough_date = max_drawdown(equity)
        calmar_ratio = calmar(equity)
        hit_rate_value = hit_rate(returns)
        avg_win, avg_loss = avg_win_loss(returns)
        vol = volatility(returns)
        sortino_ratio = sortino(returns)

        # All should be 0 or special values
        assert cagr_value == 0.0
        assert sharpe_ratio == 0.0
        assert max_dd == 0.0
        assert calmar_ratio == np.inf
        assert hit_rate_value == 0.0
        assert avg_win == 0.0
        assert avg_loss == 0.0
        assert vol == 0.0
        assert sortino_ratio == 0.0


class TestEdgeCases:
    """Test cases for edge cases and error conditions."""

    def test_single_value_equity(self):
        """Test metrics with single value equity curve."""
        equity = pd.Series(
            [100], index=pd.date_range("2020-01-01", periods=1, freq="D")
        )

        # Test metrics that can handle single values
        cagr_value = cagr(equity)
        max_dd, peak_date, trough_date = max_drawdown(equity)
        calmar_ratio = calmar(equity)

        assert cagr_value == 0.0
        assert max_dd == 0.0
        assert calmar_ratio == np.inf

    def test_two_value_equity(self):
        """Test metrics with two value equity curve."""
        equity = pd.Series(
            [100, 110], index=pd.date_range("2020-01-01", periods=2, freq="D")
        )
        returns = daily_returns(equity)

        # Test all metrics
        cagr_value = cagr(equity)
        sharpe_ratio = sharpe(returns)
        max_dd, peak_date, trough_date = max_drawdown(equity)
        calmar_ratio = calmar(equity)
        hit_rate_value = hit_rate(returns)
        avg_win, avg_loss = avg_win_loss(returns)

        # Basic checks
        assert cagr_value > 0  # Should be positive
        assert max_dd == 0.0  # No drawdown
        assert calmar_ratio == np.inf  # No drawdown
        assert hit_rate_value == 1.0  # One positive return (excluding first 0)
        assert avg_win > 0  # Should have a win
        assert avg_loss == 0.0  # No losses

    def test_invalid_confidence_levels(self):
        """Test VaR and CVaR with invalid confidence levels."""
        returns = pd.Series([0.01, -0.02, 0.03])

        with pytest.raises(
            ValueError, match="confidence_level must be between 0 and 1"
        ):
            var(returns, confidence_level=0.0)

        with pytest.raises(
            ValueError, match="confidence_level must be between 0 and 1"
        ):
            var(returns, confidence_level=1.0)

        with pytest.raises(
            ValueError, match="confidence_level must be between 0 and 1"
        ):
            cvar(returns, confidence_level=-0.1)

        with pytest.raises(
            ValueError, match="confidence_level must be between 0 and 1"
        ):
            cvar(returns, confidence_level=1.5)
