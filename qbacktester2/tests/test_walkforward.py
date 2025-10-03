"""Tests for the walkforward module."""

import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from qbacktester.walkforward import (
    _calculate_parameter_stability,
    _calculate_walkforward_summary,
    _concatenate_equity_curves,
    _count_trades,
    _evaluate_parameters,
    _generate_walkforward_windows,
    print_walkforward_results,
    walkforward_crossover,
)


class TestGenerateWalkforwardWindows:
    """Test cases for _generate_walkforward_windows function."""

    def test_generate_walkforward_windows_basic(self):
        """Test basic window generation."""
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp("2025-01-01")

        windows = _generate_walkforward_windows(
            start_date, end_date, in_sample_years=2, out_sample_years=1
        )

        assert len(windows) > 0
        assert all(
            len(window) == 4 for window in windows
        )  # (is_start, is_end, oos_start, oos_end)

        # Check first window
        first_window = windows[0]
        assert first_window[0] == start_date
        assert first_window[1] == start_date + pd.DateOffset(years=2)
        assert first_window[2] == start_date + pd.DateOffset(years=2)
        assert first_window[3] == start_date + pd.DateOffset(years=3)

    def test_generate_walkforward_windows_short_period(self):
        """Test window generation with short period."""
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp("2021-01-01")

        windows = _generate_walkforward_windows(
            start_date, end_date, in_sample_years=1, out_sample_years=1
        )

        # Should have no windows if period is too short
        assert len(windows) == 0

    def test_generate_walkforward_windows_overlapping(self):
        """Test that windows are properly overlapping."""
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp("2026-01-01")

        windows = _generate_walkforward_windows(
            start_date, end_date, in_sample_years=2, out_sample_years=1
        )

        assert len(windows) >= 2

        # Check that windows are overlapping (next IS starts before previous OOS ends)
        for i in range(len(windows) - 1):
            current_oos_end = windows[i][3]
            next_is_start = windows[i + 1][0]
            assert next_is_start < current_oos_end


class TestEvaluateParameters:
    """Test cases for _evaluate_parameters function."""

    def test_evaluate_parameters_success(self):
        """Test successful parameter evaluation."""
        # Create mock data
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        price_df = pd.DataFrame(
            {
                "Close": [100 + i * 2 for i in range(20)],
                "Open": [99 + i * 2 for i in range(20)],
                "High": [101 + i * 2 for i in range(20)],
                "Low": [98 + i * 2 for i in range(20)],
                "Volume": [1000 + i * 100 for i in range(20)],
            },
            index=dates,
        )

        oos_start = pd.Timestamp("2023-01-10")
        oos_end = pd.Timestamp("2023-01-20")

        equity_curve = _evaluate_parameters(
            price_df, oos_start, oos_end, 5, 10, 100000, 1.0, 0.5
        )

        assert equity_curve is not None
        assert isinstance(equity_curve, pd.Series)
        assert not equity_curve.empty
        assert equity_curve.iloc[0] == 100000  # Initial cash

    def test_evaluate_parameters_empty_data(self):
        """Test parameter evaluation with empty data."""
        price_df = pd.DataFrame()
        oos_start = pd.Timestamp("2023-01-10")
        oos_end = pd.Timestamp("2023-01-20")

        equity_curve = _evaluate_parameters(
            price_df, oos_start, oos_end, 5, 10, 100000, 1.0, 0.5
        )

        assert equity_curve is None

    def test_evaluate_parameters_no_oos_data(self):
        """Test parameter evaluation with no OOS data."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            {
                "Close": [100 + i for i in range(5)],
                "Open": [99 + i for i in range(5)],
                "High": [101 + i for i in range(5)],
                "Low": [98 + i for i in range(5)],
                "Volume": [1000 + i * 100 for i in range(5)],
            },
            index=dates,
        )

        oos_start = pd.Timestamp("2023-01-10")  # After data ends
        oos_end = pd.Timestamp("2023-01-20")

        equity_curve = _evaluate_parameters(
            price_df, oos_start, oos_end, 5, 10, 100000, 1.0, 0.5
        )

        assert equity_curve is None


class TestConcatenateEquityCurves:
    """Test cases for _concatenate_equity_curves function."""

    def test_concatenate_equity_curves_basic(self):
        """Test basic equity curve concatenation."""
        dates1 = pd.date_range(start="2023-01-01", periods=5, freq="D")
        dates2 = pd.date_range(start="2023-01-06", periods=5, freq="D")

        curve1 = pd.Series([100000, 101000, 102000, 101500, 103000], index=dates1)
        curve2 = pd.Series([103000, 104000, 103500, 104500, 105000], index=dates2)

        result = _concatenate_equity_curves([curve1, curve2], 100000)

        assert isinstance(result, pd.Series)
        assert len(result) == 9  # 5 + 5 - 1 (overlapping point)
        assert result.iloc[0] == 100000  # Initial cash
        assert result.iloc[-1] == 105000  # Final value

    def test_concatenate_equity_curves_empty_list(self):
        """Test concatenation with empty list."""
        result = _concatenate_equity_curves([], 100000)

        assert isinstance(result, pd.Series)
        assert len(result) == 1
        assert result.iloc[0] == 100000

    def test_concatenate_equity_curves_single_curve(self):
        """Test concatenation with single curve."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        curve = pd.Series([100000, 101000, 102000, 101500, 103000], index=dates)

        result = _concatenate_equity_curves([curve], 100000)

        assert isinstance(result, pd.Series)
        assert len(result) == 5
        assert result.iloc[0] == 100000
        assert result.iloc[-1] == 103000


class TestCountTrades:
    """Test cases for _count_trades function."""

    def test_count_trades_basic(self):
        """Test basic trade counting."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        price_df = pd.DataFrame(
            {
                "Close": [100 + i for i in range(10)],
                "Open": [99 + i for i in range(10)],
                "High": [101 + i for i in range(10)],
                "Low": [98 + i for i in range(10)],
                "Volume": [1000 + i * 100 for i in range(10)],
            },
            index=dates,
        )

        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-01-10")

        trades = _count_trades(price_df, start_date, end_date, 3, 5)

        assert isinstance(trades, int)
        assert trades >= 0

    def test_count_trades_empty_data(self):
        """Test trade counting with empty data."""
        price_df = pd.DataFrame()
        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-01-10")

        trades = _count_trades(price_df, start_date, end_date, 3, 5)

        assert trades == 0


class TestCalculateParameterStability:
    """Test cases for _calculate_parameter_stability function."""

    def test_calculate_parameter_stability_basic(self):
        """Test basic parameter stability calculation."""
        window_results = [
            {"best_fast": 10, "best_slow": 20},
            {"best_fast": 12, "best_slow": 22},
            {"best_fast": 8, "best_slow": 18},
            {"best_fast": 11, "best_slow": 21},
        ]

        stability = _calculate_parameter_stability(window_results)

        assert "fast_std" in stability
        assert "slow_std" in stability
        assert "fast_cv" in stability
        assert "slow_cv" in stability

        assert stability["fast_std"] > 0
        assert stability["slow_std"] > 0
        assert stability["fast_cv"] > 0
        assert stability["slow_cv"] > 0

    def test_calculate_parameter_stability_single_window(self):
        """Test parameter stability with single window."""
        window_results = [{"best_fast": 10, "best_slow": 20}]

        stability = _calculate_parameter_stability(window_results)

        assert stability["fast_std"] == 0
        assert stability["slow_std"] == 0
        assert stability["fast_cv"] == 0
        assert stability["slow_cv"] == 0


class TestCalculateWalkforwardSummary:
    """Test cases for _calculate_walkforward_summary function."""

    def test_calculate_walkforward_summary_basic(self):
        """Test basic walkforward summary calculation."""
        window_results = [
            {
                "oos_metrics": {"sharpe": 1.5, "cagr": 0.1, "max_dd": 0.05},
                "is_sharpe": 1.8,
            },
            {
                "oos_metrics": {"sharpe": 1.2, "cagr": 0.08, "max_dd": 0.08},
                "is_sharpe": 1.6,
            },
            {
                "oos_metrics": {"sharpe": 1.8, "cagr": 0.12, "max_dd": 0.03},
                "is_sharpe": 2.0,
            },
        ]

        summary = _calculate_walkforward_summary(window_results)

        assert "num_windows" in summary
        assert "oos_sharpe_mean" in summary
        assert "oos_sharpe_std" in summary
        assert "parameter_stability" in summary

        assert summary["num_windows"] == 3
        assert summary["oos_sharpe_mean"] > 0
        assert summary["oos_sharpe_std"] > 0

    def test_calculate_walkforward_summary_empty(self):
        """Test walkforward summary with empty results."""
        summary = _calculate_walkforward_summary([])

        assert summary == {}


class TestPrintWalkforwardResults:
    """Test cases for print_walkforward_results function."""

    def test_print_walkforward_results_basic(self):
        """Test basic results printing."""
        results = {
            "metrics": {
                "total_return": 0.15,
                "cagr": 0.12,
                "sharpe": 1.5,
                "max_drawdown": 0.05,
                "calmar": 2.4,
                "volatility": 0.08,
                "hit_rate": 0.6,
                "sortino": 1.8,
                "var_95": 0.02,
                "cvar_95": 0.03,
                "num_windows": 5,
                "total_trades": 25,
            },
            "summary": {
                "oos_sharpe_mean": 1.4,
                "oos_sharpe_std": 0.2,
                "parameter_stability": {"fast_cv": 0.1, "slow_cv": 0.15},
            },
            "windows": [
                {
                    "window": 1,
                    "is_start": pd.Timestamp("2020-01-01"),
                    "is_end": pd.Timestamp("2022-01-01"),
                    "oos_start": pd.Timestamp("2022-01-01"),
                    "oos_end": pd.Timestamp("2023-01-01"),
                    "best_fast": 10,
                    "best_slow": 20,
                    "is_sharpe": 1.8,
                    "oos_metrics": {"sharpe": 1.5, "cagr": 0.1},
                }
            ],
        }

        # Should not raise any exceptions
        print_walkforward_results(results)


class TestWalkforwardCrossover:
    """Test cases for walkforward_crossover function."""

    def test_walkforward_crossover_basic(self):
        """Test basic walkforward crossover analysis."""
        # Create mock data
        dates = pd.date_range(
            start="2020-01-01", periods=1095, freq="D"
        )  # 3 years of data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1095)
        prices = 100 * np.cumprod(1 + returns)

        price_df = pd.DataFrame(
            {
                "Close": prices,
                "Open": prices * 0.999,
                "High": prices * 1.005,
                "Low": prices * 0.995,
                "Volume": np.random.randint(1000, 5000, 1095),
            },
            index=dates,
        )

        with (
            patch("qbacktester.data.DataLoader.get_price_data") as mock_get_price_data,
            patch("qbacktester.optimize.grid_search") as mock_grid_search,
            patch("qbacktester.walkforward._evaluate_parameters") as mock_evaluate,
            patch("qbacktester.run.run_crossover_backtest") as mock_run_backtest,
        ):
            mock_get_price_data.return_value = price_df

            # Mock grid search to return valid results
            mock_grid_results = pd.DataFrame(
                {
                    "fast": [5, 10],
                    "slow": [15, 20],
                    "sharpe": [1.5, 1.2],
                    "cagr": [0.1, 0.08],
                    "max_dd": [0.05, 0.08],
                    "calmar": [2.0, 1.0],
                    "volatility": [0.1, 0.12],
                    "hit_rate": [0.6, 0.55],
                    "num_trades": [10, 8],
                }
            )
            mock_grid_search.return_value = mock_grid_results

            # Mock evaluate parameters to return a valid equity curve
            mock_equity_curve = pd.Series(
                [100000, 101000, 102000, 103000],
                index=pd.date_range("2020-06-01", periods=4, freq="D"),
            )
            mock_evaluate.return_value = mock_equity_curve

            # Mock run_crossover_backtest to return valid results
            mock_backtest_result = {
                "equity_curve": mock_equity_curve,
                "metrics": {
                    "sharpe_ratio": 1.5,
                    "cagr": 0.1,
                    "max_drawdown": (
                        0.05,
                        pd.Timestamp("2020-06-01"),
                        pd.Timestamp("2020-06-02"),
                    ),
                    "calmar_ratio": 2.0,
                    "volatility": 0.1,
                    "hit_rate": 0.6,
                    "num_trades": 10,
                },
            }
            mock_run_backtest.return_value = mock_backtest_result

            results = walkforward_crossover(
                symbol="TEST",
                start="2020-01-01",
                end="2022-12-31",
                in_sample_years=1,
                out_sample_years=1,
                fast_grid=[5, 10],
                slow_grid=[15, 20],
                verbose=False,
            )

            # Verify results structure
            assert isinstance(results, dict)
            assert "windows" in results
            assert "equity_curve" in results
            assert "metrics" in results
            assert "summary" in results

            # Verify windows
            assert len(results["windows"]) > 0
            for window in results["windows"]:
                assert "window" in window
                assert "is_start" in window
                assert "is_end" in window
                assert "oos_start" in window
                assert "oos_end" in window
                assert "best_fast" in window
                assert "best_slow" in window
                assert "is_sharpe" in window
                assert "oos_metrics" in window

            # Verify equity curve
            assert isinstance(results["equity_curve"], pd.Series)
            assert not results["equity_curve"].empty

            # Verify metrics
            metrics = results["metrics"]
            assert "total_return" in metrics
            assert "cagr" in metrics
            assert "sharpe" in metrics
            assert "max_drawdown" in metrics
            assert "num_windows" in metrics

    def test_walkforward_crossover_empty_data(self):
        """Test walkforward crossover with empty data."""
        with patch("qbacktester.data.DataLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_price_data.return_value = pd.DataFrame()
            mock_loader_class.return_value = mock_loader

            with pytest.raises(ValueError, match="No price data loaded"):
                walkforward_crossover(
                    symbol="TEST",
                    start="2020-01-01",
                    end="2020-12-31",
                    in_sample_years=1,
                    out_sample_years=1,
                    verbose=False,
                )

    def test_walkforward_crossover_short_period(self):
        """Test walkforward crossover with short period."""
        dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
        price_df = pd.DataFrame(
            {
                "Close": [100 + i for i in range(10)],
                "Open": [99 + i for i in range(10)],
                "High": [101 + i for i in range(10)],
                "Low": [98 + i for i in range(10)],
                "Volume": [1000 + i * 100 for i in range(10)],
            },
            index=dates,
        )

        with patch("qbacktester.data.DataLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_price_data.return_value = price_df
            mock_loader_class.return_value = mock_loader

            with pytest.raises(ValueError, match="No valid walk-forward windows"):
                walkforward_crossover(
                    symbol="TEST",
                    start="2020-01-01",
                    end="2020-01-10",
                    in_sample_years=2,
                    out_sample_years=1,
                    verbose=False,
                )

    def test_walkforward_crossover_different_metrics(self):
        """Test walkforward crossover with different optimization metrics."""
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        prices = 100 * np.cumprod(1 + returns)

        price_df = pd.DataFrame(
            {
                "Close": prices,
                "Open": prices * 0.999,
                "High": prices * 1.005,
                "Low": prices * 0.995,
                "Volume": np.random.randint(1000, 5000, 100),
            },
            index=dates,
        )

        with patch("qbacktester.data.DataLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_price_data.return_value = price_df
            mock_loader_class.return_value = mock_loader

            # Test different metrics
            for metric in ["sharpe", "cagr", "calmar", "max_dd"]:
                results = walkforward_crossover(
                    symbol="TEST",
                    start="2020-01-01",
                    end="2020-12-31",
                    in_sample_years=1,
                    out_sample_years=1,
                    fast_grid=[5],
                    slow_grid=[10],
                    optimization_metric=metric,
                    verbose=False,
                )

                assert isinstance(results, dict)
                assert "metrics" in results
                assert "equity_curve" in results


class TestWalkforwardIntegration:
    """Integration tests for walkforward functionality."""

    def test_walkforward_integration_complete(self):
        """Test complete walkforward integration."""
        dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 200)
        prices = 100 * np.cumprod(1 + returns)

        price_df = pd.DataFrame(
            {
                "Close": prices,
                "Open": prices * 0.999,
                "High": prices * 1.005,
                "Low": prices * 0.995,
                "Volume": np.random.randint(1000, 5000, 200),
            },
            index=dates,
        )

        with patch("qbacktester.data.DataLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_price_data.return_value = price_df
            mock_loader_class.return_value = mock_loader

            # Run walkforward analysis
            results = walkforward_crossover(
                symbol="INTEGRATION_TEST",
                start="2020-01-01",
                end="2021-12-31",
                in_sample_years=1,
                out_sample_years=1,
                fast_grid=[5, 10],
                slow_grid=[15, 20],
                verbose=False,
            )

            # Verify complete results
            assert isinstance(results, dict)
            assert len(results["windows"]) > 0
            assert not results["equity_curve"].empty

            # Verify metrics are reasonable
            metrics = results["metrics"]
            assert metrics["num_windows"] > 0
            assert metrics["total_trades"] >= 0

            # Verify summary statistics
            summary = results["summary"]
            assert "num_windows" in summary
            assert "oos_sharpe_mean" in summary
            assert "parameter_stability" in summary

    def test_walkforward_parameter_validation(self):
        """Test walkforward parameter validation."""
        with pytest.raises(ValueError, match="in_sample_years must be positive"):
            walkforward_crossover(
                symbol="TEST",
                start="2020-01-01",
                end="2020-12-31",
                in_sample_years=0,
                out_sample_years=1,
            )

        with pytest.raises(ValueError, match="out_sample_years must be positive"):
            walkforward_crossover(
                symbol="TEST",
                start="2020-01-01",
                end="2020-12-31",
                in_sample_years=1,
                out_sample_years=0,
            )

    def test_walkforward_error_handling(self):
        """Test walkforward error handling."""
        with patch("qbacktester.data.DataLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_price_data.side_effect = Exception("Data loading error")
            mock_loader_class.return_value = mock_loader

            with pytest.raises(Exception, match="Data loading error"):
                walkforward_crossover(
                    symbol="TEST",
                    start="2020-01-01",
                    end="2020-12-31",
                    in_sample_years=1,
                    out_sample_years=1,
                    verbose=False,
                )
