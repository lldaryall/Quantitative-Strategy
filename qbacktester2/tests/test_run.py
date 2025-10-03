"""Tests for the run module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from qbacktester.run import (
    _extract_trades,
    print_backtest_report,
    run_crossover_backtest,
    run_quick_backtest,
)
from qbacktester.strategy import StrategyParams


class TestRunCrossoverBacktest:
    """Test cases for run_crossover_backtest function."""

    def test_run_crossover_backtest_success(self):
        """Test successful backtest execution with mock data."""
        # Create mock data
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        mock_price_df = pd.DataFrame(
            {
                "Close": [100 + i * 0.5 + np.sin(i * 0.3) * 2 for i in range(20)],
                "Open": [100 + i * 0.5 + np.sin(i * 0.3) * 2 - 0.5 for i in range(20)],
                "High": [100 + i * 0.5 + np.sin(i * 0.3) * 2 + 1 for i in range(20)],
                "Low": [100 + i * 0.5 + np.sin(i * 0.3) * 2 - 1 for i in range(20)],
                "Volume": [1000 + i * 100 for i in range(20)],
            },
            index=dates,
        )

        # Create strategy parameters
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-20",
            fast_window=3,
            slow_window=7,
            initial_cash=100_000,
            fee_bps=5,
            slippage_bps=2,
        )

        # Mock the DataLoader
        with patch("qbacktester.run.DataLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_price_data.return_value = mock_price_df
            mock_loader_class.return_value = mock_loader

            # Run the backtest
            results = run_crossover_backtest(params)

            # Verify results structure
            assert isinstance(results, dict)
            assert "params" in results
            assert "equity_curve" in results
            assert "metrics" in results
            assert "trades" in results

            # Verify params
            assert results["params"] == params

            # Verify equity curve
            equity_curve = results["equity_curve"]
            assert isinstance(equity_curve, pd.Series)
            assert len(equity_curve) == 20
            assert equity_curve.iloc[0] == params.initial_cash

            # Verify metrics
            metrics = results["metrics"]
            expected_metrics = [
                "cagr",
                "sharpe_ratio",
                "max_drawdown",
                "calmar_ratio",
                "hit_rate",
                "volatility",
                "sortino_ratio",
                "avg_win",
                "avg_loss",
                "var_5pct",
                "cvar_5pct",
                "num_trades",
                "total_transaction_costs",
                "initial_capital",
                "final_equity",
                "total_return",
                "max_dd_peak_date",
                "max_dd_trough_date",
            ]
            for metric in expected_metrics:
                assert metric in metrics

            # Verify trades
            trades = results["trades"]
            assert isinstance(trades, pd.DataFrame)
            expected_trade_columns = [
                "timestamp",
                "side",
                "price",
                "quantity",
                "notional",
                "cost",
            ]
            for col in expected_trade_columns:
                assert col in trades.columns

    def test_run_crossover_backtest_empty_data(self):
        """Test backtest with empty data."""
        params = StrategyParams(
            symbol="INVALID",
            start="2023-01-01",
            end="2023-01-20",
            fast_window=3,
            slow_window=7,
        )

        # Mock DataLoader to return empty data
        with patch("qbacktester.run.DataLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_price_data.return_value = pd.DataFrame()
            mock_loader_class.return_value = mock_loader

            with pytest.raises(ValueError, match="No data found for symbol INVALID"):
                run_crossover_backtest(params)

    def test_run_crossover_backtest_data_loading_error(self):
        """Test backtest with data loading error."""
        params = StrategyParams(
            symbol="ERROR",
            start="2023-01-01",
            end="2023-01-20",
            fast_window=3,
            slow_window=7,
        )

        # Mock DataLoader to raise exception
        with patch("qbacktester.run.DataLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_price_data.side_effect = Exception("Network error")
            mock_loader_class.return_value = mock_loader

            with pytest.raises(Exception, match="Network error"):
                run_crossover_backtest(params)

    def test_run_crossover_backtest_short_period(self):
        """Test backtest with very short period (smoke test)."""
        # Create minimal data
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        mock_price_df = pd.DataFrame(
            {
                "Close": [100, 101, 102, 101, 103],
                "Open": [99.5, 100.5, 101.5, 100.5, 102.5],
                "High": [100.5, 101.5, 102.5, 101.5, 103.5],
                "Low": [99, 100, 101, 100, 102],
                "Volume": [1000, 1100, 1200, 1100, 1300],
            },
            index=dates,
        )

        params = StrategyParams(
            symbol="SHORT",
            start="2023-01-01",
            end="2023-01-05",
            fast_window=2,
            slow_window=3,
            initial_cash=50_000,
        )

        with patch("qbacktester.run.DataLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_price_data.return_value = mock_price_df
            mock_loader_class.return_value = mock_loader

            # This should complete without error (smoke test)
            results = run_crossover_backtest(params)

            # Basic verification
            assert isinstance(results, dict)
            assert len(results["equity_curve"]) == 5
            assert results["metrics"]["initial_capital"] == 50_000


class TestExtractTrades:
    """Test cases for _extract_trades function."""

    def test_extract_trades_basic(self):
        """Test basic trade extraction."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")

        # Create mock result DataFrame
        result = pd.DataFrame(
            {
                "trade_flag": [False, True, False, True, False],
                "position": [0, 1, 1, 0, 0],
                "trade_price": [100, 101, 102, 103, 104],
                "notional": [0, 100000, 0, 100000, 0],
                "transaction_cost": [0, 150, 0, 150, 0],
            },
            index=dates,
        )

        price_df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates)

        trades = _extract_trades(result, price_df)

        assert len(trades) == 2
        assert trades.iloc[0]["side"] == "BUY"
        assert trades.iloc[1]["side"] == "SELL"
        assert trades.iloc[0]["price"] == 101
        assert trades.iloc[1]["price"] == 103

    def test_extract_trades_no_trades(self):
        """Test trade extraction with no trades."""
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")

        result = pd.DataFrame(
            {
                "trade_flag": [False, False, False],
                "position": [0, 0, 0],
                "trade_price": [100, 101, 102],
                "notional": [0, 0, 0],
                "transaction_cost": [0, 0, 0],
            },
            index=dates,
        )

        price_df = pd.DataFrame({"Close": [100, 101, 102]}, index=dates)

        trades = _extract_trades(result, price_df)

        assert len(trades) == 0
        assert list(trades.columns) == [
            "timestamp",
            "side",
            "price",
            "quantity",
            "notional",
            "cost",
        ]


class TestRunQuickBacktest:
    """Test cases for run_quick_backtest function."""

    def test_run_quick_backtest_defaults(self):
        """Test run_quick_backtest with default parameters."""
        # Mock the run_crossover_backtest function
        with patch("qbacktester.run.run_crossover_backtest") as mock_run:
            mock_results = {
                "params": Mock(),
                "equity_curve": pd.Series([100000, 101000, 102000]),
                "metrics": {"cagr": 0.1, "sharpe_ratio": 1.0},
                "trades": pd.DataFrame(),
            }
            mock_run.return_value = mock_results

            results = run_quick_backtest()

            # Verify that run_crossover_backtest was called
            mock_run.assert_called_once()

            # Verify the parameters passed
            called_params = mock_run.call_args[0][0]
            assert called_params.symbol == "AAPL"
            assert called_params.start == "2023-01-01"
            assert called_params.end == "2023-12-31"
            assert called_params.fast_window == 5
            assert called_params.slow_window == 20
            assert called_params.initial_cash == 100_000

    def test_run_quick_backtest_custom_params(self):
        """Test run_quick_backtest with custom parameters."""
        with patch("qbacktester.run.run_crossover_backtest") as mock_run:
            mock_results = {
                "params": Mock(),
                "equity_curve": pd.Series(),
                "metrics": {},
                "trades": pd.DataFrame(),
            }
            mock_run.return_value = mock_results

            results = run_quick_backtest(
                symbol="MSFT",
                start="2022-01-01",
                end="2022-12-31",
                fast_window=10,
                slow_window=30,
                initial_cash=200_000,
                fee_bps=2.0,
                slippage_bps=1.0,
            )

            # Verify the parameters passed
            called_params = mock_run.call_args[0][0]
            assert called_params.symbol == "MSFT"
            assert called_params.start == "2022-01-01"
            assert called_params.end == "2022-12-31"
            assert called_params.fast_window == 10
            assert called_params.slow_window == 30
            assert called_params.initial_cash == 200_000
            assert called_params.fee_bps == 2.0
            assert called_params.slippage_bps == 1.0


class TestPrintBacktestReport:
    """Test cases for print_backtest_report function."""

    def test_print_backtest_report_basic(self):
        """Test basic report printing."""
        # Create mock results
        params = StrategyParams(
            symbol="TEST",
            start="2023-01-01",
            end="2023-01-20",
            fast_window=5,
            slow_window=10,
            initial_cash=100_000,
        )

        equity_curve = pd.Series(
            [100000, 101000, 102000, 101500, 103000],
            index=pd.date_range("2023-01-01", periods=5, freq="D"),
        )

        metrics = {
            "cagr": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.05,
            "calmar_ratio": 3.0,
            "hit_rate": 0.6,
            "volatility": 0.2,
            "sortino_ratio": 1.5,
            "avg_win": 0.02,
            "avg_loss": 0.015,
            "var_5pct": 0.03,
            "cvar_5pct": 0.04,
            "num_trades": 4,
            "total_transaction_costs": 200,
            "initial_capital": 100_000,
            "final_equity": 103_000,
            "total_return": 0.03,
            "max_dd_peak_date": pd.Timestamp("2023-01-03"),
            "max_dd_trough_date": pd.Timestamp("2023-01-04"),
        }

        trades = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=2, freq="D"),
                "side": ["BUY", "SELL"],
                "price": [100, 102],
                "quantity": [1000, 1000],
                "notional": [100000, 102000],
                "cost": [100, 100],
            }
        )

        results = {
            "params": params,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "trades": trades,
        }

        # This should not raise an exception
        print_backtest_report(results)

    def test_print_backtest_report_no_trades(self):
        """Test report printing with no trades."""
        params = StrategyParams("TEST", "2023-01-01", "2023-01-20", 5, 10)
        equity_curve = pd.Series(
            [100000, 100000, 100000],
            index=pd.date_range("2023-01-01", periods=3, freq="D"),
        )

        metrics = {
            "cagr": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": float("inf"),
            "hit_rate": 0.0,
            "volatility": 0.0,
            "sortino_ratio": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "var_5pct": 0.0,
            "cvar_5pct": 0.0,
            "num_trades": 0,
            "total_transaction_costs": 0,
            "initial_capital": 100_000,
            "final_equity": 100_000,
            "total_return": 0.0,
            "max_dd_peak_date": pd.Timestamp("2023-01-01"),
            "max_dd_trough_date": pd.Timestamp("2023-01-01"),
        }

        results = {
            "params": params,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "trades": pd.DataFrame(),
        }

        # This should not raise an exception
        print_backtest_report(results)


class TestIntegrationSmokeTest:
    """Integration smoke tests for the complete pipeline."""

    def test_complete_pipeline_smoke_test(self):
        """Test the complete pipeline with mock data (smoke test)."""
        # Create realistic mock data
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        np.random.seed(42)

        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, 30)  # 0.1% daily return, 2% volatility
        prices = 100 * np.cumprod(1 + returns)

        mock_price_df = pd.DataFrame(
            {
                "Close": prices,
                "Open": prices * 0.999,  # Open slightly lower
                "High": prices * 1.005,  # High slightly higher
                "Low": prices * 0.995,  # Low slightly lower
                "Volume": np.random.randint(1000, 5000, 30),
            },
            index=dates,
        )

        params = StrategyParams(
            symbol="SMOKE_TEST",
            start="2023-01-01",
            end="2023-01-30",
            fast_window=5,
            slow_window=15,
            initial_cash=100_000,
            fee_bps=1.0,
            slippage_bps=0.5,
        )

        # Mock DataLoader
        with patch("qbacktester.run.DataLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_price_data.return_value = mock_price_df
            mock_loader_class.return_value = mock_loader

            # Run the complete pipeline
            results = run_crossover_backtest(params)

            # Verify the pipeline completed successfully
            assert isinstance(results, dict)
            assert "params" in results
            assert "equity_curve" in results
            assert "metrics" in results
            assert "trades" in results

            # Verify equity curve
            equity_curve = results["equity_curve"]
            assert len(equity_curve) == 30
            assert equity_curve.iloc[0] == 100_000
            assert equity_curve.iloc[-1] > 0  # Should have some value

            # Verify metrics are reasonable
            metrics = results["metrics"]
            assert metrics["initial_capital"] == 100_000
            assert metrics["final_equity"] > 0
            assert 0 <= metrics["hit_rate"] <= 1
            assert metrics["num_trades"] >= 0
            assert metrics["total_transaction_costs"] >= 0

            # Verify trades DataFrame structure
            trades = results["trades"]
            assert isinstance(trades, pd.DataFrame)
            if not trades.empty:
                assert "timestamp" in trades.columns
                assert "side" in trades.columns
                assert "price" in trades.columns
                assert "quantity" in trades.columns
                assert "notional" in trades.columns
                assert "cost" in trades.columns

                # Verify trade sides are valid
                assert all(side in ["BUY", "SELL"] for side in trades["side"])

    def test_pipeline_with_different_parameters(self):
        """Test pipeline with different parameter combinations."""
        # Create mock data
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        mock_price_df = pd.DataFrame(
            {
                "Close": [100, 101, 102, 101, 103, 104, 103, 105, 106, 107],
                "Open": [
                    99.5,
                    100.5,
                    101.5,
                    100.5,
                    102.5,
                    103.5,
                    102.5,
                    104.5,
                    105.5,
                    106.5,
                ],
                "High": [
                    100.5,
                    101.5,
                    102.5,
                    101.5,
                    103.5,
                    104.5,
                    103.5,
                    105.5,
                    106.5,
                    107.5,
                ],
                "Low": [99, 100, 101, 100, 102, 103, 102, 104, 105, 106],
                "Volume": [1000 + i * 100 for i in range(10)],
            },
            index=dates,
        )

        # Test different parameter combinations
        test_cases = [
            {"fast_window": 2, "slow_window": 4, "initial_cash": 50_000},
            {"fast_window": 3, "slow_window": 6, "initial_cash": 200_000},
            {"fast_window": 1, "slow_window": 2, "initial_cash": 75_000},
        ]

        for case in test_cases:
            params = StrategyParams(
                symbol="PARAM_TEST", start="2023-01-01", end="2023-01-10", **case
            )

            with patch("qbacktester.run.DataLoader") as mock_loader_class:
                mock_loader = Mock()
                mock_loader.get_price_data.return_value = mock_price_df
                mock_loader_class.return_value = mock_loader

                # Each case should complete successfully
                results = run_crossover_backtest(params)

                assert results["params"].initial_cash == case["initial_cash"]
                assert results["params"].fast_window == case["fast_window"]
                assert results["params"].slow_window == case["slow_window"]
                assert len(results["equity_curve"]) == 10
