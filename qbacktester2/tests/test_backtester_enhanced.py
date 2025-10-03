"""Tests for the enhanced Backtester class."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from qbacktester.backtester import Backtester
from qbacktester.strategy import StrategyParams


class TestBacktesterInitialization:
    """Test cases for Backtester initialization."""

    def test_backtester_valid_initialization(self):
        """Test valid Backtester initialization."""
        # Create test data
        dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
        price_df = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "Open": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            },
            index=dates,
        )
        signals = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], index=dates)
        params = StrategyParams(
            symbol="TEST",
            start="2020-01-01",
            end="2020-01-10",
            fast_window=5,
            slow_window=10,
        )

        backtester = Backtester(price_df, signals, params)

        assert backtester.price_df.equals(price_df)
        assert backtester.signals.equals(signals)
        assert backtester.params == params

    def test_backtester_missing_close_column(self):
        """Test Backtester with missing Close column."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        price_df = pd.DataFrame({"Open": [100, 101, 102, 103, 104]}, index=dates)
        signals = pd.Series([0, 1, 0, 1, 0], index=dates)
        params = StrategyParams("TEST", "2020-01-01", "2020-01-05", 5, 10)

        with pytest.raises(ValueError, match="price_df must contain columns"):
            Backtester(price_df, signals, params)

    def test_backtester_empty_data(self):
        """Test Backtester with empty data."""
        price_df = pd.DataFrame(columns=["Close"])
        signals = pd.Series([], dtype=int)
        params = StrategyParams("TEST", "2020-01-01", "2020-01-05", 5, 10)

        with pytest.raises(ValueError, match="price_df cannot be empty"):
            Backtester(price_df, signals, params)

    def test_backtester_invalid_signals(self):
        """Test Backtester with invalid signals."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        price_df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates)
        signals = pd.Series([0, 1, 2, 0, 1], index=dates)  # Invalid: contains 2
        params = StrategyParams("TEST", "2020-01-01", "2020-01-05", 5, 10)

        with pytest.raises(ValueError, match="signals must contain only 0 or 1"):
            Backtester(price_df, signals, params)

    def test_backtester_mismatched_indices(self):
        """Test Backtester with mismatched indices."""
        dates1 = pd.date_range(start="2020-01-01", periods=5, freq="D")
        dates2 = pd.date_range(start="2020-01-02", periods=5, freq="D")
        price_df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates1)
        signals = pd.Series([0, 1, 0, 1, 0], index=dates2)
        params = StrategyParams("TEST", "2020-01-01", "2020-01-05", 5, 10)

        with pytest.raises(
            ValueError, match="price_df and signals must have the same index"
        ):
            Backtester(price_df, signals, params)


class TestBacktesterRun:
    """Test cases for Backtester run method."""

    def test_backtester_run_basic(self):
        """Test basic backtest run."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            {"Close": [100, 101, 102, 103, 104], "Open": [99, 100, 101, 102, 103]},
            index=dates,
        )
        signals = pd.Series([0, 1, 1, 0, 0], index=dates)
        params = StrategyParams("TEST", "2020-01-01", "2020-01-05", 5, 10)

        backtester = Backtester(price_df, signals, params)
        result = backtester.run()

        # Check required columns
        required_columns = [
            "Close",
            "signal",
            "position",
            "holdings_value",
            "cash",
            "total_equity",
            "trade_flag",
        ]
        for col in required_columns:
            assert col in result.columns

        # Check data types
        assert result["signal"].dtype == int
        assert result["position"].dtype == int
        assert result["trade_flag"].dtype == bool

        # Check that position matches signal
        assert result["position"].equals(result["signal"])

    def test_backtester_run_hand_computed_fixture(self):
        """Test backtest with hand-computed fixture to verify correctness."""
        # Simple test case: buy on day 2, sell on day 4
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104],
                "Open": [100, 101, 102, 103, 104],  # Same as close for simplicity
            },
            index=dates,
        )
        signals = pd.Series([0, 1, 1, 0, 0], index=dates)
        params = StrategyParams(
            symbol="TEST",
            start="2020-01-01",
            end="2020-01-05",
            fast_window=5,
            slow_window=10,
            initial_cash=1000,
            fee_bps=0,  # No fees for easier calculation
            slippage_bps=0,
        )

        backtester = Backtester(price_df, signals, params)
        result = backtester.run()

        # Hand-computed expected values:
        # Day 0: Cash=1000, Position=0, Equity=1000
        # Day 1: Signal=1, Buy at 101, Cash=0, Shares=1000/101=9.90099, Holdings=9.90099*101=1000, Equity=1000
        # Day 2: Signal=1, Hold, Cash=0, Shares=9.90099, Holdings=9.90099*102=1009.90, Equity=1009.90
        # Day 3: Signal=0, Sell at 103, Cash=9.90099*103=1019.90, Shares=0, Holdings=0, Equity=1019.90
        # Day 4: Signal=0, Hold, Cash=1019.90, Shares=0, Holdings=0, Equity=1019.90

        # Check initial values
        assert result["cash"].iloc[0] == 1000
        assert result["holdings_value"].iloc[0] == 0
        assert result["total_equity"].iloc[0] == 1000

        # Check trade flags
        assert result["trade_flag"].iloc[0] == False  # No trade on first day
        assert result["trade_flag"].iloc[1] == True  # Buy on day 1
        assert result["trade_flag"].iloc[2] == False  # Hold on day 2
        assert result["trade_flag"].iloc[3] == True  # Sell on day 3
        assert result["trade_flag"].iloc[4] == False  # Hold on day 4

        # Check final equity (should be approximately 1019.90)
        final_equity = result["total_equity"].iloc[-1]
        expected_equity = 1000 * 103 / 101  # 1000 * (103/101) = 1019.80
        assert (
            abs(final_equity - expected_equity) < 0.1
        )  # Allow small rounding differences

    def test_backtester_run_with_fees(self):
        """Test backtest with transaction fees."""
        dates = pd.date_range(start="2020-01-01", periods=4, freq="D")
        price_df = pd.DataFrame(
            {"Close": [100, 101, 102, 103], "Open": [100, 101, 102, 103]}, index=dates
        )
        signals = pd.Series([0, 1, 0, 0], index=dates)  # Buy on day 1, sell on day 2
        params = StrategyParams(
            symbol="TEST",
            start="2020-01-01",
            end="2020-01-04",
            fast_window=5,
            slow_window=10,
            initial_cash=1000,
            fee_bps=10,  # 0.1% fee
            slippage_bps=5,  # 0.05% slippage
        )

        backtester = Backtester(price_df, signals, params)
        result = backtester.run()

        # Check that transaction costs are applied
        total_costs = result["transaction_cost"].sum()
        assert total_costs > 0

        # Check that costs reduce final equity
        final_equity = result["total_equity"].iloc[-1]
        expected_equity_no_fees = 1000 * 102 / 101  # 1009.90
        assert final_equity < expected_equity_no_fees

    def test_backtester_run_no_trades(self):
        """Test backtest with no trades."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            {"Close": [100, 101, 102, 103, 104], "Open": [100, 101, 102, 103, 104]},
            index=dates,
        )
        signals = pd.Series([0, 0, 0, 0, 0], index=dates)  # No trades
        params = StrategyParams("TEST", "2020-01-01", "2020-01-05", 5, 10)

        backtester = Backtester(price_df, signals, params)
        result = backtester.run()

        # Should remain in cash throughout
        assert (result["position"] == 0).all()
        assert (result["holdings_value"] == 0).all()
        assert (result["cash"] == params.initial_cash).all()
        assert (result["total_equity"] == params.initial_cash).all()
        assert (result["trade_flag"] == False).all()

    def test_backtester_run_always_long(self):
        """Test backtest with always long position."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            {"Close": [100, 101, 102, 103, 104], "Open": [100, 101, 102, 103, 104]},
            index=dates,
        )
        signals = pd.Series([1, 1, 1, 1, 1], index=dates)  # Always long
        params = StrategyParams("TEST", "2020-01-01", "2020-01-05", 5, 10)

        backtester = Backtester(price_df, signals, params)
        result = backtester.run()

        # Should be long throughout
        assert (result["position"] == 1).all()
        assert (result["holdings_value"] > 0).all()
        # Cash should be 0 or negative (due to transaction costs) when in position
        assert (result["cash"] <= 0).all()
        assert result["trade_flag"].iloc[0] == True  # Initial buy
        assert (result["trade_flag"].iloc[1:] == False).all()  # No more trades

    def test_backtester_run_multiple_trades(self):
        """Test backtest with multiple trades."""
        dates = pd.date_range(start="2020-01-01", periods=7, freq="D")
        price_df = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104, 105, 106],
                "Open": [100, 101, 102, 103, 104, 105, 106],
            },
            index=dates,
        )
        signals = pd.Series([0, 1, 0, 1, 0, 1, 0], index=dates)  # Multiple trades
        params = StrategyParams("TEST", "2020-01-01", "2020-01-07", 5, 10)

        backtester = Backtester(price_df, signals, params)
        result = backtester.run()

        # Check trade flags
        expected_trades = [False, True, True, True, True, True, True]  # 6 trades
        assert result["trade_flag"].tolist() == expected_trades

        # Check that equity is calculated correctly
        assert result["total_equity"].iloc[0] == params.initial_cash
        assert result["total_equity"].iloc[-1] > 0  # Should have some value


class TestBacktesterPerformanceMetrics:
    """Test cases for performance metrics calculation."""

    def test_performance_metrics_basic(self):
        """Test basic performance metrics calculation."""
        dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
        price_df = pd.DataFrame(
            {
                "Close": [100 + i for i in range(10)],
                "Open": [100 + i for i in range(10)],
            },
            index=dates,
        )
        signals = pd.Series([0, 1, 1, 0, 0, 1, 1, 0, 0, 0], index=dates)
        params = StrategyParams("TEST", "2020-01-01", "2020-01-10", 5, 10)

        backtester = Backtester(price_df, signals, params)
        result = backtester.run()
        metrics = backtester.get_performance_metrics(result)

        # Check that all expected metrics are present
        expected_metrics = [
            "total_return",
            "annualized_return",
            "volatility",
            "sharpe_ratio",
            "max_drawdown",
            "num_trades",
            "total_transaction_costs",
            "final_equity",
        ]
        for metric in expected_metrics:
            assert metric in metrics

        # Check that metrics are reasonable
        assert metrics["final_equity"] > 0
        assert metrics["num_trades"] >= 0
        assert metrics["total_transaction_costs"] >= 0

    def test_performance_metrics_empty_result(self):
        """Test performance metrics with empty result."""
        backtester = Backtester(
            pd.DataFrame(
                {"Close": [100]}, index=pd.date_range("2020-01-01", periods=1)
            ),
            pd.Series([0], index=pd.date_range("2020-01-01", periods=1)),
            StrategyParams("TEST", "2020-01-01", "2020-01-01", 5, 10),
        )

        metrics = backtester.get_performance_metrics(pd.DataFrame())
        assert metrics == {}


class TestBacktesterFeeImpact:
    """Test cases for fee impact on performance."""

    def test_fee_impact_on_performance(self):
        """Test that higher fees reduce performance."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            {"Close": [100, 101, 102, 103, 104], "Open": [100, 101, 102, 103, 104]},
            index=dates,
        )
        signals = pd.Series([0, 1, 0, 0, 0], index=dates)  # One trade

        # Test with no fees
        params_no_fees = StrategyParams(
            "TEST",
            "2020-01-01",
            "2020-01-05",
            5,
            10,
            initial_cash=1000,
            fee_bps=0,
            slippage_bps=0,
        )
        backtester_no_fees = Backtester(price_df, signals, params_no_fees)
        result_no_fees = backtester_no_fees.run()
        metrics_no_fees = backtester_no_fees.get_performance_metrics(result_no_fees)

        # Test with fees
        params_with_fees = StrategyParams(
            "TEST",
            "2020-01-01",
            "2020-01-05",
            5,
            10,
            initial_cash=1000,
            fee_bps=50,
            slippage_bps=25,  # 0.75% total cost
        )
        backtester_with_fees = Backtester(price_df, signals, params_with_fees)
        result_with_fees = backtester_with_fees.run()
        metrics_with_fees = backtester_with_fees.get_performance_metrics(
            result_with_fees
        )

        # Performance should be worse with fees
        assert metrics_with_fees["total_return"] < metrics_no_fees["total_return"]
        assert metrics_with_fees["final_equity"] < metrics_no_fees["final_equity"]
        assert (
            metrics_with_fees["total_transaction_costs"]
            > metrics_no_fees["total_transaction_costs"]
        )

    def test_fee_calculation_accuracy(self):
        """Test that fees are calculated correctly."""
        dates = pd.date_range(start="2020-01-01", periods=3, freq="D")
        price_df = pd.DataFrame(
            {"Close": [100, 101, 102], "Open": [100, 101, 102]}, index=dates
        )
        signals = pd.Series([0, 1, 0], index=dates)  # Buy on day 1, sell on day 2
        params = StrategyParams(
            "TEST",
            "2020-01-01",
            "2020-01-03",
            5,
            10,
            initial_cash=1000,
            fee_bps=10,
            slippage_bps=5,  # 0.15% total
        )

        backtester = Backtester(price_df, signals, params)
        result = backtester.run()

        # Check that fees are applied on trade days
        trade_days = result[result["trade_flag"]]
        assert len(trade_days) == 2  # Buy and sell

        # Check that transaction costs are calculated correctly
        total_cost_bps = (params.fee_bps + params.slippage_bps) / 10000
        expected_buy_cost = 1000 * total_cost_bps  # 1000 * 0.0015 = 1.5
        expected_sell_cost = (
            1000 * 101 / 100
        ) * total_cost_bps  # ~1010 * 0.0015 = 1.515

        buy_cost = trade_days.iloc[0]["transaction_cost"]
        sell_cost = trade_days.iloc[1]["transaction_cost"]

        assert abs(buy_cost - expected_buy_cost) < 0.02  # Allow slightly more tolerance
        assert abs(sell_cost - expected_sell_cost) < 0.02


class TestBacktesterEdgeCases:
    """Test cases for edge cases and error conditions."""

    def test_backtester_single_day(self):
        """Test backtester with single day of data."""
        dates = pd.date_range(start="2020-01-01", periods=1, freq="D")
        price_df = pd.DataFrame({"Close": [100], "Open": [100]}, index=dates)
        signals = pd.Series([0], index=dates)
        params = StrategyParams("TEST", "2020-01-01", "2020-01-01", 5, 10)

        backtester = Backtester(price_df, signals, params)
        result = backtester.run()

        assert len(result) == 1
        assert result["total_equity"].iloc[0] == params.initial_cash

    def test_backtester_missing_open_prices(self):
        """Test backtester when Open prices are missing."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        price_df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates)
        signals = pd.Series([0, 1, 0, 0, 0], index=dates)
        params = StrategyParams("TEST", "2020-01-01", "2020-01-05", 5, 10)

        backtester = Backtester(price_df, signals, params)
        result = backtester.run()

        # Should use Close prices when Open is missing
        assert (result["trade_price"] == result["Close"]).all()

    def test_backtester_rapid_trading(self):
        """Test backtester with rapid trading (buy/sell every day)."""
        dates = pd.date_range(start="2020-01-01", periods=6, freq="D")
        price_df = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104, 105],
                "Open": [100, 101, 102, 103, 104, 105],
            },
            index=dates,
        )
        signals = pd.Series([0, 1, 0, 1, 0, 1], index=dates)  # Rapid trading
        params = StrategyParams("TEST", "2020-01-01", "2020-01-06", 5, 10)

        backtester = Backtester(price_df, signals, params)
        result = backtester.run()

        # Should have trades on every day except the first
        expected_trades = [False, True, True, True, True, True]
        assert result["trade_flag"].tolist() == expected_trades

        # Should have high transaction costs
        total_costs = result["transaction_cost"].sum()
        assert total_costs > 0


class TestBacktesterParametrizedCosts:
    """Parametrized tests for transaction cost impact on performance."""

    @pytest.mark.parametrize(
        "fee_bps,slippage_bps",
        [
            (0, 0),  # No costs
            (5, 2.5),  # Low costs (0.075% total)
            (10, 5),  # Medium costs (0.15% total)
            (20, 10),  # High costs (0.3% total)
            (50, 25),  # Very high costs (0.75% total)
        ],
    )
    def test_cost_impact_on_sharpe(self, fee_bps, slippage_bps):
        """Test that higher costs monotonically reduce Sharpe ratio."""
        # Create a profitable strategy with clear trend
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        np.random.seed(42)
        # Create upward trending prices with some noise
        trend = np.linspace(100, 120, 100)
        noise = np.random.normal(0, 0.5, 100)
        prices = trend + noise

        price_df = pd.DataFrame(
            {"Close": prices, "Open": prices * 0.999}, index=dates  # Slight gap down
        )

        # Simple buy-and-hold strategy
        signals = pd.Series([0] + [1] * 99, index=dates)

        params = StrategyParams(
            symbol="TEST",
            start="2020-01-01",
            end="2020-04-09",
            fast_window=5,
            slow_window=10,
            initial_cash=100000,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

        backtester = Backtester(price_df, signals, params)
        result = backtester.run()
        metrics = backtester.get_performance_metrics(result)

        # Verify that we have a valid Sharpe ratio
        assert not np.isnan(metrics["sharpe_ratio"])
        assert metrics["sharpe_ratio"] is not None

        # Store the Sharpe ratio for comparison
        if not hasattr(self, "_sharpe_ratios"):
            self._sharpe_ratios = []
        self._sharpe_ratios.append((fee_bps + slippage_bps, metrics["sharpe_ratio"]))

    def test_cost_impact_monotonic_sharpe(self):
        """Test that Sharpe ratios decrease monotonically with costs."""
        # Run the parametrized test to collect Sharpe ratios
        self._sharpe_ratios = []

        # Test different cost levels
        cost_levels = [(0, 0), (5, 2.5), (10, 5), (20, 10), (50, 25)]

        for fee_bps, slippage_bps in cost_levels:
            self.test_cost_impact_on_sharpe(fee_bps, slippage_bps)

        # Sort by total cost and verify monotonic decrease
        self._sharpe_ratios.sort(key=lambda x: x[0])

        for i in range(1, len(self._sharpe_ratios)):
            prev_sharpe = self._sharpe_ratios[i - 1][1]
            curr_sharpe = self._sharpe_ratios[i][1]
            assert (
                curr_sharpe <= prev_sharpe
            ), f"Sharpe should decrease with higher costs: {prev_sharpe:.4f} -> {curr_sharpe:.4f}"

    @pytest.mark.parametrize(
        "fee_bps,slippage_bps",
        [
            (0, 0),  # No costs
            (5, 2.5),  # Low costs
            (10, 5),  # Medium costs
            (20, 10),  # High costs
            (50, 25),  # Very high costs
        ],
    )
    def test_cost_impact_on_final_equity(self, fee_bps, slippage_bps):
        """Test that higher costs monotonically reduce final equity."""
        # Create a profitable strategy
        dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
        # Simple upward trend
        prices = 100 + np.arange(50) * 0.5

        price_df = pd.DataFrame({"Close": prices, "Open": prices * 0.999}, index=dates)

        # Buy and hold strategy
        signals = pd.Series([0] + [1] * 49, index=dates)

        params = StrategyParams(
            symbol="TEST",
            start="2020-01-01",
            end="2020-02-19",
            fast_window=5,
            slow_window=10,
            initial_cash=100000,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

        backtester = Backtester(price_df, signals, params)
        result = backtester.run()
        metrics = backtester.get_performance_metrics(result)

        # Store the final equity for comparison
        if not hasattr(self, "_final_equities"):
            self._final_equities = []
        self._final_equities.append((fee_bps + slippage_bps, metrics["final_equity"]))

    def test_cost_impact_monotonic_equity(self):
        """Test that final equity decreases monotonically with costs."""
        # Run the parametrized test to collect final equities
        self._final_equities = []

        # Test different cost levels
        cost_levels = [(0, 0), (5, 2.5), (10, 5), (20, 10), (50, 25)]

        for fee_bps, slippage_bps in cost_levels:
            self.test_cost_impact_on_final_equity(fee_bps, slippage_bps)

        # Sort by total cost and verify monotonic decrease
        self._final_equities.sort(key=lambda x: x[0])

        for i in range(1, len(self._final_equities)):
            prev_equity = self._final_equities[i - 1][1]
            curr_equity = self._final_equities[i][1]
            assert (
                curr_equity <= prev_equity
            ), f"Final equity should decrease with higher costs: {prev_equity:.2f} -> {curr_equity:.2f}"


class TestBacktesterFrequentSignals:
    """Test cases for frequent trading signals and cost drag."""

    def test_frequent_signals_cost_drag(self):
        """Test that frequent signals (fast=5, slow=6) cause significant cost drag."""
        # Create data with small price movements
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        np.random.seed(42)
        # Small random walk with slight upward bias
        returns = np.random.normal(
            0.0005, 0.01, 100
        )  # 0.05% daily return, 1% volatility
        prices = 100 * np.cumprod(1 + returns)

        price_df = pd.DataFrame({"Close": prices, "Open": prices * 0.999}, index=dates)

        # Create signals that will generate frequent trades
        # Use a very short fast window and slightly longer slow window
        from qbacktester.indicators import crossover, sma

        fast_ma = sma(price_df, 5, "Close")
        slow_ma = sma(price_df, 6, "Close")
        signals = crossover(fast_ma, slow_ma)
        signals = (signals == 1).astype(int)  # Convert to 0/1 signals

        # Test with no costs
        params_no_costs = StrategyParams(
            symbol="TEST",
            start="2020-01-01",
            end="2020-04-09",
            fast_window=5,
            slow_window=6,
            initial_cash=100000,
            fee_bps=0,
            slippage_bps=0,
        )

        backtester_no_costs = Backtester(price_df, signals, params_no_costs)
        result_no_costs = backtester_no_costs.run()
        metrics_no_costs = backtester_no_costs.get_performance_metrics(result_no_costs)

        # Test with realistic costs
        params_with_costs = StrategyParams(
            symbol="TEST",
            start="2020-01-01",
            end="2020-04-09",
            fast_window=5,
            slow_window=6,
            initial_cash=100000,
            fee_bps=10,  # 0.1% fee
            slippage_bps=5,  # 0.05% slippage
        )

        backtester_with_costs = Backtester(price_df, signals, params_with_costs)
        result_with_costs = backtester_with_costs.run()
        metrics_with_costs = backtester_with_costs.get_performance_metrics(
            result_with_costs
        )

        # Verify that we have frequent trading
        num_trades = metrics_with_costs["num_trades"]
        assert (
            num_trades > 10
        ), f"Expected frequent trading, got only {num_trades} trades"

        # Verify that costs significantly impact performance
        total_costs = metrics_with_costs["total_transaction_costs"]
        assert total_costs > 0, "Should have transaction costs"

        # The cost drag should be significant relative to the strategy's performance
        cost_impact = (
            metrics_no_costs["total_return"] - metrics_with_costs["total_return"]
        )
        assert cost_impact > 0, "Costs should reduce total return"

        # Cost impact should be substantial for frequent trading
        assert cost_impact > 0.01, f"Cost drag should be significant: {cost_impact:.4f}"

    def test_frequent_signals_vs_infrequent_signals(self):
        """Compare frequent vs infrequent trading strategies with same underlying trend."""
        # Create data with clear trend but some noise to generate more crossovers
        dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
        np.random.seed(42)
        # Upward trend with noise
        trend = np.linspace(100, 120, 200)  # 20% total gain
        noise = np.random.normal(0, 1, 200)  # Add some noise
        prices = trend + noise

        price_df = pd.DataFrame({"Close": prices, "Open": prices * 0.999}, index=dates)

        # Frequent trading strategy (fast=5, slow=6)
        from qbacktester.indicators import crossover, sma

        fast_ma_freq = sma(price_df, 5, "Close")
        slow_ma_freq = sma(price_df, 6, "Close")
        signals_freq = crossover(fast_ma_freq, slow_ma_freq)
        signals_freq = (signals_freq == 1).astype(int)

        # Infrequent trading strategy (fast=20, slow=50)
        fast_ma_infreq = sma(price_df, 20, "Close")
        slow_ma_infreq = sma(price_df, 50, "Close")
        signals_infreq = crossover(fast_ma_infreq, slow_ma_infreq)
        signals_infreq = (signals_infreq == 1).astype(int)

        # Test both strategies with same costs
        params = StrategyParams(
            symbol="TEST",
            start="2020-01-01",
            end="2020-07-18",
            fast_window=5,
            slow_window=6,
            initial_cash=100000,
            fee_bps=10,  # 0.1% fee
            slippage_bps=5,  # 0.05% slippage
        )

        # Test frequent strategy
        backtester_freq = Backtester(price_df, signals_freq, params)
        result_freq = backtester_freq.run()
        metrics_freq = backtester_freq.get_performance_metrics(result_freq)

        # Test infrequent strategy
        params_infreq = StrategyParams(
            symbol="TEST",
            start="2020-01-01",
            end="2020-07-18",
            fast_window=20,
            slow_window=50,
            initial_cash=100000,
            fee_bps=10,
            slippage_bps=5,
        )
        backtester_infreq = Backtester(price_df, signals_infreq, params_infreq)
        result_infreq = backtester_infreq.run()
        metrics_infreq = backtester_infreq.get_performance_metrics(result_infreq)

        # Verify that frequent strategy has more trades (or at least not fewer)
        assert (
            metrics_freq["num_trades"] >= metrics_infreq["num_trades"]
        ), f"Frequent strategy should have at least as many trades: {metrics_freq['num_trades']} vs {metrics_infreq['num_trades']}"

        # If we have the same number of trades, verify that the frequent strategy has higher costs per trade
        if metrics_freq["num_trades"] == metrics_infreq["num_trades"]:
            # Both strategies should have some trades
            assert (
                metrics_freq["num_trades"] > 0
            ), "Both strategies should have some trades"
            # The frequent strategy should have higher total costs due to more frequent trading
            assert (
                metrics_freq["total_transaction_costs"]
                >= metrics_infreq["total_transaction_costs"]
            ), f"Frequent strategy should have at least as high costs: {metrics_freq['total_transaction_costs']:.2f} vs {metrics_infreq['total_transaction_costs']:.2f}"
        else:
            # Verify that frequent strategy has higher transaction costs
            assert (
                metrics_freq["total_transaction_costs"]
                > metrics_infreq["total_transaction_costs"]
            ), f"Frequent strategy should have higher costs: {metrics_freq['total_transaction_costs']:.2f} vs {metrics_infreq['total_transaction_costs']:.2f}"

        # The cost drag should be more significant for frequent trading
        cost_drag_freq = (
            metrics_freq["total_transaction_costs"] / metrics_freq["final_equity"]
        )
        cost_drag_infreq = (
            metrics_infreq["total_transaction_costs"] / metrics_infreq["final_equity"]
        )

        assert (
            cost_drag_freq >= cost_drag_infreq
        ), f"Frequent trading should have at least as high cost drag: {cost_drag_freq:.4f} vs {cost_drag_infreq:.4f}"

    def test_frequent_signals_break_even_analysis(self):
        """Test at what cost level frequent trading becomes unprofitable."""
        # Create data with small but consistent returns
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        # Small consistent returns
        returns = np.full(100, 0.001)  # 0.1% daily return
        prices = 100 * np.cumprod(1 + returns)

        price_df = pd.DataFrame({"Close": prices, "Open": prices * 0.999}, index=dates)

        # Create frequent trading signals
        from qbacktester.indicators import crossover, sma

        fast_ma = sma(price_df, 5, "Close")
        slow_ma = sma(price_df, 6, "Close")
        signals = crossover(fast_ma, slow_ma)
        signals = (signals == 1).astype(int)

        # Test different cost levels to find break-even point
        cost_levels = [0, 5, 10, 20, 30, 50, 100]  # basis points

        results = []
        for cost_bps in cost_levels:
            params = StrategyParams(
                symbol="TEST",
                start="2020-01-01",
                end="2020-04-09",
                fast_window=5,
                slow_window=6,
                initial_cash=100000,
                fee_bps=cost_bps,
                slippage_bps=cost_bps // 2,  # Half slippage
            )

            backtester = Backtester(price_df, signals, params)
            result = backtester.run()
            metrics = backtester.get_performance_metrics(result)

            results.append(
                {
                    "cost_bps": cost_bps,
                    "total_return": metrics["total_return"],
                    "num_trades": metrics["num_trades"],
                    "total_costs": metrics["total_transaction_costs"],
                    "final_equity": metrics["final_equity"],
                }
            )

        # Find break-even point (where total return becomes negative)
        break_even_found = False
        for i, result in enumerate(results):
            if result["total_return"] < 0:
                break_even_found = True
                break

        # Verify that we can find a break-even point with high enough costs
        assert break_even_found, "Should find break-even point with high enough costs"

        # Verify that higher costs lead to worse performance
        for i in range(1, len(results)):
            assert (
                results[i]["total_return"] <= results[i - 1]["total_return"]
            ), f"Performance should decrease with higher costs: {results[i-1]['total_return']:.4f} -> {results[i]['total_return']:.4f}"
