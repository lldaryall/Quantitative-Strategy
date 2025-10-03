"""Tests for the plotting module."""

import os
import tempfile
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from qbacktester.plotting import (
    create_plot_style,
    plot_comprehensive_backtest,
    plot_drawdown,
    plot_equity,
    plot_equity_with_metrics,
    plot_price_signals,
    save_figure,
)


class TestPlotEquity:
    """Test cases for plot_equity function."""

    def test_plot_equity_basic(self):
        """Test basic equity curve plotting."""
        # Create test data
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        equity_curve = pd.Series([100000 + i * 1000 for i in range(10)], index=dates)

        # Test function
        fig = plot_equity(equity_curve, "Test Equity Curve")

        # Verify return type
        assert isinstance(fig, Figure)

        # Verify figure has content
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert len(ax.lines) == 2  # Equity curve + initial capital line
        assert ax.get_title() == "Test Equity Curve"

        # Clean up
        plt.close(fig)

    def test_plot_equity_minimal_data(self):
        """Test equity plotting with minimal data."""
        # Single data point
        dates = pd.date_range(start="2023-01-01", periods=1, freq="D")
        equity_curve = pd.Series([100000], index=dates)

        fig = plot_equity(equity_curve)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_plot_equity_empty_series(self):
        """Test equity plotting with empty series."""
        equity_curve = pd.Series([], dtype=float)

        # Should handle empty series gracefully
        fig = plot_equity(equity_curve)

        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotDrawdown:
    """Test cases for plot_drawdown function."""

    def test_plot_drawdown_basic(self):
        """Test basic drawdown plotting."""
        # Create test data with drawdown
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        values = [
            100000,
            105000,
            110000,
            108000,
            112000,
            109000,
            115000,
            113000,
            118000,
            120000,
        ]
        equity_curve = pd.Series(values, index=dates)

        fig = plot_drawdown(equity_curve, "Test Drawdown")

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_title() == "Test Drawdown"

        plt.close(fig)

    def test_plot_drawdown_constant_series(self):
        """Test drawdown plotting with constant series."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        equity_curve = pd.Series([100000] * 5, index=dates)

        fig = plot_drawdown(equity_curve)

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_drawdown_monotonic_increase(self):
        """Test drawdown plotting with monotonic increase."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        equity_curve = pd.Series([100000 + i * 1000 for i in range(5)], index=dates)

        fig = plot_drawdown(equity_curve)

        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotPriceSignals:
    """Test cases for plot_price_signals function."""

    def test_plot_price_signals_basic(self):
        """Test basic price and signals plotting."""
        # Create test data
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        price_df = pd.DataFrame(
            {
                "Close": [100 + i * 2 for i in range(10)],
                "Open": [99 + i * 2 for i in range(10)],
                "High": [101 + i * 2 for i in range(10)],
                "Low": [98 + i * 2 for i in range(10)],
                "Volume": [1000 + i * 100 for i in range(10)],
            },
            index=dates,
        )

        signals = pd.Series([0, 0, 1, 0, 0, 1, 0, 0, 0, 0], index=dates)
        fast = pd.Series([100 + i * 1.5 for i in range(10)], index=dates)
        slow = pd.Series([100 + i * 1.8 for i in range(10)], index=dates)

        fig = plot_price_signals(price_df, signals, fast, slow, "Test Price Signals")

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2  # Price plot + signals plot

        plt.close(fig)

    def test_plot_price_signals_no_signals(self):
        """Test price plotting with no trading signals."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        price_df = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104],
                "Open": [99, 100, 101, 102, 103],
                "High": [101, 102, 103, 104, 105],
                "Low": [98, 99, 100, 101, 102],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=dates,
        )

        signals = pd.Series([0, 0, 0, 0, 0], index=dates)
        fast = pd.Series([100, 100.5, 101, 101.5, 102], index=dates)
        slow = pd.Series([100, 100.2, 100.4, 100.6, 100.8], index=dates)

        fig = plot_price_signals(price_df, signals, fast, slow)

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_price_signals_all_signals(self):
        """Test price plotting with all buy signals."""
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        price_df = pd.DataFrame(
            {
                "Close": [100, 101, 102],
                "Open": [99, 100, 101],
                "High": [101, 102, 103],
                "Low": [98, 99, 100],
                "Volume": [1000, 1100, 1200],
            },
            index=dates,
        )

        signals = pd.Series([1, 1, 1], index=dates)
        fast = pd.Series([100, 100.5, 101], index=dates)
        slow = pd.Series([100, 100.2, 100.4], index=dates)

        fig = plot_price_signals(price_df, signals, fast, slow)

        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotEquityWithMetrics:
    """Test cases for plot_equity_with_metrics function."""

    def test_plot_equity_with_metrics_basic(self):
        """Test equity plotting with metrics overlay."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        equity_curve = pd.Series([100000 + i * 1000 for i in range(10)], index=dates)

        metrics = {
            "total_return": 0.05,
            "cagr": 0.12,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.03,
            "volatility": 0.15,
            "hit_rate": 0.6,
            "num_trades": 4,
        }

        fig = plot_equity_with_metrics(equity_curve, metrics, "Test with Metrics")

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_title() == "Test with Metrics"

        # Check that metrics text is present
        text_objects = ax.texts
        assert len(text_objects) > 0  # Should have metrics text

        plt.close(fig)

    def test_plot_equity_with_metrics_empty_metrics(self):
        """Test equity plotting with empty metrics."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        equity_curve = pd.Series([100000 + i * 1000 for i in range(5)], index=dates)

        metrics = {}

        fig = plot_equity_with_metrics(equity_curve, metrics)

        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotComprehensiveBacktest:
    """Test cases for plot_comprehensive_backtest function."""

    def test_plot_comprehensive_backtest_basic(self):
        """Test comprehensive backtest plotting."""
        # Create test data
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        equity_curve = pd.Series([100000 + i * 1000 for i in range(10)], index=dates)

        price_df = pd.DataFrame(
            {
                "Close": [100 + i * 2 for i in range(10)],
                "Open": [99 + i * 2 for i in range(10)],
                "High": [101 + i * 2 for i in range(10)],
                "Low": [98 + i * 2 for i in range(10)],
                "Volume": [1000 + i * 100 for i in range(10)],
            },
            index=dates,
        )

        signals = pd.Series([0, 0, 1, 0, 0, 1, 0, 0, 0, 0], index=dates)
        fast = pd.Series([100 + i * 1.5 for i in range(10)], index=dates)
        slow = pd.Series([100 + i * 1.8 for i in range(10)], index=dates)

        metrics = {
            "total_return": 0.05,
            "cagr": 0.12,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.03,
            "volatility": 0.15,
            "hit_rate": 0.6,
            "num_trades": 4,
        }

        fig = plot_comprehensive_backtest(
            equity_curve, price_df, signals, fast, slow, metrics
        )

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 4  # Four subplots

        plt.close(fig)

    def test_plot_comprehensive_backtest_minimal_data(self):
        """Test comprehensive plotting with minimal data."""
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        equity_curve = pd.Series([100000, 101000, 102000], index=dates)

        price_df = pd.DataFrame(
            {
                "Close": [100, 101, 102],
                "Open": [99, 100, 101],
                "High": [101, 102, 103],
                "Low": [98, 99, 100],
                "Volume": [1000, 1100, 1200],
            },
            index=dates,
        )

        signals = pd.Series([0, 1, 0], index=dates)
        fast = pd.Series([100, 100.5, 101], index=dates)
        slow = pd.Series([100, 100.2, 100.4], index=dates)

        metrics = {"total_return": 0.02}

        fig = plot_comprehensive_backtest(
            equity_curve, price_df, signals, fast, slow, metrics
        )

        assert isinstance(fig, Figure)
        plt.close(fig)


class TestSaveFigure:
    """Test cases for save_figure function."""

    def test_save_figure_basic(self):
        """Test basic figure saving."""
        # Create a simple figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Figure")

        # Test saving
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = save_figure(fig, "test_plot", temp_dir)

            # Verify file was created
            assert os.path.exists(filepath)
            assert filepath.endswith(".png")
            assert "test_plot" in filepath
            assert temp_dir in filepath

            # Verify file has content
            assert os.path.getsize(filepath) > 0

        plt.close(fig)

    def test_save_figure_creates_directory(self):
        """Test that save_figure creates output directory if it doesn't exist."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "new_directory")
            filepath = save_figure(fig, "test_plot", new_dir)

            # Verify directory was created
            assert os.path.exists(new_dir)
            assert os.path.exists(filepath)

        plt.close(fig)

    def test_save_figure_timestamp(self):
        """Test that save_figure adds timestamp to filename."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = save_figure(fig, "test_plot", temp_dir)

            # Verify timestamp is in filename
            assert "_" in filepath  # Should have timestamp separator
            assert filepath.endswith(".png")

        plt.close(fig)


class TestCreatePlotStyle:
    """Test cases for create_plot_style function."""

    def test_create_plot_style_no_error(self):
        """Test that create_plot_style runs without error."""
        # Should not raise any exceptions
        create_plot_style()

        # Verify some style settings were applied
        assert plt.rcParams["figure.dpi"] == 100
        assert plt.rcParams["savefig.dpi"] == 300
        assert plt.rcParams["savefig.bbox"] == "tight"


class TestPlottingIntegration:
    """Integration tests for plotting functions."""

    def test_all_plotting_functions_return_figures(self):
        """Test that all plotting functions return Figure objects."""
        # Create test data
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        equity_curve = pd.Series([100000 + i * 1000 for i in range(10)], index=dates)

        price_df = pd.DataFrame(
            {
                "Close": [100 + i * 2 for i in range(10)],
                "Open": [99 + i * 2 for i in range(10)],
                "High": [101 + i * 2 for i in range(10)],
                "Low": [98 + i * 2 for i in range(10)],
                "Volume": [1000 + i * 100 for i in range(10)],
            },
            index=dates,
        )

        signals = pd.Series([0, 0, 1, 0, 0, 1, 0, 0, 0, 0], index=dates)
        fast = pd.Series([100 + i * 1.5 for i in range(10)], index=dates)
        slow = pd.Series([100 + i * 1.8 for i in range(10)], index=dates)

        metrics = {
            "total_return": 0.05,
            "cagr": 0.12,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.03,
            "volatility": 0.15,
            "hit_rate": 0.6,
            "num_trades": 4,
        }

        # Test all plotting functions
        functions_to_test = [
            (plot_equity, (equity_curve,)),
            (plot_drawdown, (equity_curve,)),
            (plot_price_signals, (price_df, signals, fast, slow)),
            (plot_equity_with_metrics, (equity_curve, metrics)),
            (
                plot_comprehensive_backtest,
                (equity_curve, price_df, signals, fast, slow, metrics),
            ),
        ]

        for func, args in functions_to_test:
            fig = func(*args)
            assert isinstance(fig, Figure)
            plt.close(fig)

    def test_plotting_functions_with_edge_cases(self):
        """Test plotting functions with various edge cases."""
        # Test with single data point
        dates = pd.date_range(start="2023-01-01", periods=1, freq="D")
        equity_curve = pd.Series([100000], index=dates)

        fig = plot_equity(equity_curve)
        assert isinstance(fig, Figure)
        plt.close(fig)

        fig = plot_drawdown(equity_curve)
        assert isinstance(fig, Figure)
        plt.close(fig)

        # Test with constant values
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        equity_curve = pd.Series([100000] * 5, index=dates)

        fig = plot_equity(equity_curve)
        assert isinstance(fig, Figure)
        plt.close(fig)

        fig = plot_drawdown(equity_curve)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plotting_functions_handle_nan_values(self):
        """Test that plotting functions handle NaN values gracefully."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        values = [100000, np.nan, 102000, np.nan, 104000]
        equity_curve = pd.Series(values, index=dates)

        # Should not raise errors
        fig = plot_equity(equity_curve)
        assert isinstance(fig, Figure)
        plt.close(fig)

        fig = plot_drawdown(equity_curve)
        assert isinstance(fig, Figure)
        plt.close(fig)
