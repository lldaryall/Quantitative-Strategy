"""Plotting utilities for qbacktester."""

import os
from datetime import datetime
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_equity(equity_curve: pd.Series, title: str = "Equity Curve") -> Figure:
    """
    Create a clean equity curve plot.

    Args:
        equity_curve: Series of portfolio values over time
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot equity curve (only if series is not empty)
    if len(equity_curve) > 0:
        ax.plot(
            equity_curve.index,
            equity_curve.values,
            linewidth=2,
            color="#2E86AB",
            label="Portfolio Value",
        )

        # Add initial capital line
        initial_capital = equity_curve.iloc[0]
        ax.axhline(
            y=initial_capital,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Initial Capital",
        )
    else:
        # Handle empty series
        ax.text(
            0.5,
            0.5,
            "No data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )

    # Formatting
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Tight layout
    plt.tight_layout()

    return fig


def plot_drawdown(equity_curve: pd.Series, title: str = "Portfolio Drawdown") -> Figure:
    """
    Create a clean drawdown plot.

    Args:
        equity_curve: Series of portfolio values over time
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100

    # Plot drawdown
    ax.fill_between(
        equity_curve.index,
        drawdown.values,
        0,
        color="#F24236",
        alpha=0.7,
        label="Drawdown",
    )

    # Add zero line
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)

    # Formatting
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Tight layout
    plt.tight_layout()

    return fig


def plot_price_signals(
    price_df: pd.DataFrame,
    signals: pd.Series,
    fast: pd.Series,
    slow: pd.Series,
    title: str = "Price and Signals",
) -> Figure:
    """
    Create a plot showing price data, moving averages, and trading signals.

    Args:
        price_df: DataFrame with OHLCV data
        signals: Series of trading signals (0 or 1)
        fast: Series of fast moving average values
        slow: Series of slow moving average values
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Price and moving averages
    ax1.plot(
        price_df.index,
        price_df["Close"],
        linewidth=1.5,
        color="#2E86AB",
        label="Close Price",
        alpha=0.8,
    )
    ax1.plot(
        fast.index,
        fast.values,
        linewidth=2,
        color="#F24236",
        label="Fast MA",
        alpha=0.8,
    )
    ax1.plot(
        slow.index,
        slow.values,
        linewidth=2,
        color="#F6AE2D",
        label="Slow MA",
        alpha=0.8,
    )

    # Add buy/sell signals
    buy_signals = signals[signals == 1]
    if not buy_signals.empty:
        buy_prices = price_df.loc[buy_signals.index, "Close"]
        ax1.scatter(
            buy_signals.index,
            buy_prices,
            color="green",
            marker="^",
            s=100,
            label="Buy Signal",
            zorder=5,
        )

    # Formatting for price plot
    ax1.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax1.set_ylabel("Price ($)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:.2f}"))

    # Plot 2: Signals
    signal_colors = ["red" if s == 0 else "green" for s in signals]
    ax2.bar(signals.index, signals.values, color=signal_colors, alpha=0.7, width=1)
    ax2.set_ylabel("Signal", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Tight layout
    plt.tight_layout()

    return fig


def save_figure(fig: Figure, filename: str, output_dir: str = "reports") -> str:
    """
    Save a figure to a file with proper formatting.

    Args:
        fig: matplotlib Figure object
        filename: Name of the file (without extension)
        output_dir: Directory to save the file

    Returns:
        Path to the saved file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate full filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename}_{timestamp}.png"
    filepath = os.path.join(output_dir, full_filename)

    # Save figure
    fig.savefig(
        filepath, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )

    return filepath


def plot_equity_with_metrics(
    equity_curve: pd.Series, metrics: dict, title: str = "Equity Curve with Metrics"
) -> Figure:
    """
    Create an equity curve plot with performance metrics overlay.

    Args:
        equity_curve: Series of portfolio values over time
        metrics: Dictionary of performance metrics
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot equity curve
    ax.plot(
        equity_curve.index,
        equity_curve.values,
        linewidth=2,
        color="#2E86AB",
        label="Portfolio Value",
    )

    # Add initial capital line
    initial_capital = equity_curve.iloc[0]
    ax.axhline(
        y=initial_capital,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="Initial Capital",
    )

    # Add performance metrics as text
    metrics_text = f"""Performance Summary:
Total Return: {metrics.get('total_return', 0):.2%}
CAGR: {metrics.get('cagr', 0):.2%}
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
Volatility: {metrics.get('volatility', 0):.2%}
Hit Rate: {metrics.get('hit_rate', 0):.2%}
Number of Trades: {metrics.get('num_trades', 0):.0f}"""

    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=9,
        fontfamily="monospace",
    )

    # Formatting
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Tight layout
    plt.tight_layout()

    return fig


def plot_comprehensive_backtest(
    equity_curve: pd.Series,
    price_df: pd.DataFrame,
    signals: pd.Series,
    fast: pd.Series,
    slow: pd.Series,
    metrics: dict,
    title: str = "Comprehensive Backtest Analysis",
) -> Figure:
    """
    Create a comprehensive 4-panel backtest analysis plot.

    Args:
        equity_curve: Series of portfolio values over time
        price_df: DataFrame with OHLCV data
        signals: Series of trading signals (0 or 1)
        fast: Series of fast moving average values
        slow: Series of slow moving average values
        metrics: Dictionary of performance metrics
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Equity curve
    ax1.plot(
        equity_curve.index,
        equity_curve.values,
        linewidth=2,
        color="#2E86AB",
        label="Portfolio Value",
    )
    ax1.axhline(
        y=equity_curve.iloc[0],
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="Initial Capital",
    )
    ax1.set_title("Equity Curve", fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Plot 2: Price and signals
    ax2.plot(
        price_df.index,
        price_df["Close"],
        linewidth=1,
        color="#2E86AB",
        label="Close Price",
        alpha=0.8,
    )
    ax2.plot(
        fast.index,
        fast.values,
        linewidth=2,
        color="#F24236",
        label="Fast MA",
        alpha=0.8,
    )
    ax2.plot(
        slow.index,
        slow.values,
        linewidth=2,
        color="#F6AE2D",
        label="Slow MA",
        alpha=0.8,
    )

    buy_signals = signals[signals == 1]
    if not buy_signals.empty:
        buy_prices = price_df.loc[buy_signals.index, "Close"]
        ax2.scatter(
            buy_signals.index,
            buy_prices,
            color="green",
            marker="^",
            s=50,
            label="Buy Signal",
            zorder=5,
        )

    ax2.set_title("Price and Signals", fontweight="bold")
    ax2.set_ylabel("Price ($)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:.2f}"))

    # Plot 3: Drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100
    ax3.fill_between(
        equity_curve.index,
        drawdown.values,
        0,
        color="#F24236",
        alpha=0.7,
        label="Drawdown",
    )
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax3.set_title("Portfolio Drawdown", fontweight="bold")
    ax3.set_ylabel("Drawdown (%)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Signals timeline
    signal_colors = ["red" if s == 0 else "green" for s in signals]
    ax4.bar(signals.index, signals.values, color=signal_colors, alpha=0.7, width=1)
    ax4.set_title("Trading Signals", fontweight="bold")
    ax4.set_ylabel("Signal")
    ax4.set_xlabel("Date")
    ax4.set_ylim(-0.1, 1.1)
    ax4.grid(True, alpha=0.3)

    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # Tight layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    return fig


def create_plot_style() -> None:
    """
    Set up a clean plotting style for qbacktester plots.
    """
    plt.style.use("default")

    # Set default colors
    colors = ["#2E86AB", "#F24236", "#F6AE2D", "#7209B7", "#F77F00"]
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)

    # Set default font
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]

    # Set default figure properties
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.edgecolor"] = "none"

    # Set default line properties
    plt.rcParams["lines.linewidth"] = 1.5
    # Note: lines.alpha is not a valid rcParam, using individual plot alpha instead

    # Set default grid properties
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "-"

    # Set default legend properties
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.fancybox"] = True
    plt.rcParams["legend.shadow"] = True
    plt.rcParams["legend.framealpha"] = 0.9
