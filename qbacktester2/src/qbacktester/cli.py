"""Command-line interface for qbacktester."""

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .optimize import optimize_strategy
from .run import print_backtest_report, run_crossover_backtest
from .strategy import StrategyParams
from .walkforward import print_walkforward_results, walkforward_crossover


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="qbacktester - A quantitative backtesting framework", prog="qbt"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a crossover backtest")
    run_parser.add_argument(
        "--symbol", required=True, help="Stock symbol (e.g., SPY, AAPL)"
    )
    run_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    run_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    run_parser.add_argument(
        "--fast", type=int, default=20, help="Fast moving average window (default: 20)"
    )
    run_parser.add_argument(
        "--slow", type=int, default=50, help="Slow moving average window (default: 50)"
    )
    run_parser.add_argument(
        "--cash", type=float, default=100000, help="Initial cash (default: 100000)"
    )
    run_parser.add_argument(
        "--fee-bps", type=float, default=1.0, help="Fee in basis points (default: 1.0)"
    )
    run_parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.5,
        help="Slippage in basis points (default: 0.5)",
    )
    run_parser.add_argument(
        "--plot", action="store_true", help="Generate equity curve plot"
    )
    run_parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for plots (default: reports)",
    )

    # Optimize command
    optimize_parser = subparsers.add_parser(
        "optimize", help="Optimize strategy parameters"
    )
    optimize_parser.add_argument(
        "--symbol", required=True, help="Stock symbol (e.g., SPY, AAPL)"
    )
    optimize_parser.add_argument(
        "--start", required=True, help="Start date (YYYY-MM-DD)"
    )
    optimize_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    optimize_parser.add_argument(
        "--fast", required=True, help="Fast windows (comma-separated, e.g., 5,10,20,50)"
    )
    optimize_parser.add_argument(
        "--slow",
        required=True,
        help="Slow windows (comma-separated, e.g., 50,100,150,200)",
    )
    optimize_parser.add_argument(
        "--metric",
        default="sharpe",
        choices=["sharpe", "cagr", "calmar", "max_dd"],
        help="Metric to optimize for (default: sharpe)",
    )
    optimize_parser.add_argument(
        "--cash", type=float, default=100000, help="Initial cash (default: 100000)"
    )
    optimize_parser.add_argument(
        "--fee-bps", type=float, default=1.0, help="Fee in basis points (default: 1.0)"
    )
    optimize_parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.5,
        help="Slippage in basis points (default: 0.5)",
    )
    optimize_parser.add_argument(
        "--jobs", type=int, default=None, help="Number of parallel jobs (default: auto)"
    )
    optimize_parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top results to display (default: 5)",
    )
    optimize_parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for results (default: reports)",
    )

    # Walkforward command
    walkforward_parser = subparsers.add_parser(
        "walkforward", help="Run walk-forward analysis"
    )
    walkforward_parser.add_argument(
        "--symbol", required=True, help="Stock symbol (e.g., SPY, AAPL)"
    )
    walkforward_parser.add_argument(
        "--start", required=True, help="Start date (YYYY-MM-DD)"
    )
    walkforward_parser.add_argument(
        "--end", required=True, help="End date (YYYY-MM-DD)"
    )
    walkforward_parser.add_argument(
        "--is", type=int, default=3, help="In-sample years (default: 3)"
    )
    walkforward_parser.add_argument(
        "--oos", type=int, default=1, help="Out-of-sample years (default: 1)"
    )
    walkforward_parser.add_argument(
        "--fast",
        default="10,20,50",
        help="Fast windows (comma-separated, default: 10,20,50)",
    )
    walkforward_parser.add_argument(
        "--slow",
        default="50,100,200",
        help="Slow windows (comma-separated, default: 50,100,200)",
    )
    walkforward_parser.add_argument(
        "--metric",
        default="sharpe",
        choices=["sharpe", "cagr", "calmar", "max_dd"],
        help="Optimization metric (default: sharpe)",
    )
    walkforward_parser.add_argument(
        "--cash", type=float, default=100000, help="Initial cash (default: 100000)"
    )
    walkforward_parser.add_argument(
        "--fee-bps", type=float, default=1.0, help="Fee in basis points (default: 1.0)"
    )
    walkforward_parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.5,
        help="Slippage in basis points (default: 0.5)",
    )
    walkforward_parser.add_argument(
        "--jobs", type=int, default=None, help="Number of parallel jobs (default: auto)"
    )
    walkforward_parser.add_argument(
        "--plot", action="store_true", help="Generate equity curve plot"
    )
    walkforward_parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for plots (default: reports)",
    )

    args = parser.parse_args()

    if args.command == "run":
        return handle_run_command(args)
    elif args.command == "optimize":
        return handle_optimize_command(args)
    elif args.command == "walkforward":
        return handle_walkforward_command(args)
    else:
        parser.print_help()
        return 1


def handle_run_command(args) -> int:
    """Handle run command."""
    try:
        # Validate arguments
        if args.fast >= args.slow:
            print("Error: Fast window must be less than slow window", file=sys.stderr)
            return 1

        if args.cash <= 0:
            print("Error: Initial cash must be positive", file=sys.stderr)
            return 1

        if args.fee_bps < 0 or args.slippage_bps < 0:
            print("Error: Fee and slippage must be non-negative", file=sys.stderr)
            return 1

        # Validate dates
        if not validate_date(args.start):
            print("Error: Invalid start date format. Use YYYY-MM-DD", file=sys.stderr)
            return 1

        if not validate_date(args.end):
            print("Error: Invalid end date format. Use YYYY-MM-DD", file=sys.stderr)
            return 1

        # Create strategy parameters
        params = StrategyParams(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            fast_window=args.fast,
            slow_window=args.slow,
            initial_cash=args.cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
        )

        # Run backtest
        print(f"Running crossover backtest for {args.symbol}...")
        results = run_crossover_backtest(params)

        # Print results
        print_backtest_report(
            results, title=f"{args.symbol} Crossover Strategy ({args.fast}/{args.slow})"
        )

        # Generate plot if requested
        if args.plot:
            plot_file = generate_equity_plot(results, args.output_dir)
            print(f"\nEquity curve plot saved to: {plot_file}")

        return 0

    except KeyboardInterrupt:
        print("\nBacktest interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error running backtest: {e}", file=sys.stderr)
        return 1


def handle_optimize_command(args) -> int:
    """Handle optimize command."""
    try:
        # Parse fast and slow grids
        try:
            fast_grid = [int(x.strip()) for x in args.fast.split(",")]
            slow_grid = [int(x.strip()) for x in args.slow.split(",")]
        except ValueError as e:
            print(
                f"Error: Invalid fast/slow grid format. Use comma-separated integers. {e}",
                file=sys.stderr,
            )
            return 1

        # Validate grids
        if not fast_grid or not slow_grid:
            print("Error: Fast and slow grids cannot be empty", file=sys.stderr)
            return 1

        if any(x <= 0 for x in fast_grid + slow_grid):
            print("Error: All grid values must be positive", file=sys.stderr)
            return 1

        # Validate dates
        if not validate_date(args.start):
            print("Error: Invalid start date format. Use YYYY-MM-DD", file=sys.stderr)
            return 1

        if not validate_date(args.end):
            print("Error: Invalid end date format. Use YYYY-MM-DD", file=sys.stderr)
            return 1

        # Validate other parameters
        if args.cash <= 0:
            print("Error: Initial cash must be positive", file=sys.stderr)
            return 1

        if args.fee_bps < 0 or args.slippage_bps < 0:
            print("Error: Fee and slippage must be non-negative", file=sys.stderr)
            return 1

        if args.top <= 0:
            print("Error: Top results count must be positive", file=sys.stderr)
            return 1

        # Run optimization
        print(f"🚀 Starting optimization for {args.symbol}...")
        results_df = optimize_strategy(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            fast_grid=fast_grid,
            slow_grid=slow_grid,
            metric=args.metric,
            initial_cash=args.cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            n_jobs=args.jobs,
            top_n=args.top,
            save_results=True,
            output_dir=args.output_dir,
            verbose=True,
        )

        print(f"✅ Optimization completed successfully!")
        print(f"📊 Tested {len(results_df)} parameter combinations")

        return 0

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error during optimization: {e}", file=sys.stderr)
        return 1


def handle_walkforward_command(args) -> int:
    """Handle walkforward command."""
    try:
        # Parse fast and slow grids
        try:
            fast_grid = [int(x.strip()) for x in args.fast.split(",")]
            slow_grid = [int(x.strip()) for x in args.slow.split(",")]
        except ValueError as e:
            print(
                f"Error: Invalid fast/slow grid format. Use comma-separated integers. {e}",
                file=sys.stderr,
            )
            return 1

        # Validate grids
        if not fast_grid or not slow_grid:
            print("Error: Fast and slow grids cannot be empty", file=sys.stderr)
            return 1

        if any(x <= 0 for x in fast_grid + slow_grid):
            print("Error: All grid values must be positive", file=sys.stderr)
            return 1

        # Validate dates
        if not validate_date(args.start):
            print("Error: Invalid start date format. Use YYYY-MM-DD", file=sys.stderr)
            return 1

        if not validate_date(args.end):
            print("Error: Invalid end date format. Use YYYY-MM-DD", file=sys.stderr)
            return 1

        # Validate other parameters
        if args.cash <= 0:
            print("Error: Initial cash must be positive", file=sys.stderr)
            return 1

        if args.fee_bps < 0 or args.slippage_bps < 0:
            print("Error: Fee and slippage must be non-negative", file=sys.stderr)
            return 1

        if args.in_sample_years <= 0 or args.out_sample_years <= 0:
            print(
                "Error: In-sample and out-of-sample years must be positive",
                file=sys.stderr,
            )
            return 1

        # Run walk-forward analysis
        print(f"🔄 Starting walk-forward analysis for {args.symbol}...")
        results = walkforward_crossover(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            in_sample_years=args.in_sample_years,
            out_sample_years=args.out_sample_years,
            fast_grid=fast_grid,
            slow_grid=slow_grid,
            initial_cash=args.cash,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            optimization_metric=args.metric,
            n_jobs=args.jobs,
            verbose=True,
        )

        # Print results
        print_walkforward_results(
            results,
            title=f"{args.symbol} Walk-Forward Analysis ({args.in_sample_years}Y IS / {args.out_sample_years}Y OOS)",
        )

        # Generate plot if requested
        if args.plot:
            print(f"\\n📊 Generating equity curve plot...")
            plot_path = generate_walkforward_plot(results, args.symbol, args.output_dir)
            if plot_path:
                print(f"📈 Walk-forward plot saved to: {plot_path}")
            else:
                print("❌ Failed to generate walk-forward plot")

        print(f"\\n✅ Walk-forward analysis completed successfully!")
        print(f"📊 Processed {results['metrics']['num_windows']} windows")
        print(f"📈 Final Sharpe: {results['metrics']['sharpe']:.3f}")
        print(f"📉 Final Max DD: {results['metrics']['max_drawdown']:.2%}")

        return 0

    except KeyboardInterrupt:
        print("\\nWalk-forward analysis interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error during walk-forward analysis: {e}", file=sys.stderr)
        return 1


def generate_walkforward_plot(results: dict, symbol: str, output_dir: str) -> str:
    """
    Generate walk-forward equity curve plot.

    Args:
        results: Walk-forward analysis results
        symbol: Stock symbol
        output_dir: Output directory for the plot

    Returns:
        Path to the saved plot file
    """
    try:
        equity_curve = results["equity_curve"]
        metrics = results["metrics"]

        if equity_curve.empty:
            print("Warning: Cannot plot empty equity curve")
            return ""

        # Create plot
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        fig.suptitle(f"{symbol} Walk-Forward Analysis", fontsize=16, fontweight="bold")

        # Equity curve
        ax1.plot(
            equity_curve.index,
            equity_curve,
            linewidth=2,
            color="#2E86AB",
            label="Walk-Forward Equity",
        )
        ax1.axhline(
            equity_curve.iloc[0],
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Initial Capital",
        )
        ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Drawdown
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        ax2.fill_between(
            drawdown.index, drawdown, 0, color="#F24236", alpha=0.6, label="Drawdown"
        )
        ax2.axhline(0, color="black", linestyle="-", alpha=0.5)
        ax2.set_ylabel("Drawdown (%)", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2%}"))

        # Add metrics text
        metrics_text = (
            f"Final Sharpe: {metrics['sharpe']:.3f}\\n"
            f"Final CAGR: {metrics['cagr']:.2%}\\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\\n"
            f"Calmar Ratio: {metrics['calmar']:.3f}\\n"
            f"Windows: {metrics['num_windows']}\\n"
            f"Total Trades: {metrics['total_trades']}"
        )
        ax1.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"walkforward_{symbol}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return filepath

    except Exception as e:
        print(f"Error generating walk-forward plot: {e}")
        return ""


def generate_equity_plot(results: dict, output_dir: str) -> str:
    """
    Generate equity curve and drawdown plot.

    Args:
        results: Results dictionary from run_crossover_backtest
        output_dir: Directory to save the plot

    Returns:
        Path to the saved plot file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    params = results["params"]
    equity_curve = results["equity_curve"]
    metrics = results["metrics"]

    # Calculate drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Equity curve
    ax1.plot(
        equity_curve.index,
        equity_curve.values,
        linewidth=2,
        color="#2E86AB",
        label="Portfolio Value",
    )
    ax1.axhline(
        y=params.initial_cash,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="Initial Capital",
    )
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title(
        f"{params.symbol} Crossover Strategy ({params.fast_window}/{params.slow_window}) - Equity Curve"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Plot 2: Drawdown
    ax2.fill_between(
        equity_curve.index,
        drawdown.values,
        0,
        color="#F24236",
        alpha=0.7,
        label="Drawdown",
    )
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.set_title("Portfolio Drawdown")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Add performance metrics as text
    metrics_text = f"""Performance Summary:
Total Return: {metrics['total_return']:.2%}
CAGR: {metrics['cagr']:.2%}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Max Drawdown: {metrics['max_drawdown']:.2%}
Volatility: {metrics['volatility']:.2%}
Hit Rate: {metrics['hit_rate']:.2%}
Number of Trades: {metrics['num_trades']:.0f}"""

    ax1.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=9,
        fontfamily="monospace",
    )

    # Adjust layout
    plt.tight_layout()

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"equity_{params.symbol}_{params.fast_window}_{params.slow_window}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # Save plot
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    return filepath


def validate_date(date_str: str) -> bool:
    """Validate date string format."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    sys.exit(main())
