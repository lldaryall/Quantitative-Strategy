"""Optimization utilities for qbacktester."""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtester import Backtester
from .data import DataError, DataLoader
from .metrics import (
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
from .strategy import StrategyParams, generate_signals


def _run_single_backtest(args: Tuple) -> Dict[str, Any]:
    """
    Run a single backtest for optimization.

    This function is designed to be called by multiprocessing workers.

    Args:
        args: Tuple containing (symbol, start, end, fast_window, slow_window,
              initial_cash, fee_bps, slippage_bps, metric)

    Returns:
        Dictionary with optimization results
    """
    (
        symbol,
        start,
        end,
        fast_window,
        slow_window,
        initial_cash,
        fee_bps,
        slippage_bps,
        metric,
    ) = args

    try:
        # Create strategy parameters
        params = StrategyParams(
            symbol=symbol,
            start=start,
            end=end,
            fast_window=fast_window,
            slow_window=slow_window,
            initial_cash=initial_cash,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

        # Load data
        data_loader = DataLoader()
        price_df = data_loader.get_price_data(symbol, start, end)

        if price_df.empty:
            return {
                "fast": fast_window,
                "slow": slow_window,
                "sharpe": np.nan,
                "max_dd": np.nan,
                "cagr": np.nan,
                "equity_final": np.nan,
                "volatility": np.nan,
                "calmar": np.nan,
                "hit_rate": np.nan,
                "num_trades": 0,
                "error": "No data available",
            }

        # Generate signals
        signals_df = generate_signals(price_df, fast_window, slow_window)
        signals = signals_df["signal"]

        # Run backtest
        backtester = Backtester(price_df, signals, params)
        backtest_result = backtester.run()

        # Calculate metrics
        equity_curve = backtest_result["total_equity"]
        returns = daily_returns(equity_curve)

        # Calculate all metrics
        sharpe_ratio = sharpe(returns)
        max_dd, _, _ = max_drawdown(equity_curve)
        cagr_value = cagr(equity_curve)
        volatility_value = volatility(returns)
        calmar_ratio = calmar(equity_curve)
        hit_rate_value = hit_rate(returns)
        num_trades = backtest_result["trade_flag"].sum()

        return {
            "fast": fast_window,
            "slow": slow_window,
            "sharpe": sharpe_ratio,
            "max_dd": max_dd,
            "cagr": cagr_value,
            "equity_final": (
                equity_curve.iloc[-1] if not equity_curve.empty else initial_cash
            ),
            "volatility": volatility_value,
            "calmar": calmar_ratio,
            "hit_rate": hit_rate_value,
            "num_trades": num_trades,
            "error": None,
        }

    except Exception as e:
        return {
            "fast": fast_window,
            "slow": slow_window,
            "sharpe": np.nan,
            "max_dd": np.nan,
            "cagr": np.nan,
            "equity_final": np.nan,
            "volatility": np.nan,
            "calmar": np.nan,
            "hit_rate": np.nan,
            "num_trades": 0,
            "error": str(e),
        }


def grid_search(
    symbol: str,
    start: str,
    end: str,
    fast_grid: List[int],
    slow_grid: List[int],
    metric: str = "sharpe",
    initial_cash: float = 100_000,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Perform grid search optimization over fast and slow moving average windows.

    Args:
        symbol: Stock symbol to optimize
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        fast_grid: List of fast window values to test
        slow_grid: List of slow window values to test
        metric: Metric to optimize for ("sharpe", "cagr", "calmar", "max_dd")
        initial_cash: Initial capital for backtests
        fee_bps: Transaction fee in basis points
        slippage_bps: Slippage in basis points
        n_jobs: Number of parallel jobs (None for auto, -1 for all cores)
        verbose: Whether to print progress information

    Returns:
        DataFrame with optimization results sorted by chosen metric
    """
    if verbose:
        print(f"🔍 Starting grid search optimization for {symbol}")
        print(f"📅 Period: {start} to {end}")
        print(f"⚡ Fast windows: {fast_grid}")
        print(f"🐌 Slow windows: {slow_grid}")
        print(f"📊 Optimizing for: {metric}")

    # Validate inputs
    if not fast_grid or not slow_grid:
        raise ValueError("fast_grid and slow_grid cannot be empty")

    if metric not in ["sharpe", "cagr", "calmar", "max_dd"]:
        raise ValueError(
            f"Invalid metric: {metric}. Must be one of: sharpe, cagr, calmar, max_dd"
        )

    # Create parameter combinations
    param_combinations = []
    for fast in fast_grid:
        for slow in slow_grid:
            if fast < slow:  # Only valid combinations
                param_combinations.append(
                    (
                        symbol,
                        start,
                        end,
                        fast,
                        slow,
                        initial_cash,
                        fee_bps,
                        slippage_bps,
                        metric,
                    )
                )

    if not param_combinations:
        raise ValueError("No valid parameter combinations found (fast must be < slow)")

    if verbose:
        print(f"🎯 Testing {len(param_combinations)} parameter combinations...")

    # Determine number of jobs
    if n_jobs is None:
        n_jobs = min(cpu_count(), len(param_combinations))
    elif n_jobs == -1:
        n_jobs = cpu_count()

    if verbose:
        print(f"🚀 Using {n_jobs} parallel workers...")

    # Run optimization
    results = []

    if n_jobs == 1 or len(param_combinations) == 1:
        # Sequential execution
        for i, args in enumerate(param_combinations):
            if verbose and i % 10 == 0:
                print(f"⏳ Progress: {i+1}/{len(param_combinations)}")

            result = _run_single_backtest(args)
            results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all jobs
            future_to_params = {
                executor.submit(_run_single_backtest, args): args
                for args in param_combinations
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_params):
                result = future.result()
                results.append(result)
                completed += 1

                if verbose and completed % 10 == 0:
                    print(f"⏳ Progress: {completed}/{len(param_combinations)}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by chosen metric (descending for most metrics, ascending for max_dd)
    if metric == "max_dd":
        results_df = results_df.sort_values(metric, ascending=True)
    else:
        results_df = results_df.sort_values(metric, ascending=False)

    # Reset index
    results_df = results_df.reset_index(drop=True)

    if verbose:
        print(f"✅ Optimization complete! Found {len(results_df)} valid results")
        print(f"🏆 Best {metric}: {results_df[metric].iloc[0]:.4f}")

    return results_df


def print_optimization_results(
    results_df: pd.DataFrame, metric: str = "sharpe", top_n: int = 5, symbol: str = ""
) -> None:
    """
    Print formatted optimization results.

    Args:
        results_df: DataFrame with optimization results
        metric: Metric that was optimized for
        top_n: Number of top results to display
        symbol: Symbol being optimized (for display)
    """
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Create title
    title = f"Optimization Results for {symbol}" if symbol else "Optimization Results"
    console.print(
        Panel(f"[bold green]{title}[/bold green]", expand=False, border_style="green")
    )

    # Create results table
    table = Table(
        title=f"Top {top_n} Parameter Sets (by {metric})", box=box.ROUNDED, style="blue"
    )
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Fast", style="magenta", width=6)
    table.add_column("Slow", style="magenta", width=6)
    table.add_column("Sharpe", style="green", width=8)
    table.add_column("CAGR", style="yellow", width=8)
    table.add_column("Max DD", style="red", width=8)
    table.add_column("Calmar", style="blue", width=8)
    table.add_column("Volatility", style="purple", width=10)
    table.add_column("Hit Rate", style="yellow", width=8)
    table.add_column("Trades", style="cyan", width=7)

    # Add top results
    for i in range(min(top_n, len(results_df))):
        row = results_df.iloc[i]
        table.add_row(
            str(i + 1),
            str(int(row["fast"])),
            str(int(row["slow"])),
            f"{row['sharpe']:.3f}",
            f"{row['cagr']:.2%}",
            f"{row['max_dd']:.2%}",
            f"{row['calmar']:.2f}",
            f"{row['volatility']:.2%}",
            f"{row['hit_rate']:.2%}",
            str(int(row["num_trades"])),
        )

    console.print(table)

    # Print summary statistics
    summary_table = Table(title="Summary Statistics", box=box.ROUNDED, style="yellow")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Best", style="magenta")
    summary_table.add_column("Worst", style="red")
    summary_table.add_column("Mean", style="green")

    for col in ["sharpe", "cagr", "max_dd", "calmar", "volatility", "hit_rate"]:
        if col in results_df.columns:
            best = results_df[col].max() if col != "max_dd" else results_df[col].min()
            worst = results_df[col].min() if col != "max_dd" else results_df[col].max()
            mean = results_df[col].mean()

            if col in ["cagr", "max_dd", "volatility", "hit_rate"]:
                summary_table.add_row(
                    col.title(), f"{best:.2%}", f"{worst:.2%}", f"{mean:.2%}"
                )
            else:
                summary_table.add_row(
                    col.title(), f"{best:.3f}", f"{worst:.3f}", f"{mean:.3f}"
                )

    console.print(summary_table)


def save_optimization_results(
    results_df: pd.DataFrame, symbol: str, output_dir: str = "reports"
) -> str:
    """
    Save optimization results to CSV file.

    Args:
        results_df: DataFrame with optimization results
        symbol: Symbol being optimized
        output_dir: Directory to save the file

    Returns:
        Path to the saved file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename
    filename = f"opt_grid_{symbol}.csv"
    filepath = os.path.join(output_dir, filename)

    # Save to CSV
    results_df.to_csv(filepath, index=False)

    return filepath


def optimize_strategy(
    symbol: str,
    start: str,
    end: str,
    fast_grid: List[int],
    slow_grid: List[int],
    metric: str = "sharpe",
    initial_cash: float = 100_000,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    n_jobs: Optional[int] = None,
    top_n: int = 5,
    save_results: bool = True,
    output_dir: str = "reports",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Complete optimization workflow with results display and saving.

    Args:
        symbol: Stock symbol to optimize
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        fast_grid: List of fast window values to test
        slow_grid: List of slow window values to test
        metric: Metric to optimize for
        initial_cash: Initial capital for backtests
        fee_bps: Transaction fee in basis points
        slippage_bps: Slippage in basis points
        n_jobs: Number of parallel jobs
        top_n: Number of top results to display
        save_results: Whether to save results to CSV
        output_dir: Directory to save results
        verbose: Whether to print progress information

    Returns:
        DataFrame with optimization results
    """
    # Run optimization
    results_df = grid_search(
        symbol=symbol,
        start=start,
        end=end,
        fast_grid=fast_grid,
        slow_grid=slow_grid,
        metric=metric,
        initial_cash=initial_cash,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # Print results
    if verbose:
        print_optimization_results(results_df, metric, top_n, symbol)

    # Save results
    if save_results:
        filepath = save_optimization_results(results_df, symbol, output_dir)
        if verbose:
            print(f"💾 Results saved to: {filepath}")

    return results_df


# Multiprocessing safety
if __name__ == "__main__":
    # This ensures multiprocessing works correctly
    pass
