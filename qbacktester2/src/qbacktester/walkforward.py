"""Walk-forward analysis utilities for qbacktester."""

import warnings
from datetime import datetime, timedelta
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
from .optimize import grid_search
from .strategy import StrategyParams, generate_signals


def walkforward_crossover(
    symbol: str,
    start: str,
    end: str,
    in_sample_years: int = 3,
    out_sample_years: int = 1,
    fast_grid: List[int] = [10, 20, 50],
    slow_grid: List[int] = [50, 100, 200],
    initial_cash: float = 100_000,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    optimization_metric: str = "sharpe",
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Perform walk-forward analysis with rolling optimization windows.

    This function implements a walk-forward analysis where:
    1. Data is split into rolling in-sample (IS) and out-of-sample (OOS) periods
    2. For each IS period, parameters are optimized using grid search
    3. The best parameters are applied to the corresponding OOS period
    4. OOS equity curves are concatenated to form the final walk-forward equity curve

    Args:
        symbol: Stock symbol to analyze
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        in_sample_years: Number of years for in-sample optimization
        out_sample_years: Number of years for out-of-sample evaluation
        fast_grid: List of fast moving average windows to test
        slow_grid: List of slow moving average windows to test
        initial_cash: Initial capital for backtests
        fee_bps: Transaction fee in basis points
        slippage_bps: Slippage in basis points
        optimization_metric: Metric to optimize for ("sharpe", "cagr", "calmar", "max_dd")
        n_jobs: Number of parallel jobs for optimization
        verbose: Whether to print progress information

    Returns:
        Dictionary containing:
        - "windows": List of window results with IS/OOS periods and best parameters
        - "equity_curve": Concatenated OOS equity curve
        - "metrics": Final walk-forward performance metrics
        - "summary": Summary statistics across all windows
    """
    # Validate parameters
    if in_sample_years <= 0:
        raise ValueError("in_sample_years must be positive")
    if out_sample_years <= 0:
        raise ValueError("out_sample_years must be positive")

    if optimization_metric not in ["sharpe", "cagr", "calmar", "max_dd"]:
        raise ValueError(
            "optimization_metric must be one of: sharpe, cagr, calmar, max_dd"
        )

    if verbose:
        print(f"🔄 Starting walk-forward analysis for {symbol}")
        print(f"📅 Period: {start} to {end}")
        print(
            f"📊 In-sample: {in_sample_years} years, Out-of-sample: {out_sample_years} years"
        )
        print(f"⚡ Fast grid: {fast_grid}, Slow grid: {slow_grid}")

    # Load data
    data_loader = DataLoader()
    try:
        price_df = data_loader.get_price_data(symbol, start, end)
        if price_df.empty:
            raise DataError("No price data loaded for the specified period.")
    except DataError as e:
        if verbose:
            print(f"❌ Error loading data: {e}")
        raise

    # Parse dates
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    # Generate walk-forward windows
    windows = _generate_walkforward_windows(
        start_date, end_date, in_sample_years, out_sample_years
    )

    if verbose:
        print(f"🎯 Generated {len(windows)} walk-forward windows")

    # Process each window
    window_results = []
    oos_equity_curves = []

    for i, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
        if verbose:
            print(f"\\n📈 Processing window {i+1}/{len(windows)}")
            print(
                f"   IS: {is_start.strftime('%Y-%m-%d')} to {is_end.strftime('%Y-%m-%d')}"
            )
            print(
                f"   OOS: {oos_start.strftime('%Y-%m-%d')} to {oos_end.strftime('%Y-%m-%d')}"
            )

        try:
            # Optimize on in-sample period
            if verbose:
                print("   🔍 Optimizing parameters on in-sample data...")

            is_results = grid_search(
                symbol=symbol,
                start=is_start.strftime("%Y-%m-%d"),
                end=is_end.strftime("%Y-%m-%d"),
                fast_grid=fast_grid,
                slow_grid=slow_grid,
                metric=optimization_metric,
                initial_cash=initial_cash,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                n_jobs=n_jobs,
                verbose=False,
            )

            if is_results.empty or is_results["sharpe"].isna().all():
                if verbose:
                    print("   ⚠️  No valid parameters found for in-sample period")
                continue

            # Get best parameters
            best_params = is_results.iloc[0]
            best_fast = int(best_params["fast"])
            best_slow = int(best_params["slow"])
            is_sharpe = best_params["sharpe"]

            if verbose:
                print(
                    f"   ✅ Best parameters: Fast={best_fast}, Slow={best_slow} (Sharpe={is_sharpe:.3f})"
                )

            # Evaluate on out-of-sample period
            if verbose:
                print("   📊 Evaluating on out-of-sample data...")

            oos_equity_curve = _evaluate_parameters(
                price_df,
                oos_start,
                oos_end,
                best_fast,
                best_slow,
                initial_cash,
                fee_bps,
                slippage_bps,
            )

            if oos_equity_curve is not None and not oos_equity_curve.empty:
                # Calculate OOS metrics
                oos_returns = daily_returns(oos_equity_curve)
                oos_metrics = {
                    "sharpe": sharpe(oos_returns),
                    "cagr": cagr(oos_equity_curve),
                    "max_dd": max_drawdown(oos_equity_curve)[0],
                    "calmar": calmar(oos_equity_curve),
                    "volatility": volatility(oos_returns),
                    "hit_rate": hit_rate(oos_returns),
                    "num_trades": _count_trades(
                        price_df, oos_start, oos_end, best_fast, best_slow
                    ),
                }

                if verbose:
                    print(
                        f"   📈 OOS Sharpe: {oos_metrics['sharpe']:.3f}, CAGR: {oos_metrics['cagr']:.2%}"
                    )

                # Store window results
                window_result = {
                    "window": i + 1,
                    "is_start": is_start,
                    "is_end": is_end,
                    "oos_start": oos_start,
                    "oos_end": oos_end,
                    "best_fast": best_fast,
                    "best_slow": best_slow,
                    "is_sharpe": is_sharpe,
                    "oos_metrics": oos_metrics,
                    "oos_equity_curve": oos_equity_curve,
                }

                window_results.append(window_result)
                oos_equity_curves.append(oos_equity_curve)

            else:
                if verbose:
                    print("   ⚠️  No valid OOS equity curve generated")

        except Exception as e:
            if verbose:
                print(f"   ❌ Error processing window {i+1}: {e}")
            continue

    if not window_results:
        if verbose:
            print("⚠️  No valid walk-forward windows could be processed")
        return {
            "windows": [],
            "equity_curve": pd.Series(dtype=float),
            "metrics": {
                "total_return": 0.0,
                "cagr": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "calmar": 0.0,
                "volatility": 0.0,
                "hit_rate": 0.0,
                "sortino": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "num_windows": 0,
                "total_trades": 0,
            },
            "summary": {
                "oos_sharpe_mean": 0.0,
                "oos_sharpe_std": 0.0,
                "oos_cagr_mean": 0.0,
                "oos_cagr_std": 0.0,
                "is_sharpe_mean": 0.0,
                "is_sharpe_std": 0.0,
                "sharpe_ratio_consistency": 0.0,
                "parameter_stability": {"fast_cv": 0.0, "slow_cv": 0.0},
            },
        }

    # Concatenate OOS equity curves
    if verbose:
        print(
            f"\\n🔗 Concatenating {len(oos_equity_curves)} out-of-sample equity curves..."
        )

    final_equity_curve = _concatenate_equity_curves(oos_equity_curves, initial_cash)

    # Calculate final metrics
    final_returns = daily_returns(final_equity_curve)
    final_metrics = {
        "total_return": (final_equity_curve.iloc[-1] / final_equity_curve.iloc[0]) - 1,
        "cagr": cagr(final_equity_curve),
        "sharpe": sharpe(final_returns),
        "max_drawdown": max_drawdown(final_equity_curve)[0],
        "calmar": calmar(final_equity_curve),
        "volatility": volatility(final_returns),
        "hit_rate": hit_rate(final_returns),
        "sortino": sortino(final_returns),
        "var_95": var(final_returns, confidence_level=0.05),
        "cvar_95": cvar(final_returns, confidence_level=0.05),
        "num_windows": len(window_results),
        "total_trades": sum(wr["oos_metrics"]["num_trades"] for wr in window_results),
    }

    # Calculate summary statistics
    summary = _calculate_walkforward_summary(window_results)

    if verbose:
        print(f"\\n✅ Walk-forward analysis completed!")
        print(f"📊 Final Sharpe: {final_metrics['sharpe']:.3f}")
        print(f"📈 Final CAGR: {final_metrics['cagr']:.2%}")
        print(f"📉 Max Drawdown: {final_metrics['max_drawdown']:.2%}")
        print(f"🎯 Processed {len(window_results)} windows")

    return {
        "windows": window_results,
        "equity_curve": final_equity_curve,
        "metrics": final_metrics,
        "summary": summary,
    }


def _generate_walkforward_windows(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    in_sample_years: int,
    out_sample_years: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate walk-forward analysis windows.

    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        in_sample_years: Number of years for in-sample optimization
        out_sample_years: Number of years for out-of-sample evaluation

    Returns:
        List of tuples (is_start, is_end, oos_start, oos_end)
    """
    windows = []
    current_date = start_date

    while current_date < end_date:
        # In-sample period
        is_start = current_date
        is_end = is_start + pd.DateOffset(years=in_sample_years)

        # Out-of-sample period
        oos_start = is_end
        oos_end = oos_start + pd.DateOffset(years=out_sample_years)

        # Check if we have enough data
        if oos_end <= end_date:
            windows.append((is_start, is_end, oos_start, oos_end))

        # Move to next window (overlapping)
        current_date = oos_start

    return windows


def _evaluate_parameters(
    price_df: pd.DataFrame,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    fast_window: int,
    slow_window: int,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
) -> Optional[pd.Series]:
    """
    Evaluate parameters on out-of-sample data.

    Args:
        price_df: Full price DataFrame
        oos_start: Out-of-sample start date
        oos_end: Out-of-sample end date
        fast_window: Fast moving average window
        slow_window: Slow moving average window
        initial_cash: Initial capital
        fee_bps: Transaction fee in basis points
        slippage_bps: Slippage in basis points

    Returns:
        Out-of-sample equity curve or None if evaluation fails
    """
    try:
        # Filter data for OOS period
        oos_data = price_df[(price_df.index >= oos_start) & (price_df.index <= oos_end)]

        if oos_data.empty:
            return None

        # Generate signals
        signals_df = generate_signals(oos_data, fast_window, slow_window)
        signals = signals_df["signal"]

        # Create strategy parameters
        params = StrategyParams(
            symbol="WALKFORWARD",
            start=oos_start.strftime("%Y-%m-%d"),
            end=oos_end.strftime("%Y-%m-%d"),
            fast_window=fast_window,
            slow_window=slow_window,
            initial_cash=initial_cash,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

        # Run backtest
        backtester = Backtester(oos_data, signals, params)
        result = backtester.run()

        return result["total_equity"]

    except Exception:
        return None


def _concatenate_equity_curves(
    equity_curves: List[pd.Series], initial_cash: float
) -> pd.Series:
    """
    Concatenate multiple equity curves into a single walk-forward curve.

    Args:
        equity_curves: List of equity curve Series
        initial_cash: Initial capital

    Returns:
        Concatenated equity curve
    """
    if not equity_curves:
        return pd.Series([initial_cash], index=[pd.Timestamp.now()])

    # Start with initial cash
    final_curve = [initial_cash]
    final_index = [equity_curves[0].index[0]]

    for i, curve in enumerate(equity_curves):
        if i == 0:
            # First curve: use as-is
            final_curve.extend(curve.iloc[1:].values)
            final_index.extend(curve.index[1:])
        else:
            # Subsequent curves: scale to previous curve's end value
            prev_end = final_curve[-1]
            curve_start = curve.iloc[0]
            scale_factor = prev_end / curve_start

            scaled_curve = curve * scale_factor
            final_curve.extend(scaled_curve.iloc[1:].values)
            final_index.extend(curve.index[1:])

    return pd.Series(final_curve, index=final_index)


def _count_trades(
    price_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    fast_window: int,
    slow_window: int,
) -> int:
    """
    Count the number of trades in a given period.

    Args:
        price_df: Full price DataFrame
        start_date: Start date
        end_date: End date
        fast_window: Fast moving average window
        slow_window: Slow moving average window

    Returns:
        Number of trades
    """
    try:
        # Filter data for the period
        period_data = price_df[
            (price_df.index >= start_date) & (price_df.index <= end_date)
        ]

        if period_data.empty:
            return 0

        # Generate signals
        signals_df = generate_signals(period_data, fast_window, slow_window)
        signals = signals_df["signal"]

        # Count signal changes (trades)
        signal_changes = signals.diff().fillna(0)
        return int((signal_changes != 0).sum())

    except Exception:
        return 0


def _calculate_walkforward_summary(
    window_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate summary statistics for walk-forward analysis.

    Args:
        window_results: List of window results

    Returns:
        Summary statistics dictionary
    """
    if not window_results:
        return {}

    # Extract metrics
    oos_sharpes = [wr["oos_metrics"]["sharpe"] for wr in window_results]
    oos_cagrs = [wr["oos_metrics"]["cagr"] for wr in window_results]
    oos_max_dds = [wr["oos_metrics"]["max_dd"] for wr in window_results]
    is_sharpes = [wr["is_sharpe"] for wr in window_results]

    # Calculate statistics
    summary = {
        "num_windows": len(window_results),
        "oos_sharpe_mean": np.mean(oos_sharpes),
        "oos_sharpe_std": np.std(oos_sharpes),
        "oos_cagr_mean": np.mean(oos_cagrs),
        "oos_cagr_std": np.std(oos_cagrs),
        "oos_max_dd_mean": np.mean(oos_max_dds),
        "oos_max_dd_std": np.std(oos_max_dds),
        "is_sharpe_mean": np.mean(is_sharpes),
        "is_sharpe_std": np.std(is_sharpes),
        "sharpe_ratio_consistency": (
            np.corrcoef(is_sharpes, oos_sharpes)[0, 1] if len(is_sharpes) > 1 else 0
        ),
        "parameter_stability": _calculate_parameter_stability(window_results),
    }

    return summary


def _calculate_parameter_stability(
    window_results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Calculate parameter stability across windows.

    Args:
        window_results: List of window results

    Returns:
        Parameter stability metrics
    """
    if len(window_results) < 2:
        return {"fast_std": 0, "slow_std": 0, "fast_cv": 0, "slow_cv": 0}

    # Extract parameter values, handling missing keys
    fast_windows = [
        wr.get("best_fast", np.nan) for wr in window_results if "best_fast" in wr
    ]
    slow_windows = [
        wr.get("best_slow", np.nan) for wr in window_results if "best_slow" in wr
    ]

    # Filter out NaN values
    fast_windows = [x for x in fast_windows if not np.isnan(x)]
    slow_windows = [x for x in slow_windows if not np.isnan(x)]

    return {
        "fast_std": np.std(fast_windows),
        "slow_std": np.std(slow_windows),
        "fast_cv": (
            np.std(fast_windows) / np.mean(fast_windows)
            if np.mean(fast_windows) > 0
            else 0
        ),
        "slow_cv": (
            np.std(slow_windows) / np.mean(slow_windows)
            if np.mean(slow_windows) > 0
            else 0
        ),
    }


def print_walkforward_results(
    results: Dict[str, Any], title: str = "Walk-Forward Analysis Results"
) -> None:
    """
    Print formatted walk-forward analysis results.

    Args:
        results: Dictionary containing walk-forward results
        title: Title for the report
    """
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Create title
    console.print(
        Panel(f"[bold green]{title}[/bold green]", expand=False, border_style="green")
    )

    # Final metrics table
    metrics = results["metrics"]
    metrics_table = Table(
        title="Final Walk-Forward Performance", box=box.ROUNDED, style="blue"
    )
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="magenta")
    metrics_table.add_column("Description", style="green")

    metrics_table.add_row(
        "Total Return", f"{metrics['total_return']:.2%}", "Overall portfolio return"
    )
    metrics_table.add_row(
        "CAGR", f"{metrics['cagr']:.2%}", "Compound Annual Growth Rate"
    )
    metrics_table.add_row(
        "Sharpe Ratio", f"{metrics['sharpe']:.3f}", "Risk-adjusted return"
    )
    metrics_table.add_row(
        "Max Drawdown",
        f"{metrics['max_drawdown']:.2%}",
        "Maximum peak-to-trough decline",
    )
    metrics_table.add_row(
        "Calmar Ratio", f"{metrics['calmar']:.3f}", "CAGR / Max Drawdown"
    )
    metrics_table.add_row(
        "Volatility", f"{metrics['volatility']:.2%}", "Annualized volatility"
    )
    metrics_table.add_row(
        "Hit Rate", f"{metrics['hit_rate']:.2%}", "Percentage of positive return days"
    )
    metrics_table.add_row(
        "Sortino Ratio", f"{metrics['sortino']:.3f}", "Downside risk-adjusted return"
    )
    metrics_table.add_row("VaR (5%)", f"{metrics['var_95']:.2%}", "5% Value at Risk")
    metrics_table.add_row(
        "CVaR (5%)", f"{metrics['cvar_95']:.2%}", "5% Conditional Value at Risk"
    )
    metrics_table.add_row(
        "Number of Windows", f"{metrics['num_windows']}", "Total walk-forward windows"
    )
    metrics_table.add_row(
        "Total Trades", f"{metrics['total_trades']}", "Total trades across all windows"
    )

    console.print(metrics_table)

    # Summary statistics
    summary = results["summary"]
    summary_table = Table(
        title="Walk-Forward Summary Statistics", box=box.ROUNDED, style="yellow"
    )
    summary_table.add_column("Statistic", style="cyan")
    summary_table.add_column("Value", style="magenta")
    summary_table.add_column("Description", style="green")

    summary_table.add_row(
        "OOS Sharpe Mean",
        f"{summary.get('oos_sharpe_mean', 0):.3f}",
        "Average out-of-sample Sharpe",
    )
    summary_table.add_row(
        "OOS Sharpe Std",
        f"{summary.get('oos_sharpe_std', 0):.3f}",
        "OOS Sharpe standard deviation",
    )
    summary_table.add_row(
        "OOS CAGR Mean",
        f"{summary.get('oos_cagr_mean', 0):.2%}",
        "Average out-of-sample CAGR",
    )
    summary_table.add_row(
        "OOS CAGR Std",
        f"{summary.get('oos_cagr_std', 0):.2%}",
        "OOS CAGR standard deviation",
    )
    summary_table.add_row(
        "IS Sharpe Mean",
        f"{summary.get('is_sharpe_mean', 0):.3f}",
        "Average in-sample Sharpe",
    )
    summary_table.add_row(
        "IS Sharpe Std",
        f"{summary.get('is_sharpe_std', 0):.3f}",
        "IS Sharpe standard deviation",
    )
    summary_table.add_row(
        "Sharpe Consistency",
        f"{summary.get('sharpe_ratio_consistency', 0):.3f}",
        "IS vs OOS Sharpe correlation",
    )

    param_stability = summary.get("parameter_stability", {})
    summary_table.add_row(
        "Fast MA Stability",
        f"{param_stability.get('fast_cv', 0):.3f}",
        "Fast MA coefficient of variation",
    )
    summary_table.add_row(
        "Slow MA Stability",
        f"{param_stability.get('slow_cv', 0):.3f}",
        "Slow MA coefficient of variation",
    )

    console.print(summary_table)

    # Window details
    windows = results["windows"]
    if windows:
        window_table = Table(title="Window Details", box=box.ROUNDED, style="green")
        window_table.add_column("Window", style="cyan", width=6)
        window_table.add_column("IS Period", style="magenta", width=20)
        window_table.add_column("OOS Period", style="yellow", width=20)
        window_table.add_column("Best Fast", style="blue", width=8)
        window_table.add_column("Best Slow", style="blue", width=8)
        window_table.add_column("IS Sharpe", style="green", width=8)
        window_table.add_column("OOS Sharpe", style="red", width=9)
        window_table.add_column("OOS CAGR", style="purple", width=8)

        for wr in windows[:10]:  # Show first 10 windows
            window_table.add_row(
                str(wr["window"]),
                f"{wr['is_start'].strftime('%Y-%m-%d')} to {wr['is_end'].strftime('%Y-%m-%d')}",
                f"{wr['oos_start'].strftime('%Y-%m-%d')} to {wr['oos_end'].strftime('%Y-%m-%d')}",
                str(wr["best_fast"]),
                str(wr["best_slow"]),
                f"{wr['is_sharpe']:.3f}",
                f"{wr['oos_metrics']['sharpe']:.3f}",
                f"{wr['oos_metrics']['cagr']:.2%}",
            )

        if len(windows) > 10:
            window_table.add_row("...", "...", "...", "...", "...", "...", "...", "...")

        console.print(window_table)
