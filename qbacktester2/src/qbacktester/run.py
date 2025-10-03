"""Main execution module for running backtests."""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .backtester import Backtester
from .data import DataLoader
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
from .strategy import generate_signals


def run_crossover_backtest(params: "StrategyParams") -> Dict[str, Any]:
    """
    Run a complete crossover backtest pipeline.

    This function orchestrates the entire backtesting process:
    1. Pull data using DataLoader
    2. Generate signals using crossover strategy
    3. Run backtest using Backtester
    4. Compute comprehensive performance metrics
    5. Extract trade information

    Args:
        params: StrategyParams object with strategy configuration

    Returns:
        Dictionary containing:
        - "params": Original strategy parameters
        - "equity_curve": Series of portfolio values over time
        - "metrics": Dictionary of performance metrics
        - "trades": DataFrame with trade information (timestamp, side, price)

    Raises:
        Exception: If any step in the pipeline fails
    """
    console = Console()

    try:
        # Step 1: Pull data
        console.print(
            f"[cyan]Loading data for {params.symbol} from {params.start} to {params.end}[/cyan]"
        )
        data_loader = DataLoader()
        price_df = data_loader.get_price_data(
            symbol=params.symbol, start=params.start, end=params.end, interval="1d"
        )

        if price_df.empty:
            raise ValueError(f"No data found for symbol {params.symbol}")

        console.print(f"[green]Loaded {len(price_df)} days of data[/green]")

        # Step 2: Generate signals
        console.print(
            f"[cyan]Generating signals with fast_window={params.fast_window}, slow_window={params.slow_window}[/cyan]"
        )
        signals_df = generate_signals(price_df, params.fast_window, params.slow_window)
        signals = signals_df["signal"]

        # Count signals
        signal_count = signals.sum()
        console.print(
            f"[green]Generated {signal_count} buy signals out of {len(signals)} days[/green]"
        )

        # Step 3: Run backtest
        console.print("[cyan]Running backtest...[/cyan]")
        backtester = Backtester(price_df, signals, params)
        result = backtester.run()

        # Extract equity curve
        equity_curve = result["total_equity"]

        # Step 4: Compute metrics
        console.print("[cyan]Computing performance metrics...[/cyan]")
        daily_ret = daily_returns(equity_curve)

        metrics = {
            # Basic metrics
            "cagr": cagr(equity_curve),
            "sharpe_ratio": sharpe(daily_ret),
            "max_drawdown": max_drawdown(equity_curve)[0],  # Just the magnitude
            "calmar_ratio": calmar(equity_curve),
            "hit_rate": hit_rate(daily_ret),
            "volatility": volatility(daily_ret),
            "sortino_ratio": sortino(daily_ret),
            # Win/Loss metrics
            "avg_win": avg_win_loss(daily_ret)[0],
            "avg_loss": avg_win_loss(daily_ret)[1],
            # Risk metrics
            "var_5pct": var(daily_ret, confidence_level=0.05),
            "cvar_5pct": cvar(daily_ret, confidence_level=0.05),
            # Trade metrics
            "num_trades": result["trade_flag"].sum(),
            "total_transaction_costs": result["transaction_cost"].sum(),
            # Portfolio metrics
            "initial_capital": params.initial_cash,
            "final_equity": equity_curve.iloc[-1],
            "total_return": (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
            # Drawdown details
            "max_dd_peak_date": max_drawdown(equity_curve)[1],
            "max_dd_trough_date": max_drawdown(equity_curve)[2],
        }

        # Step 5: Extract trade information
        trades = _extract_trades(result, price_df)

        console.print("[green]Backtest completed successfully![/green]")

        return {
            "params": params,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "trades": trades,
        }

    except Exception as e:
        console.print(f"[red]Backtest failed: {e}[/red]")
        raise


def _extract_trades(result: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract trade information from backtest results.

    Args:
        result: DataFrame from Backtester.run()
        price_df: Original price DataFrame

    Returns:
        DataFrame with columns: timestamp, side, price, quantity, notional, cost
    """
    trade_days = result[result["trade_flag"]].copy()

    if trade_days.empty:
        return pd.DataFrame(
            columns=["timestamp", "side", "price", "quantity", "notional", "cost"]
        )

    trades = []

    for idx, row in trade_days.iterrows():
        # Determine trade side based on position change
        if row["position"] == 1:
            side = "BUY"
        else:
            side = "SELL"

        # Calculate quantity (approximate)
        if row["notional"] > 0:
            quantity = row["notional"] / row["trade_price"]
        else:
            quantity = 0

        trades.append(
            {
                "timestamp": idx,
                "side": side,
                "price": row["trade_price"],
                "quantity": quantity,
                "notional": row["notional"],
                "cost": row["transaction_cost"],
            }
        )

    return pd.DataFrame(trades)


def print_backtest_report(
    results: Dict[str, Any], title: str = "Backtest Results"
) -> None:
    """
    Print a formatted backtest report using Rich.

    Args:
        results: Results dictionary from run_crossover_backtest
        title: Title for the report
    """
    console = Console()

    # Extract data
    params = results["params"]
    metrics = results["metrics"]
    trades = results["trades"]

    # Create main panel
    console.print(Panel.fit(title, style="bold blue"))

    # Strategy parameters table
    params_table = Table(
        title="Strategy Parameters", show_header=True, header_style="bold magenta"
    )
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="white")

    params_table.add_row("Symbol", params.symbol)
    params_table.add_row("Start Date", params.start)
    params_table.add_row("End Date", params.end)
    params_table.add_row("Fast Window", str(params.fast_window))
    params_table.add_row("Slow Window", str(params.slow_window))
    params_table.add_row("Initial Capital", f"${params.initial_cash:,.2f}")
    params_table.add_row("Fee (bps)", f"{params.fee_bps:.1f}")
    params_table.add_row("Slippage (bps)", f"{params.slippage_bps:.1f}")

    console.print(params_table)
    console.print()

    # Performance metrics table
    metrics_table = Table(
        title="Performance Metrics", show_header=True, header_style="bold green"
    )
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="white")
    metrics_table.add_column("Description", style="dim")

    # Basic performance
    metrics_table.add_row(
        "Total Return", f"{metrics['total_return']:.2%}", "Total portfolio return"
    )
    metrics_table.add_row(
        "CAGR", f"{metrics['cagr']:.2%}", "Compound Annual Growth Rate"
    )
    metrics_table.add_row(
        "Final Equity", f"${metrics['final_equity']:,.2f}", "Final portfolio value"
    )

    # Risk metrics
    metrics_table.add_row(
        "Max Drawdown",
        f"{metrics['max_drawdown']:.2%}",
        "Maximum peak-to-trough decline",
    )
    metrics_table.add_row(
        "Calmar Ratio", f"{metrics['calmar_ratio']:.2f}", "CAGR / Max Drawdown"
    )
    metrics_table.add_row(
        "Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}", "Risk-adjusted return"
    )
    metrics_table.add_row(
        "Sortino Ratio",
        f"{metrics['sortino_ratio']:.2f}",
        "Downside risk-adjusted return",
    )
    metrics_table.add_row(
        "Volatility", f"{metrics['volatility']:.2%}", "Annualized volatility"
    )

    # Trade metrics
    metrics_table.add_row(
        "Hit Rate", f"{metrics['hit_rate']:.2%}", "Percentage of positive return days"
    )
    metrics_table.add_row(
        "Avg Win", f"{metrics['avg_win']:.2%}", "Average winning day return"
    )
    metrics_table.add_row(
        "Avg Loss", f"{metrics['avg_loss']:.2%}", "Average losing day return"
    )
    metrics_table.add_row(
        "Number of Trades",
        f"{metrics['num_trades']:.0f}",
        "Total number of trades executed",
    )
    metrics_table.add_row(
        "Transaction Costs",
        f"${metrics['total_transaction_costs']:,.2f}",
        "Total fees and slippage",
    )

    # Risk metrics (VaR/CVaR)
    metrics_table.add_row("VaR (5%)", f"{metrics['var_5pct']:.2%}", "5% Value at Risk")
    metrics_table.add_row(
        "CVaR (5%)", f"{metrics['cvar_5pct']:.2%}", "5% Conditional Value at Risk"
    )

    console.print(metrics_table)
    console.print()

    # Drawdown details
    if metrics["max_drawdown"] > 0:
        dd_table = Table(
            title="Maximum Drawdown Details", show_header=True, header_style="bold red"
        )
        dd_table.add_column("Detail", style="cyan")
        dd_table.add_column("Value", style="white")

        dd_table.add_row("Magnitude", f"{metrics['max_drawdown']:.2%}")
        dd_table.add_row("Peak Date", str(metrics["max_dd_peak_date"].date()))
        dd_table.add_row("Trough Date", str(metrics["max_dd_trough_date"].date()))

        console.print(dd_table)
        console.print()

    # Trade summary
    if not trades.empty:
        trade_table = Table(
            title="Trade Summary", show_header=True, header_style="bold yellow"
        )
        trade_table.add_column("Timestamp", style="cyan")
        trade_table.add_column("Side", style="white")
        trade_table.add_column("Price", style="white")
        trade_table.add_column("Quantity", style="white")
        trade_table.add_column("Notional", style="white")
        trade_table.add_column("Cost", style="white")

        # Show first 10 trades
        for _, trade in trades.head(10).iterrows():
            trade_table.add_row(
                str(trade["timestamp"].date()),
                trade["side"],
                f"${trade['price']:.2f}",
                f"{trade['quantity']:.2f}",
                f"${trade['notional']:,.2f}",
                f"${trade['cost']:.2f}",
            )

        if len(trades) > 10:
            trade_table.add_row("...", "...", "...", "...", "...", "...")
            trade_table.add_row(f"({len(trades) - 10} more trades)", "", "", "", "", "")

        console.print(trade_table)
    else:
        console.print("[yellow]No trades executed during this period[/yellow]")


def run_quick_backtest(
    symbol: str = "AAPL",
    start: str = "2023-01-01",
    end: str = "2023-12-31",
    fast_window: int = 5,
    slow_window: int = 20,
    initial_cash: float = 100_000,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
) -> Dict[str, Any]:
    """
    Run a quick backtest with default parameters.

    This is a convenience function for running backtests with common parameters.

    Args:
        symbol: Stock symbol to backtest
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        fast_window: Fast moving average window
        slow_window: Slow moving average window
        initial_cash: Starting capital
        fee_bps: Fee in basis points
        slippage_bps: Slippage in basis points

    Returns:
        Results dictionary from run_crossover_backtest
    """
    from .strategy import StrategyParams

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

    return run_crossover_backtest(params)
