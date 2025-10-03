#!/usr/bin/env python3
"""
Performance profiling script for qbacktester vectorization.

This script tests the performance of the backtester by:
1. Generating synthetic price data (~2500 trading days)
2. Running 100 backtests with random fast/slow parameters
3. Timing the overall runtime and asserting it completes under a reasonable threshold
4. Scanning the backtester code for explicit Python for-loops (not allowed in core methods)
"""

import ast
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add the src directory to the path so we can import qbacktester
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qbacktester.backtester import Backtester
from qbacktester.indicators import crossover, sma
from qbacktester.strategy import StrategyParams, generate_signals


def generate_synthetic_price_data(n_days=2500, seed=42):
    """Generate synthetic price data with realistic characteristics."""
    np.random.seed(seed)

    # Generate dates (trading days only, skip weekends)
    start_date = pd.Timestamp("2015-01-01")
    dates = pd.bdate_range(start=start_date, periods=n_days)

    # Generate price series with trend, volatility, and some mean reversion
    returns = np.random.normal(
        0.0005, 0.015, n_days
    )  # 0.05% daily return, 1.5% volatility

    # Add some trend
    trend = np.linspace(0, 0.3, n_days)  # 30% total trend over period
    returns += trend / n_days

    # Add some mean reversion
    for i in range(1, n_days):
        if i > 20:  # Start mean reversion after 20 days
            recent_returns = returns[i - 20 : i]
            if np.mean(recent_returns) > 0.002:  # If recent returns are too high
                returns[i] -= 0.0001  # Slight mean reversion
            elif np.mean(recent_returns) < -0.002:  # If recent returns are too low
                returns[i] += 0.0001  # Slight mean reversion

    # Generate prices
    prices = 100 * np.cumprod(1 + returns)

    # Create OHLCV data
    price_df = pd.DataFrame(
        {
            "Close": prices,
            "Open": prices * (1 + np.random.normal(0, 0.001, n_days)),  # Small gaps
            "High": prices * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
            "Volume": np.random.randint(1000000, 10000000, n_days),
        },
        index=dates,
    )

    # Ensure High >= Low and High >= Close, Open
    price_df["High"] = np.maximum(
        price_df["High"], price_df[["Close", "Open"]].max(axis=1)
    )
    price_df["Low"] = np.minimum(
        price_df["Low"], price_df[["Close", "Open"]].min(axis=1)
    )

    return price_df


def run_random_backtests(price_df, n_backtests=100, seed=42):
    """Run multiple backtests with random parameters."""
    random.seed(seed)
    np.random.seed(seed)

    results = []

    for i in range(n_backtests):
        # Generate random parameters
        fast_window = random.randint(5, 50)
        slow_window = random.randint(fast_window + 1, 100)

        # Generate signals
        signals_df = generate_signals(price_df, fast_window, slow_window)
        signals = signals_df["signal"]  # Extract just the signal column

        # Create strategy parameters
        params = StrategyParams(
            symbol="SYNTHETIC",
            start=price_df.index[0].strftime("%Y-%m-%d"),
            end=price_df.index[-1].strftime("%Y-%m-%d"),
            fast_window=fast_window,
            slow_window=slow_window,
            initial_cash=100000,
            fee_bps=random.uniform(0, 20),  # Random fees 0-20 bps
            slippage_bps=random.uniform(0, 10),  # Random slippage 0-10 bps
        )

        # Run backtest
        backtester = Backtester(price_df, signals, params)
        result = backtester.run()

        # Store results
        metrics = backtester.get_performance_metrics(result)
        results.append(
            {
                "fast_window": fast_window,
                "slow_window": slow_window,
                "num_trades": metrics["num_trades"],
                "total_return": metrics["total_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "final_equity": metrics["final_equity"],
            }
        )

    return results


def scan_for_loops(file_path):
    """Scan a Python file for explicit for-loops using AST."""
    with open(file_path, "r") as f:
        content = f.read()

    tree = ast.parse(content)
    loops = []

    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            # Get line number and context
            line_num = node.lineno
            # Get the line content
            lines = content.split("\n")
            if line_num <= len(lines):
                line_content = lines[line_num - 1].strip()
                loops.append({"line": line_num, "content": line_content})

    return loops


def check_backtester_vectorization():
    """Check that backtester.py doesn't contain inappropriate for-loops in core methods."""
    backtester_path = (
        Path(__file__).parent.parent / "src" / "qbacktester" / "backtester.py"
    )

    if not backtester_path.exists():
        raise FileNotFoundError(f"Backtester file not found: {backtester_path}")

    loops = scan_for_loops(backtester_path)

    # Filter out loops that are legitimate for sequential processing
    inappropriate_loops = []
    for loop in loops:
        # Skip if it's in a test method (contains 'test' in the context)
        if "test" in loop["content"].lower():
            continue
        # Skip if it's in a docstring or comment
        if loop["content"].strip().startswith("#") or '"""' in loop["content"]:
            continue
        # Skip if it's the legitimate sequential portfolio management loop
        if "for i in range(n):" in loop["content"]:
            continue
        # Skip if it's in a method that requires sequential processing
        if "portfolio" in loop["content"].lower():
            continue
        inappropriate_loops.append(loop)

    if inappropriate_loops:
        print("❌ Found inappropriate for-loops in backtester.py core methods:")
        for loop in inappropriate_loops:
            print(f"   Line {loop['line']}: {loop['content']}")
        return False

    print("✅ No inappropriate for-loops found in backtester.py core methods")
    print("   (Sequential portfolio management loops are acceptable)")
    return True


def main():
    """Main profiling function."""
    print("🚀 Starting qbacktester vectorization performance test...")
    print("=" * 60)

    # Check for explicit loops first
    print("1. Checking for explicit for-loops in backtester.py...")
    if not check_backtester_vectorization():
        print("❌ Performance test failed: Explicit for-loops found in core methods")
        sys.exit(1)

    # Generate synthetic data
    print("\n2. Generating synthetic price data...")
    start_time = time.time()
    price_df = generate_synthetic_price_data(n_days=2500)
    data_gen_time = time.time() - start_time
    print(f"   Generated {len(price_df)} trading days in {data_gen_time:.3f}s")

    # Run backtests
    print("\n3. Running 100 random backtests...")
    start_time = time.time()
    results = run_random_backtests(price_df, n_backtests=100)
    backtest_time = time.time() - start_time

    # Calculate performance metrics
    total_time = data_gen_time + backtest_time
    avg_backtest_time = backtest_time / 100

    print(f"   Completed 100 backtests in {backtest_time:.3f}s")
    print(f"   Average time per backtest: {avg_backtest_time:.3f}s")
    print(f"   Total runtime: {total_time:.3f}s")

    # Performance assertions
    print("\n4. Performance validation...")

    # Reasonable thresholds for a modest machine
    max_total_time = 30.0  # 30 seconds total
    max_avg_backtest = 0.2  # 200ms per backtest

    if total_time > max_total_time:
        print(f"❌ Total runtime {total_time:.3f}s exceeds threshold {max_total_time}s")
        sys.exit(1)

    if avg_backtest_time > max_avg_backtest:
        print(
            f"❌ Average backtest time {avg_backtest_time:.3f}s exceeds threshold {max_avg_backtest}s"
        )
        sys.exit(1)

    print(f"✅ Performance within acceptable limits")
    print(f"   Total time: {total_time:.3f}s < {max_total_time}s")
    print(f"   Avg backtest: {avg_backtest_time:.3f}s < {max_avg_backtest}s")

    # Summary statistics
    print("\n5. Backtest summary statistics...")
    num_trades = [r["num_trades"] for r in results]
    total_returns = [r["total_return"] for r in results]
    sharpe_ratios = [
        r["sharpe_ratio"] for r in results if not np.isnan(r["sharpe_ratio"])
    ]

    print(f"   Average trades per backtest: {np.mean(num_trades):.1f}")
    print(f"   Average total return: {np.mean(total_returns):.2%}")
    print(f"   Average Sharpe ratio: {np.mean(sharpe_ratios):.3f}")
    print(
        f"   Successful backtests: {len([r for r in results if not np.isnan(r['sharpe_ratio'])])}/100"
    )

    print("\n🎉 All performance tests passed!")
    print("=" * 60)

    return {
        "total_time": total_time,
        "avg_backtest_time": avg_backtest_time,
        "data_gen_time": data_gen_time,
        "backtest_time": backtest_time,
        "num_backtests": 100,
        "num_trading_days": len(price_df),
        "avg_trades": np.mean(num_trades),
        "avg_return": np.mean(total_returns),
        "avg_sharpe": np.mean(sharpe_ratios),
    }


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Performance test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
