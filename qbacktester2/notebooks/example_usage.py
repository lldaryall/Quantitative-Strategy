"""
Example usage of qbacktester library.

This script demonstrates how to use the qbacktester library for backtesting
trading strategies with real market data.
"""

import sys
import os
sys.path.append(os.path.join('..', 'src'))

from qbacktester import Backtester, DataLoader, BuyAndHoldStrategy, MovingAverageCrossoverStrategy


def main():
    """Main example function."""
    print("qbacktester Example Usage")
    print("=" * 40)
    
    # 1. Load data
    print("\n1. Loading data...")
    data_loader = DataLoader()
    data = data_loader.load_yahoo_data(
        symbol="AAPL",
        start_date="2020-01-01",
        end_date="2023-12-31"
    )
    
    if data.empty:
        print("Failed to load data. Please check your internet connection.")
        return
    
    print(f"Loaded {len(data)} records for AAPL")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # 2. Validate data
    print("\n2. Validating data...")
    if not data_loader.validate_data(data):
        print("Data validation failed.")
        return
    print("Data validation passed.")
    
    # 3. Test Buy and Hold Strategy
    print("\n3. Testing Buy and Hold Strategy...")
    strategy = BuyAndHoldStrategy()
    backtester = Backtester(initial_capital=100000)
    results = backtester.run(strategy, data)
    backtester.print_results(results)
    
    # 4. Test Moving Average Crossover Strategy
    print("\n4. Testing Moving Average Crossover Strategy...")
    strategy = MovingAverageCrossoverStrategy(short_window=20, long_window=50)
    backtester = Backtester(initial_capital=100000)
    results = backtester.run(strategy, data)
    backtester.print_results(results)
    
    # 5. Compare strategies
    print("\n5. Strategy Comparison...")
    strategies = {
        'Buy & Hold': BuyAndHoldStrategy(),
        'MA Crossover': MovingAverageCrossoverStrategy(20, 50)
    }
    
    for name, strategy in strategies.items():
        backtester = Backtester(initial_capital=100000)
        results = backtester.run(strategy, data)
        print(f"\n{name}:")
        print(f"  Total Return: {results['total_return']:.2%}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
