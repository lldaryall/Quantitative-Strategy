# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Professional CI/CD pipeline with GitHub Actions
- Pre-commit hooks for code quality
- Comprehensive error handling with custom exceptions
- Configuration management system
- Professional development tools (Makefile, setup scripts)
- Security scanning with bandit
- Enhanced type hints and documentation
- Contributing guidelines and security policy
- Professional project metadata and classifiers

### Changed
- Enhanced project structure and organization
- Improved error handling throughout the codebase
- Updated project metadata for professional appearance
- Enhanced documentation and docstrings
- Improved code quality and consistency

### Security
- Added security scanning with bandit
- Implemented comprehensive error handling
- Added security policy and vulnerability reporting process
- Enhanced input validation and sanitization

## [0.1.0] - 2024-12-01

### Added
- Initial release of qbacktester
- **Core Backtesting Engine**
  - `Backtester` class for vectorized portfolio simulation
  - Support for long-only strategies with realistic transaction costs
  - Next-bar execution to prevent look-ahead bias
  - Configurable fees and slippage modeling

- **Data Management**
  - `DataLoader` class with Yahoo Finance integration
  - Local caching with parquet files for performance
  - Automatic data validation and error handling
  - Support for multiple data sources and intervals

- **Technical Indicators**
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Crossover detection
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Stochastic Oscillator

- **Strategy Framework**
  - `StrategyParams` dataclass for configuration
  - Moving Average Crossover strategy implementation
  - Buy and Hold strategy for benchmarking
  - Vectorized signal generation

- **Performance Metrics**
  - Total return, CAGR, Sharpe ratio
  - Maximum drawdown with peak/trough dates
  - Calmar ratio, Sortino ratio, volatility
  - Hit rate, average win/loss ratio
  - VaR (Value at Risk) and CVaR (Conditional VaR)
  - Transaction cost analysis

- **Visualization**
  - Equity curve plotting with performance metrics
  - Drawdown visualization
  - Price charts with buy/sell signals
  - Comprehensive backtest plots
  - Professional matplotlib styling

- **Parameter Optimization**
  - Grid search across parameter combinations
  - Parallel processing support
  - Multiple optimization metrics (Sharpe, CAGR, etc.)
  - Results export to CSV

- **Walk-Forward Analysis**
  - Rolling window optimization
  - In-sample/out-of-sample validation
  - Parameter stability analysis
  - Overfitting detection

- **Command Line Interface**
  - `qbacktester run` for basic backtesting
  - `qbacktester optimize` for parameter optimization
  - `qbacktester walkforward` for robust validation
  - Rich terminal output with tables and colors
  - Plot generation and saving

- **Python API**
  - Clean, intuitive API design
  - Comprehensive type hints
  - Extensive documentation
  - Jupyter notebook tutorial

- **Testing & Quality**
  - Comprehensive test suite with pytest
  - Parametrized tests for edge cases
  - Performance profiling and validation
  - Code coverage reporting
  - Linting with black, isort, and mypy

- **Documentation**
  - Detailed README with examples
  - API documentation
  - Jupyter notebook quickstart guide
  - Performance benchmarks
  - Best practices and assumptions

### Performance
- Vectorized operations throughout for maximum speed
- ~50% performance improvement over looped implementations
- Supports 100+ backtests in under 1 second
- Memory-efficient data structures
- Parallel processing for optimization

### Dependencies
- Python 3.11+
- pandas >= 2.0.0
- numpy >= 1.24.0
- yfinance >= 0.2.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- rich >= 13.0.0

### License
- MIT License
