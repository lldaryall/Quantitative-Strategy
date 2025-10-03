"""
qbacktester: A professional quantitative backtesting library for financial strategies.

This package provides comprehensive tools for backtesting trading strategies, analyzing 
performance, and generating professional reports for quantitative finance research.

Key Features:
- Vectorized backtesting engine for high performance
- Professional CLI interface with rich formatting
- Comprehensive risk and performance metrics
- Walk-forward analysis for robust strategy validation
- Parameter optimization with parallel processing
- Beautiful visualizations and reporting
- Docker support for easy deployment
- Extensive test coverage and type hints

Author: Quantitative Strategy Team
License: MIT
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Quantitative Strategy Team"
__email__ = "quant@example.com"
__license__ = "MIT"
__status__ = "Beta"

from .backtester import Backtester
from .config import config
from .data import DataError, DataLoader
from .exceptions import (
    BacktestError,
    CacheError,
    ConfigurationError,
    DataError as QBacktesterDataError,
    NetworkError,
    OptimizationError,
    PlottingError,
    QBacktesterError,
    StrategyError,
    ValidationError,
)
from .indicators import bollinger_bands, crossover, ema, macd, rsi, sma, stochastic
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
from .optimize import (
    grid_search,
    optimize_strategy,
    print_optimization_results,
    save_optimization_results,
)
from .plotting import (
    create_plot_style,
    plot_comprehensive_backtest,
    plot_drawdown,
    plot_equity,
    plot_equity_with_metrics,
    plot_price_signals,
    save_figure,
)
from .run import print_backtest_report, run_crossover_backtest, run_quick_backtest
from .strategy import (
    BuyAndHoldStrategy,
    MovingAverageCrossoverStrategy,
    Strategy,
    StrategyParams,
    generate_signals,
)
from .walkforward import print_walkforward_results, walkforward_crossover

__all__ = [
    # Core classes
    "Backtester",
    "DataLoader",
    "Strategy",
    "BuyAndHoldStrategy",
    "MovingAverageCrossoverStrategy",
    "StrategyParams",
    "config",
    # Exceptions
    "QBacktesterError",
    "DataError",
    "QBacktesterDataError",
    "ValidationError",
    "BacktestError",
    "StrategyError",
    "OptimizationError",
    "PlottingError",
    "ConfigurationError",
    "NetworkError",
    "CacheError",
    # Strategy functions
    "generate_signals",
    # Technical indicators
    "sma",
    "ema",
    "crossover",
    "rsi",
    "macd",
    "bollinger_bands",
    "stochastic",
    # Performance metrics
    "daily_returns",
    "cagr",
    "sharpe",
    "max_drawdown",
    "calmar",
    "hit_rate",
    "avg_win_loss",
    "volatility",
    "sortino",
    "var",
    "cvar",
    # Backtesting functions
    "run_crossover_backtest",
    "print_backtest_report",
    "run_quick_backtest",
    # Plotting functions
    "plot_equity",
    "plot_drawdown",
    "plot_price_signals",
    "save_figure",
    "plot_equity_with_metrics",
    "plot_comprehensive_backtest",
    "create_plot_style",
    # Optimization functions
    "grid_search",
    "optimize_strategy",
    "print_optimization_results",
    "save_optimization_results",
    # Walk-forward analysis
    "walkforward_crossover",
    "print_walkforward_results",
]
