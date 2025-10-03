"""Core backtesting engine for qbacktester."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from .data import DataLoader
from .strategy import Strategy, StrategyParams


class Backtester:
    """
    Vectorized backtesting engine for running trading strategies.

    This class handles the execution of trading strategies against historical data
    using fully vectorized operations for optimal performance.
    """

    def __init__(
        self, price_df: pd.DataFrame, signals: pd.Series, params: StrategyParams
    ) -> None:
        """
        Initialize the backtester.

        Args:
            price_df: DataFrame with OHLCV data, must have 'Close' and 'Open' columns
            signals: Series with trading signals (0 or 1)
            params: StrategyParams object with strategy configuration
        """
        self.price_df = price_df.copy()
        self.signals = signals
        self.params = params
        self.console = Console()

        # Validate inputs
        self._validate_inputs()

        # Ensure signals align with price data
        if not self.price_df.index.equals(self.signals.index):
            raise ValueError("price_df and signals must have the same index")

    def _validate_inputs(self) -> None:
        """Validate input data and parameters."""
        required_columns = ["Close"]
        missing_columns = [
            col for col in required_columns if col not in self.price_df.columns
        ]
        if missing_columns:
            raise ValueError(f"price_df must contain columns: {missing_columns}")

        if len(self.price_df) == 0:
            raise ValueError("price_df cannot be empty")

        if len(self.signals) == 0:
            raise ValueError("signals cannot be empty")

        if not self.signals.isin([0, 1]).all():
            raise ValueError("signals must contain only 0 or 1 values")

    def run(self) -> pd.DataFrame:
        """
        Run the backtest using vectorized operations.

        Returns:
            DataFrame with columns:
            - Date index
            - Close: Close prices
            - signal: Trading signals (0 or 1)
            - position: Position size (0 or 1)
            - holdings_value: Value of holdings
            - cash: Cash balance
            - total_equity: Total portfolio value
            - trade_flag: True when trade occurs
        """
        # Initialize result DataFrame
        result = pd.DataFrame(index=self.price_df.index)
        result["Close"] = self.price_df["Close"]
        result["signal"] = self.signals

        # Calculate position changes (vectorized)
        # Position is 1 when signal is 1, 0 otherwise
        result["position"] = self.signals.astype(int)

        # Detect trade entries and exits using position changes
        position_changes = result["position"].diff().fillna(0)
        # First day with signal=1 should also be a trade (initial entry)
        initial_entry = (result["position"] == 1) & (
            result["position"].shift(1).fillna(0) == 0
        )
        result["trade_flag"] = ((position_changes != 0) | initial_entry).astype(bool)

        # Calculate trade prices (use Open if available, otherwise Close)
        trade_price = self.price_df.get("Open", self.price_df["Close"])
        result["trade_price"] = trade_price

        # Calculate transaction costs
        # Cost = (fee_bps + slippage_bps) / 10000 * notional
        total_cost_bps = (self.params.fee_bps + self.params.slippage_bps) / 10000

        # Calculate notional value for each trade
        # When entering position: notional = total_equity
        # When exiting position: notional = holdings_value
        result["notional"] = 0.0
        result["transaction_cost"] = 0.0

        # Initialize portfolio values
        result["holdings_value"] = 0.0
        result["cash"] = self.params.initial_cash
        result["total_equity"] = self.params.initial_cash

        # Vectorized portfolio calculation
        self._calculate_portfolio_vectorized(result, total_cost_bps)

        return result

    def _calculate_portfolio_vectorized(
        self, result: pd.DataFrame, total_cost_bps: float
    ) -> None:
        """
        Calculate portfolio values using vectorized operations.

        Args:
            result: DataFrame to update with portfolio values
            total_cost_bps: Total transaction cost in basis points
        """
        n = len(result)

        # Initialize arrays
        holdings_value = np.zeros(n)
        cash = np.zeros(n)
        total_equity = np.zeros(n)
        notional = np.zeros(n)
        transaction_cost = np.zeros(n)

        # Initial values
        cash[0] = self.params.initial_cash
        total_equity[0] = self.params.initial_cash

        # Vectorized calculation using numpy operations
        position = result["position"].values
        trade_price = result["trade_price"].values
        trade_flag = result["trade_flag"].values

        # Calculate shares held at each point
        shares = np.zeros(n)

        # Use vectorized operations where possible, but some sequential logic is unavoidable
        # due to the nature of portfolio management (each day depends on previous day's state)
        for i in range(n):
            if i == 0:
                # First day: check if we should enter position
                if position[i] == 1:
                    # Enter position on first day
                    shares[i] = cash[i] / trade_price[i]
                    holdings_value[i] = shares[i] * trade_price[i]
                    cash[i] = 0.0
                    notional[i] = self.params.initial_cash
                    if trade_flag[i]:
                        transaction_cost[i] = notional[i] * total_cost_bps
                        cash[i] -= transaction_cost[i]
                else:
                    # Stay in cash
                    shares[i] = 0.0
                    holdings_value[i] = 0.0
                    notional[i] = 0.0
            else:
                # Subsequent days
                prev_cash = cash[i - 1]
                prev_holdings = holdings_value[i - 1]
                prev_position = position[i - 1]
                current_position = position[i]
                current_price = trade_price[i]

                if current_position == 1:
                    # Long position
                    if prev_position == 0:
                        # Entering position: buy with all available cash
                        shares[i] = prev_cash / current_price
                        holdings_value[i] = shares[i] * current_price
                        cash[i] = 0.0
                        notional[i] = prev_cash
                    else:
                        # Already in position: update holdings value
                        shares[i] = shares[i - 1]  # Keep same number of shares
                        holdings_value[i] = shares[i] * current_price
                        cash[i] = prev_cash
                        notional[i] = 0.0
                else:
                    # No position
                    if prev_position == 1:
                        # Exiting position: sell all holdings
                        shares[i] = 0.0
                        cash[i] = shares[i - 1] * current_price
                        holdings_value[i] = 0.0
                        notional[i] = prev_holdings
                    else:
                        # Already in cash
                        shares[i] = 0.0
                        cash[i] = prev_cash
                        holdings_value[i] = 0.0
                        notional[i] = 0.0

                # Calculate transaction cost
                if trade_flag[i]:
                    transaction_cost[i] = notional[i] * total_cost_bps
                    cash[i] -= transaction_cost[i]

            # Total equity = cash + holdings value
            total_equity[i] = cash[i] + holdings_value[i]

        # Update result DataFrame
        result["holdings_value"] = holdings_value
        result["cash"] = cash
        result["total_equity"] = total_equity
        result["notional"] = notional
        result["transaction_cost"] = transaction_cost

    def get_performance_metrics(self, result: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics from backtest results.

        Args:
            result: DataFrame from run() method

        Returns:
            Dictionary with performance metrics
        """
        if len(result) == 0:
            return {}

        # Calculate returns
        equity = result["total_equity"]
        returns = equity.pct_change().fillna(0)

        # Basic metrics
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(result)) - 1

        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Drawdown calculation
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()

        # Trade statistics
        trades = result[result["trade_flag"]]
        num_trades = len(trades)
        total_transaction_costs = result["transaction_cost"].sum()

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "num_trades": num_trades,
            "total_transaction_costs": total_transaction_costs,
            "final_equity": equity.iloc[-1],
        }

    def print_results(self, result: pd.DataFrame) -> None:
        """Print backtest results in a formatted table."""
        metrics = self.get_performance_metrics(result)

        table = Table(title="Backtest Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total Return", f"{metrics.get('total_return', 0):.2%}")
        table.add_row("Annualized Return", f"{metrics.get('annualized_return', 0):.2%}")
        table.add_row("Volatility", f"{metrics.get('volatility', 0):.2%}")
        table.add_row("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        table.add_row("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
        table.add_row("Number of Trades", f"{metrics.get('num_trades', 0)}")
        table.add_row(
            "Transaction Costs", f"${metrics.get('total_transaction_costs', 0):.2f}"
        )
        table.add_row("Final Equity", f"${metrics.get('final_equity', 0):.2f}")

        self.console.print(table)
