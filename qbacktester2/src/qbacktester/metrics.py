"""Performance metrics calculation for qbacktester.

This module provides vectorized, numerically stable functions for calculating
various performance metrics from equity curves and return series.
"""

import warnings
from typing import Tuple, Union

import numpy as np
import pandas as pd


def daily_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate daily returns from equity curve.

    Args:
        equity_curve: Series of equity values (portfolio values over time)

    Returns:
        Series of daily returns (percentage changes)

    Raises:
        ValueError: If equity_curve is empty or contains non-positive values
    """
    if len(equity_curve) == 0:
        raise ValueError("equity_curve cannot be empty")

    if (equity_curve <= 0).any():
        raise ValueError("equity_curve must contain only positive values")

    # Calculate percentage change, handling potential division by zero
    returns = equity_curve.pct_change()

    # Fill first value with 0 (no return on first day)
    returns.iloc[0] = 0.0

    return returns


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).

    Args:
        equity_curve: Series of equity values
        periods_per_year: Number of periods per year (default: 252 for daily data)

    Returns:
        CAGR as a decimal (e.g., 0.10 for 10%)

    Raises:
        ValueError: If equity_curve is empty or contains non-positive values
    """
    if len(equity_curve) == 0:
        raise ValueError("equity_curve cannot be empty")

    if (equity_curve <= 0).any():
        raise ValueError("equity_curve must contain only positive values")

    if len(equity_curve) < 2:
        return 0.0

    # Calculate total return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # Calculate number of years
    n_periods = len(equity_curve) - 1
    n_years = n_periods / periods_per_year

    if n_years <= 0:
        return 0.0

    # Calculate CAGR: (1 + total_return)^(1/n_years) - 1
    cagr_value = (1 + total_return) ** (1 / n_years) - 1

    return cagr_value


def sharpe(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free: Risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily data)

    Returns:
        Sharpe ratio (annualized)

    Raises:
        ValueError: If returns is empty
    """
    if len(returns) == 0:
        raise ValueError("returns cannot be empty")

    # Remove NaN values
    clean_returns = returns.dropna()

    if len(clean_returns) == 0:
        return 0.0

    # Calculate excess returns
    excess_returns = clean_returns - (risk_free / periods_per_year)

    # Calculate mean and standard deviation
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()

    # Handle case where standard deviation is zero
    if std_excess_return == 0:
        return 0.0 if mean_excess_return == 0 else np.inf

    # Annualize the ratio
    sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(periods_per_year)

    return sharpe_ratio


def max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and its dates.

    Args:
        equity_curve: Series of equity values

    Returns:
        Tuple of (max_drawdown_magnitude, peak_date, trough_date)
        - max_drawdown_magnitude: Maximum drawdown as a positive decimal
        - peak_date: Date of the peak before maximum drawdown
        - trough_date: Date of the trough (lowest point during drawdown)

    Raises:
        ValueError: If equity_curve is empty
    """
    if len(equity_curve) == 0:
        raise ValueError("equity_curve cannot be empty")

    # Calculate running maximum (peak values)
    running_max = equity_curve.expanding().max()

    # Calculate drawdown as percentage from peak
    drawdown = (equity_curve - running_max) / running_max

    # Find maximum drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_value = abs(drawdown.iloc[drawdown.argmin()])

    # Find peak date (last occurrence of the peak value before max drawdown)
    peak_value = running_max.loc[max_dd_idx]
    peak_candidates = equity_curve[equity_curve == peak_value]
    peak_date = peak_candidates[peak_candidates.index <= max_dd_idx].index[-1]

    return max_dd_value, peak_date, max_dd_idx


def calmar(equity_curve: pd.Series) -> float:
    """
    Calculate Calmar ratio (CAGR / Max Drawdown).

    Args:
        equity_curve: Series of equity values

    Returns:
        Calmar ratio

    Raises:
        ValueError: If equity_curve is empty or contains non-positive values
    """
    if len(equity_curve) == 0:
        raise ValueError("equity_curve cannot be empty")

    if (equity_curve <= 0).any():
        raise ValueError("equity_curve must contain only positive values")

    # Calculate CAGR
    cagr_value = cagr(equity_curve)

    # Calculate maximum drawdown
    max_dd, _, _ = max_drawdown(equity_curve)

    # Handle case where max drawdown is zero
    if max_dd == 0:
        return np.inf if cagr_value >= 0 else 0.0

    return cagr_value / max_dd


def hit_rate(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Calculate hit rate (percentage of positive returns above threshold).

    Args:
        returns: Series of returns
        threshold: Minimum return to count as a "hit" (default: 0.0)

    Returns:
        Hit rate as a decimal (e.g., 0.60 for 60%)

    Raises:
        ValueError: If returns is empty
    """
    if len(returns) == 0:
        raise ValueError("returns cannot be empty")

    # Remove NaN values
    clean_returns = returns.dropna()

    if len(clean_returns) == 0:
        return 0.0

    # Calculate hit rate (exclude first return which is always 0)
    if len(clean_returns) > 1:
        # Exclude first return (which is always 0) for hit rate calculation
        returns_for_hit_rate = clean_returns.iloc[1:]
        hits = (returns_for_hit_rate > threshold).sum()
        total = len(returns_for_hit_rate)
    else:
        hits = 0
        total = 1

    return hits / total


def avg_win_loss(returns: pd.Series) -> Tuple[float, float]:
    """
    Calculate average win and average loss.

    Args:
        returns: Series of returns

    Returns:
        Tuple of (average_win, average_loss)
        - average_win: Average of positive returns
        - average_loss: Average of negative returns (as positive value)

    Raises:
        ValueError: If returns is empty
    """
    if len(returns) == 0:
        raise ValueError("returns cannot be empty")

    # Remove NaN values
    clean_returns = returns.dropna()

    if len(clean_returns) == 0:
        return 0.0, 0.0

    # Separate wins and losses
    wins = clean_returns[clean_returns > 0]
    losses = clean_returns[clean_returns < 0]

    # Calculate averages
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0

    return avg_win, avg_loss


def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year (default: 252 for daily data)

    Returns:
        Annualized volatility as a decimal

    Raises:
        ValueError: If returns is empty
    """
    if len(returns) == 0:
        raise ValueError("returns cannot be empty")

    # Remove NaN values
    clean_returns = returns.dropna()

    if len(clean_returns) == 0:
        return 0.0

    # Calculate standard deviation and annualize
    vol = clean_returns.std() * np.sqrt(periods_per_year)

    return vol


def sortino(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (downside deviation version of Sharpe).

    Args:
        returns: Series of returns
        risk_free: Risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily data)

    Returns:
        Sortino ratio (annualized)

    Raises:
        ValueError: If returns is empty
    """
    if len(returns) == 0:
        raise ValueError("returns cannot be empty")

    # Remove NaN values
    clean_returns = returns.dropna()

    if len(clean_returns) == 0:
        return 0.0

    # Calculate excess returns
    excess_returns = clean_returns - (risk_free / periods_per_year)

    # Calculate mean excess return
    mean_excess_return = excess_returns.mean()

    # Calculate downside deviation (standard deviation of negative returns only)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return np.inf if mean_excess_return > 0 else 0.0

    downside_deviation = downside_returns.std()

    if downside_deviation == 0:
        return 0.0 if mean_excess_return == 0 else np.inf

    # Annualize the ratio
    sortino_ratio = (mean_excess_return / downside_deviation) * np.sqrt(
        periods_per_year
    )

    return sortino_ratio


def var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Series of returns
        confidence_level: Confidence level (default: 0.05 for 5% VaR)

    Returns:
        VaR as a decimal (negative value representing loss)

    Raises:
        ValueError: If returns is empty or confidence_level is invalid
    """
    if len(returns) == 0:
        raise ValueError("returns cannot be empty")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    # Remove NaN values
    clean_returns = returns.dropna()

    if len(clean_returns) == 0:
        return 0.0

    # Calculate VaR (negative of the percentile)
    var_value = -np.percentile(clean_returns, confidence_level * 100)

    return var_value


def cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) or Expected Shortfall.

    Args:
        returns: Series of returns
        confidence_level: Confidence level (default: 0.05 for 5% CVaR)

    Returns:
        CVaR as a decimal (negative value representing expected loss)

    Raises:
        ValueError: If returns is empty or confidence_level is invalid
    """
    if len(returns) == 0:
        raise ValueError("returns cannot be empty")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    # Remove NaN values
    clean_returns = returns.dropna()

    if len(clean_returns) == 0:
        return 0.0

    # Calculate VaR threshold
    var_threshold = np.percentile(clean_returns, confidence_level * 100)

    # Calculate CVaR as mean of returns below VaR threshold
    tail_returns = clean_returns[clean_returns <= var_threshold]

    if len(tail_returns) == 0:
        return 0.0

    cvar_value = -tail_returns.mean()

    return cvar_value
