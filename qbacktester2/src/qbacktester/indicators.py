"""Technical indicators for qbacktester.

This module provides vectorized technical indicators commonly used in quantitative trading.
All functions are optimized for performance and handle edge cases like NaN values and short series.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd


def sma(df: pd.DataFrame, window: int, col: str = "Close") -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).

    Args:
        df: DataFrame containing price data
        window: Number of periods for the moving average
        col: Column name to calculate SMA for (default: "Close")

    Returns:
        Series containing the SMA values

    Raises:
        ValueError: If window is not positive or column doesn't exist
    """
    if window <= 0:
        raise ValueError(f"Window must be positive, got {window}")

    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    # Calculate SMA using pandas rolling mean with min_periods=1
    return df[col].rolling(window=window, min_periods=1).mean()


def ema(df: pd.DataFrame, span: int, col: str = "Close") -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).

    Args:
        df: DataFrame containing price data
        span: Span for the EMA calculation (equivalent to window)
        col: Column name to calculate EMA for (default: "Close")

    Returns:
        Series containing the EMA values

    Raises:
        ValueError: If span is not positive or column doesn't exist
    """
    if span <= 0:
        raise ValueError(f"Span must be positive, got {span}")

    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    # Calculate EMA using pandas ewm
    return df[col].ewm(span=span, adjust=False).mean()


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """
    Detect crossovers between two series.

    Args:
        fast: Fast moving series
        slow: Slow moving series

    Returns:
        Series with values:
        - 1: when fast crosses above slow on this bar
        - -1: when fast crosses below slow on this bar
        - 0: otherwise

    Raises:
        ValueError: If series have different indices or are empty
    """
    if len(fast) == 0 or len(slow) == 0:
        raise ValueError("Input series cannot be empty")

    if not fast.index.equals(slow.index):
        raise ValueError("Fast and slow series must have the same index")

    # Handle case where both series are identical (no crossovers possible)
    if fast.equals(slow):
        return pd.Series(0, index=fast.index, name="Crossover")

    # Calculate differences
    diff = fast - slow
    diff_prev = diff.shift(1)

    # Vectorized crossover detection
    # Fast crosses above slow: diff > 0 and diff_prev <= 0
    cross_above = (diff > 0) & (diff_prev <= 0)

    # Fast crosses below slow: diff < 0 and diff_prev >= 0
    cross_below = (diff < 0) & (diff_prev >= 0)

    # Create result series
    result = pd.Series(0, index=fast.index, name="Crossover")
    result[cross_above] = 1
    result[cross_below] = -1

    # Handle NaN values in the first position
    result.iloc[0] = 0

    return result


def rsi(df: pd.DataFrame, window: int = 14, col: str = "Close") -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        df: DataFrame containing price data
        window: Number of periods for RSI calculation (default: 14)
        col: Column name to calculate RSI for (default: "Close")

    Returns:
        Series containing RSI values (0-100)

    Raises:
        ValueError: If window is not positive or column doesn't exist
    """
    if window <= 0:
        raise ValueError(f"Window must be positive, got {window}")

    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    # Calculate price changes
    delta = df[col].diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses using EMA
    avg_gains = gains.ewm(span=window, adjust=False).mean()
    avg_losses = losses.ewm(span=window, adjust=False).mean()

    # Calculate RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    return rsi


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = "Close",
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        df: DataFrame containing price data
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)
        col: Column name to calculate MACD for (default: "Close")

    Returns:
        DataFrame with columns: 'MACD', 'Signal', 'Histogram'

    Raises:
        ValueError: If periods are not positive or column doesn't exist
    """
    if any(x <= 0 for x in [fast, slow, signal]):
        raise ValueError(
            f"All periods must be positive, got fast={fast}, slow={slow}, signal={signal}"
        )

    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    # Calculate EMAs
    ema_fast = ema(df, fast, col)
    ema_slow = ema(df, slow, col)

    # Calculate MACD line
    macd_line = ema_fast - ema_slow

    # Calculate signal line (EMA of MACD)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    # Calculate histogram
    histogram = macd_line - signal_line

    # Create result DataFrame
    result = pd.DataFrame(
        {"MACD": macd_line, "Signal": signal_line, "Histogram": histogram},
        index=df.index,
    )

    return result


def bollinger_bands(
    df: pd.DataFrame, window: int = 20, std_dev: float = 2.0, col: str = "Close"
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Args:
        df: DataFrame containing price data
        window: Number of periods for the moving average (default: 20)
        std_dev: Number of standard deviations (default: 2.0)
        col: Column name to calculate Bollinger Bands for (default: "Close")

    Returns:
        DataFrame with columns: 'Upper', 'Middle', 'Lower'

    Raises:
        ValueError: If window is not positive or column doesn't exist
    """
    if window <= 0:
        raise ValueError(f"Window must be positive, got {window}")

    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    # Calculate middle band (SMA)
    middle = sma(df, window, col)

    # Calculate standard deviation
    rolling_std = df[col].rolling(window=window, min_periods=1).std()

    # Calculate upper and lower bands
    upper = middle + (rolling_std * std_dev)
    lower = middle - (rolling_std * std_dev)

    # Create result DataFrame
    result = pd.DataFrame(
        {"Upper": upper, "Middle": middle, "Lower": lower}, index=df.index
    )

    return result


def stochastic(
    df: pd.DataFrame, k_window: int = 14, d_window: int = 3, col: str = "Close"
) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator.

    Args:
        df: DataFrame containing OHLC data
        k_window: Period for %K calculation (default: 14)
        d_window: Period for %D calculation (default: 3)
        col: Column name for close price (default: "Close")

    Returns:
        DataFrame with columns: 'K', 'D'

    Raises:
        ValueError: If windows are not positive or required columns don't exist
    """
    if any(x <= 0 for x in [k_window, d_window]):
        raise ValueError(
            f"All windows must be positive, got k_window={k_window}, d_window={d_window}"
        )

    required_cols = ["High", "Low", col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Calculate highest high and lowest low over k_window
    highest_high = df["High"].rolling(window=k_window, min_periods=1).max()
    lowest_low = df["Low"].rolling(window=k_window, min_periods=1).min()

    # Calculate %K
    k_percent = 100 * (df[col] - lowest_low) / (highest_high - lowest_low)

    # Calculate %D (SMA of %K)
    d_percent = k_percent.rolling(window=d_window, min_periods=1).mean()

    # Create result DataFrame
    result = pd.DataFrame({"K": k_percent, "D": d_percent}, index=df.index)

    return result
