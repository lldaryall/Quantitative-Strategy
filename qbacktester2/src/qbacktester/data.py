"""Data loading and management utilities for qbacktester."""

import os
import time
from datetime import date, datetime
from typing import List, Optional, Union

import pandas as pd
import yfinance as yf
from rich.console import Console


class DataError(Exception):
    """Custom exception for data-related errors."""

    pass


class DataLoader:
    """
    Data loader for fetching and managing financial data.

    This class provides methods to load data from various sources
    including Yahoo Finance and local files, with local caching support.
    """

    def __init__(self, cache_dir: str = "data") -> None:
        """
        Initialize the data loader.

        Args:
            cache_dir: Directory to store cached data files
        """
        self.console = Console()
        self.cache_dir = cache_dir
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_price_data(
        self, symbol: str, start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get price data with local caching support.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, 5m, etc.)

        Returns:
            DataFrame with OHLCV data, sorted by date

        Raises:
            DataError: If data cannot be loaded or is invalid
        """
        # Validate inputs
        self._validate_date_range(start, end)

        # Check cache first
        cache_file = self._get_cache_file_path(symbol, interval, start, end)
        if os.path.exists(cache_file):
            self.console.print(f"[blue]Loading cached data for {symbol}[/blue]")
            try:
                data = pd.read_parquet(cache_file)
                self.console.print(
                    f"[green]Loaded {len(data)} cached records for {symbol}[/green]"
                )
                return data
            except Exception as e:
                self.console.print(
                    f"[yellow]Cache file corrupted, downloading fresh data: {e}[/yellow]"
                )

        # Download data with retry logic
        data = self._download_with_retry(symbol, start, end, interval)

        # Validate and process data
        data = self._process_data(data, symbol)

        # Save to cache
        try:
            data.to_parquet(cache_file)
            self.console.print(
                f"[green]Cached data for {symbol} to {cache_file}[/green]"
            )
        except Exception as e:
            self.console.print(f"[yellow]Failed to cache data: {e}[/yellow]")

        return data

    def _validate_date_range(self, start: str, end: str) -> None:
        """Validate date range inputs."""
        try:
            start_date = datetime.strptime(start, "%Y-%m-%d").date()
            end_date = datetime.strptime(end, "%Y-%m-%d").date()

            if start_date >= end_date:
                raise DataError(f"Start date ({start}) must be before end date ({end})")

            if start_date > date.today():
                raise DataError(f"Start date ({start}) cannot be in the future")

        except ValueError as e:
            raise DataError(f"Invalid date format. Use YYYY-MM-DD: {e}")

    def _get_cache_file_path(
        self, symbol: str, interval: str, start: str, end: str
    ) -> str:
        """Generate cache file path."""
        filename = f"{symbol}_{interval}_{start}_{end}.parquet"
        return os.path.join(self.cache_dir, filename)

    def _download_with_retry(
        self, symbol: str, start: str, end: str, interval: str, max_retries: int = 3
    ) -> pd.DataFrame:
        """Download data with retry logic and exponential backoff."""
        last_exception = None

        for attempt in range(max_retries):
            try:
                self.console.print(
                    f"[cyan]Downloading data for {symbol} (attempt {attempt + 1}/{max_retries})[/cyan]"
                )

                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start, end=end, interval=interval)

                if data.empty:
                    raise DataError(f"No data found for symbol: {symbol}")

                return data

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    self.console.print(
                        f"[yellow]Download failed, retrying in {wait_time}s: {e}[/yellow]"
                    )
                    time.sleep(wait_time)
                else:
                    self.console.print(
                        f"[red]All download attempts failed for {symbol}[/red]"
                    )

        raise DataError(
            f"Failed to download data for {symbol} after {max_retries} attempts: {last_exception}"
        )

    def _process_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process and validate downloaded data."""
        if data.empty:
            raise DataError(f"Empty dataset for symbol: {symbol}")

        # Ensure column names are lowercase
        data.columns = data.columns.str.lower()

        # Validate required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataError(f"Missing required columns for {symbol}: {missing_columns}")

        # Sort by date
        data = data.sort_index()

        # Drop rows with all NaN values
        data = data.dropna(how="all")

        # Check for remaining NaN values in OHLCV
        ohlcv_columns = ["open", "high", "low", "close", "volume"]
        nan_counts = data[ohlcv_columns].isnull().sum()
        if nan_counts.any():
            self.console.print(
                f"[yellow]Warning: {symbol} has NaN values: {nan_counts[nan_counts > 0].to_dict()}[/yellow]"
            )
            # Forward fill then backward fill for price data
            data[["open", "high", "low", "close"]] = (
                data[["open", "high", "low", "close"]].ffill().bfill()
            )
            # Fill volume with 0
            data["volume"] = data["volume"].fillna(0)

        self.console.print(f"[green]Processed {len(data)} records for {symbol}[/green]")
        return data

    def load_yahoo_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
    ) -> pd.DataFrame:
        """
        Load data from Yahoo Finance (legacy method for backward compatibility).

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert period to start/end dates if needed
            if start_date and end_date:
                return self.get_price_data(symbol, start_date, end_date)
            else:
                # Use period-based download
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)

                if data.empty:
                    self.console.print(f"[red]No data found for symbol: {symbol}[/red]")
                    return pd.DataFrame()

                return self._process_data(data, symbol)

        except Exception as e:
            self.console.print(f"[red]Error loading data for {symbol}: {e}[/red]")
            return pd.DataFrame()

    def load_csv_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with OHLCV data
        """
        try:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)

            # Ensure required columns exist
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]

            if missing_columns:
                self.console.print(
                    f"[red]Missing required columns: {missing_columns}[/red]"
                )
                return pd.DataFrame()

            self.console.print(
                f"[green]Loaded {len(data)} records from {filepath}[/green]"
            )
            return data

        except Exception as e:
            self.console.print(f"[red]Error loading CSV data: {e}[/red]")
            return pd.DataFrame()

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required structure for backtesting.

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ["open", "high", "low", "close", "volume"]

        if data.empty:
            self.console.print("[red]Data is empty[/red]")
            return False

        if not all(col in data.columns for col in required_columns):
            self.console.print("[red]Missing required OHLCV columns[/red]")
            return False

        if not isinstance(data.index, pd.DatetimeIndex):
            self.console.print("[red]Index must be datetime[/red]")
            return False

        if data.isnull().any().any():
            self.console.print("[yellow]Warning: Data contains null values[/yellow]")

        return True
