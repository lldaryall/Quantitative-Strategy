"""Tests for the data module."""

import os
import shutil
import tempfile
from datetime import date, datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from qbacktester.data import DataError, DataLoader


class TestDataLoader:
    """Test cases for the DataLoader class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_loader = DataLoader(cache_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test data loader initialization."""
        loader = DataLoader()
        assert loader.cache_dir == "data"
        assert os.path.exists("data")

    def test_initialization_custom_cache_dir(self):
        """Test data loader initialization with custom cache directory."""
        custom_dir = os.path.join(self.temp_dir, "custom_cache")
        loader = DataLoader(cache_dir=custom_dir)
        assert loader.cache_dir == custom_dir
        assert os.path.exists(custom_dir)

    def test_validate_date_range_valid(self):
        """Test date range validation with valid dates."""
        # Should not raise any exception
        self.data_loader._validate_date_range("2020-01-01", "2020-12-31")

    def test_validate_date_range_invalid_format(self):
        """Test date range validation with invalid date format."""
        with pytest.raises(DataError, match="Invalid date format"):
            self.data_loader._validate_date_range("2020/01/01", "2020-12-31")

    def test_validate_date_range_start_after_end(self):
        """Test date range validation when start is after end."""
        with pytest.raises(DataError, match="Start date.*must be before end date"):
            self.data_loader._validate_date_range("2020-12-31", "2020-01-01")

    def test_validate_date_range_future_start(self):
        """Test date range validation with future start date."""
        future_date = (date.today().replace(year=date.today().year + 1)).strftime(
            "%Y-%m-%d"
        )
        future_end_date = (date.today().replace(year=date.today().year + 2)).strftime(
            "%Y-%m-%d"
        )
        with pytest.raises(DataError, match="Start date.*cannot be in the future"):
            self.data_loader._validate_date_range(future_date, future_end_date)

    def test_get_cache_file_path(self):
        """Test cache file path generation."""
        path = self.data_loader._get_cache_file_path(
            "AAPL", "1d", "2020-01-01", "2020-12-31"
        )
        expected = os.path.join(self.temp_dir, "AAPL_1d_2020-01-01_2020-12-31.parquet")
        assert path == expected

    def test_process_data_valid(self):
        """Test data processing with valid data."""
        # Create sample data
        dates = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
        data = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [105, 106, 107, 108, 109],
                "Low": [95, 96, 97, 98, 99],
                "Close": [102, 103, 104, 105, 106],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=dates,
        )

        processed = self.data_loader._process_data(data, "AAPL")

        # Check that columns are lowercase
        assert all(
            col in processed.columns
            for col in ["open", "high", "low", "close", "volume"]
        )
        assert len(processed) == 5
        assert isinstance(processed.index, pd.DatetimeIndex)

    def test_process_data_empty(self):
        """Test data processing with empty data."""
        empty_data = pd.DataFrame()
        with pytest.raises(DataError, match="Empty dataset for symbol"):
            self.data_loader._process_data(empty_data, "AAPL")

    def test_process_data_missing_columns(self):
        """Test data processing with missing columns."""
        dates = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
        data = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [105, 106, 107, 108, 109],
                # Missing Low, Close, Volume
            },
            index=dates,
        )

        with pytest.raises(DataError, match="Missing required columns"):
            self.data_loader._process_data(data, "AAPL")

    def test_process_data_with_nan_values(self):
        """Test data processing with NaN values."""
        dates = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
        data = pd.DataFrame(
            {
                "Open": [100, np.nan, 102, 103, 104],
                "High": [105, 106, 107, 108, 109],
                "Low": [95, 96, 97, 98, 99],
                "Close": [102, 103, np.nan, 105, 106],
                "Volume": [1000, 1100, 1200, np.nan, 1400],
            },
            index=dates,
        )

        processed = self.data_loader._process_data(data, "AAPL")

        # Check that NaN values are handled
        assert not processed[["open", "high", "low", "close"]].isnull().any().any()
        assert processed["volume"].isnull().sum() == 0  # Volume NaN filled with 0

    @patch("qbacktester.data.yf.Ticker")
    def test_download_with_retry_success(self, mock_ticker_class):
        """Test successful data download with retry logic."""
        # Mock successful download
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        dates = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [105, 106, 107, 108, 109],
                "Low": [95, 96, 97, 98, 99],
                "Close": [102, 103, 104, 105, 106],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=dates,
        )
        mock_ticker.history.return_value = mock_data

        result = self.data_loader._download_with_retry(
            "AAPL", "2020-01-01", "2020-01-05", "1d"
        )

        assert len(result) == 5
        mock_ticker.history.assert_called_once_with(
            start="2020-01-01", end="2020-01-05", interval="1d"
        )

    @patch("qbacktester.data.yf.Ticker")
    def test_download_with_retry_empty_data(self, mock_ticker_class):
        """Test download retry with empty data."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()

        with pytest.raises(DataError, match="No data found for symbol"):
            self.data_loader._download_with_retry(
                "INVALID", "2020-01-01", "2020-01-05", "1d"
            )

    @patch("qbacktester.data.yf.Ticker")
    def test_download_with_retry_failure(self, mock_ticker_class):
        """Test download retry with repeated failures."""
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.side_effect = Exception("Network error")

        with pytest.raises(
            DataError, match="Failed to download data.*after 3 attempts"
        ):
            self.data_loader._download_with_retry(
                "AAPL", "2020-01-01", "2020-01-05", "1d"
            )

    @patch("qbacktester.data.yf.Ticker")
    def test_get_price_data_success(self, mock_ticker_class):
        """Test successful get_price_data call."""
        # Mock successful download
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker

        dates = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [105, 106, 107, 108, 109],
                "Low": [95, 96, 97, 98, 99],
                "Close": [102, 103, 104, 105, 106],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=dates,
        )
        mock_ticker.history.return_value = mock_data

        result = self.data_loader.get_price_data("AAPL", "2020-01-01", "2020-01-05")

        assert len(result) == 5
        assert all(
            col in result.columns for col in ["open", "high", "low", "close", "volume"]
        )
        assert isinstance(result.index, pd.DatetimeIndex)

        # Check that cache file was created
        cache_file = self.data_loader._get_cache_file_path(
            "AAPL", "1d", "2020-01-01", "2020-01-05"
        )
        assert os.path.exists(cache_file)

    @patch("qbacktester.data.yf.Ticker")
    def test_get_price_data_from_cache(self, mock_ticker_class):
        """Test get_price_data loading from cache."""
        # First, create a cache file
        dates = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
        cache_data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [102, 103, 104, 105, 106],
                "volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=dates,
        )

        cache_file = self.data_loader._get_cache_file_path(
            "AAPL", "1d", "2020-01-01", "2020-01-05"
        )
        cache_data.to_parquet(cache_file)

        # Now call get_price_data - should load from cache
        result = self.data_loader.get_price_data("AAPL", "2020-01-01", "2020-01-05")

        assert len(result) == 5
        assert all(
            col in result.columns for col in ["open", "high", "low", "close", "volume"]
        )

        # Verify that yfinance was not called
        mock_ticker_class.assert_not_called()

    def test_get_price_data_invalid_dates(self):
        """Test get_price_data with invalid date range."""
        with pytest.raises(DataError, match="Start date.*must be before end date"):
            self.data_loader.get_price_data("AAPL", "2020-12-31", "2020-01-01")

    def test_load_csv_data_success(self):
        """Test successful CSV data loading."""
        # Create a temporary CSV file
        dates = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
        csv_data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [102, 103, 104, 105, 106],
                "volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=dates,
        )

        csv_file = os.path.join(self.temp_dir, "test_data.csv")
        csv_data.to_csv(csv_file)

        result = self.data_loader.load_csv_data(csv_file)

        assert len(result) == 5
        assert all(
            col in result.columns for col in ["open", "high", "low", "close", "volume"]
        )

    def test_load_csv_data_missing_columns(self):
        """Test CSV data loading with missing columns."""
        # Create a CSV file with missing columns
        dates = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
        csv_data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                # Missing low, close, volume
            },
            index=dates,
        )

        csv_file = os.path.join(self.temp_dir, "test_data.csv")
        csv_data.to_csv(csv_file)

        result = self.data_loader.load_csv_data(csv_file)

        assert result.empty

    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        dates = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
        data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [102, 103, 104, 105, 106],
                "volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=dates,
        )

        assert self.data_loader.validate_data(data) is True

    def test_validate_data_empty(self):
        """Test data validation with empty data."""
        empty_data = pd.DataFrame()
        assert self.data_loader.validate_data(empty_data) is False

    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        dates = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
        data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                # Missing low, close, volume
            },
            index=dates,
        )

        assert self.data_loader.validate_data(data) is False

    def test_validate_data_non_datetime_index(self):
        """Test data validation with non-datetime index."""
        data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [102, 103, 104, 105, 106],
                "volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=[0, 1, 2, 3, 4],
        )

        assert self.data_loader.validate_data(data) is False


class TestDataError:
    """Test cases for the DataError exception."""

    def test_data_error_creation(self):
        """Test DataError exception creation."""
        error = DataError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
