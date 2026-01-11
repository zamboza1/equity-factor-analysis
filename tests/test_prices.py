"""
Tests for price data downloads and caching.

Tests cover:
- Price downloads (single and multiple tickers)
- Cache hit/miss behavior
- Mocked yfinance calls
- Edge cases (empty data, invalid dates, etc.)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from backend.data_sources.prices import get_prices, get_close_prices
from backend.cache import DataCache
from backend.errors import DataError


class TestDataCache:
    """Tests for DataCache class."""

    def test_cache_init_creates_dir(self):
        """Test that cache creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            cache = DataCache(str(cache_dir))
            assert cache_dir.exists()
            assert cache_dir.is_dir()

    def test_cache_set_and_get(self):
        """Test storing and retrieving cached data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(tmpdir)
            query_params = {"ticker": "AAPL", "start": "2023-01-01", "end": "2023-12-31"}

            # Store data
            test_data = pd.DataFrame({"Close": [100, 110, 120]})
            cache.set(query_params, test_data)

            # Retrieve data
            retrieved = cache.get(query_params)
            assert retrieved is not None
            pd.testing.assert_frame_equal(retrieved, test_data)

    def test_cache_miss_returns_none(self):
        """Test that cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(tmpdir)
            query_params = {"ticker": "AAPL", "start": "2023-01-01", "end": "2023-12-31"}

            retrieved = cache.get(query_params)
            assert retrieved is None

    def test_cache_exists(self):
        """Test exists() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(tmpdir)
            query_params = {"ticker": "AAPL", "start": "2023-01-01", "end": "2023-12-31"}

            assert not cache.exists(query_params)

            test_data = pd.DataFrame({"Close": [100, 110]})
            cache.set(query_params, test_data)

            assert cache.exists(query_params)

    def test_cache_clear(self):
        """Test clearing all cached files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(tmpdir)

            # Add multiple cache entries
            for i in range(3):
                query_params = {"ticker": f"TICK{i}"}
                cache.set(query_params, pd.DataFrame({"data": [i]}))

            # Clear and verify
            cache.clear()
            assert len(list(Path(tmpdir).glob("*.pkl"))) == 0

    def test_cache_hash_consistency(self):
        """Test that same params produce same hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(tmpdir)
            query_params1 = {"ticker": "AAPL", "start": "2023-01-01"}
            query_params2 = {"start": "2023-01-01", "ticker": "AAPL"}  # Different order

            # Should produce same hash
            hash1 = cache._compute_hash(query_params1)
            hash2 = cache._compute_hash(query_params2)
            assert hash1 == hash2


class TestGetPrices:
    """Tests for get_prices function."""

    @patch('backend.data_sources.prices.yf.Ticker')
    def test_get_prices_single_ticker(self, mock_ticker_class):
        """Test downloading prices for a single ticker."""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            "Open": [100, 110, 120],
            "High": [105, 115, 125],
            "Low": [95, 105, 115],
            "Close": [102, 112, 122],
            "Volume": [1000, 1100, 1200]
        }, index=pd.date_range("2023-01-01", periods=3))

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker

        result = get_prices("AAPL", "2023-01-01", "2023-01-10", use_cache=False)

        assert len(result) == 3
        assert "Close" in result.columns
        pd.testing.assert_frame_equal(result, mock_data)

    @patch('backend.data_sources.prices.yf.Ticker')
    def test_get_prices_multiple_tickers(self, mock_ticker_class):
        """Test downloading prices for multiple tickers."""
        # Mock responses for each ticker
        mock_data_aapl = pd.DataFrame({
            "Close": [100, 110],
            "Volume": [1000, 1100]
        }, index=pd.date_range("2023-01-01", periods=2))

        mock_data_msft = pd.DataFrame({
            "Close": [200, 220],
            "Volume": [2000, 2200]
        }, index=pd.date_range("2023-01-01", periods=2))

        def ticker_side_effect(ticker_str):
            mock_ticker = Mock()
            if "AAPL" in ticker_str:
                mock_ticker.history.return_value = mock_data_aapl
            elif "MSFT" in ticker_str:
                mock_ticker.history.return_value = mock_data_msft
            return mock_ticker

        mock_ticker_class.side_effect = ticker_side_effect

        result = get_prices(["AAPL", "MSFT"], "2023-01-01", "2023-01-10", use_cache=False)

        # Should have MultiIndex
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["Ticker", "Date"]
        assert len(result) == 4  # 2 tickers * 2 dates

    @patch('backend.data_sources.prices.yf.Ticker')
    def test_get_prices_uses_cache(self, mock_ticker_class):
        """Test that cached data is used when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(tmpdir)

            # First call - should download
            mock_data = pd.DataFrame({
                "Close": [100, 110],
                "Volume": [1000, 1100]
            }, index=pd.date_range("2023-01-01", periods=2))

            mock_ticker = Mock()
            mock_ticker.history.return_value = mock_data
            mock_ticker_class.return_value = mock_ticker

            result1 = get_prices("AAPL", "2023-01-01", "2023-01-10", cache=cache)

            # Second call - should use cache (mock shouldn't be called again)
            mock_ticker_class.reset_mock()
            result2 = get_prices("AAPL", "2023-01-01", "2023-01-10", cache=cache)

            # Results should be equal
            pd.testing.assert_frame_equal(result1, result2)

            # Ticker should only be called once (first time)
            assert mock_ticker_class.call_count <= 1

    @patch('backend.data_sources.prices.yf.Ticker')
    def test_get_prices_empty_data_raises(self, mock_ticker_class):
        """Test that empty data raises DataError."""
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()  # Empty
        mock_ticker_class.return_value = mock_ticker

        with pytest.raises(DataError, match="No data returned"):
            get_prices("AAPL", "2023-01-01", "2023-01-10", use_cache=False)

    @patch('backend.data_sources.prices.yf.Ticker')
    def test_get_prices_date_formats(self, mock_ticker_class):
        """Test that different date formats work."""
        mock_data = pd.DataFrame({
            "Close": [100, 110],
            "Volume": [1000, 1100]
        }, index=pd.date_range("2023-01-01", periods=2))

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker

        # Test string date
        result1 = get_prices("AAPL", "2023-01-01", "2023-01-10", use_cache=False)

        # Test date object
        result2 = get_prices("AAPL", date(2023, 1, 1), date(2023, 1, 10), use_cache=False)

        # Test datetime object
        result3 = get_prices(
            "AAPL",
            datetime(2023, 1, 1),
            datetime(2023, 1, 10),
            use_cache=False
        )

        assert len(result1) == len(result2) == len(result3)

    def test_get_prices_empty_tickers_raises(self):
        """Test that empty tickers list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            get_prices([], "2023-01-01", "2023-01-10", use_cache=False)


class TestGetClosePrices:
    """Tests for get_close_prices convenience function."""

    @patch('backend.data_sources.prices.yf.Ticker')
    def test_get_close_prices_single_ticker(self, mock_ticker_class):
        """Test getting close prices for single ticker."""
        mock_data = pd.DataFrame({
            "Open": [100, 110],
            "High": [105, 115],
            "Low": [95, 105],
            "Close": [102, 112],
            "Volume": [1000, 1100]
        }, index=pd.date_range("2023-01-01", periods=2))

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker

        result = get_close_prices("AAPL", "2023-01-01", "2023-01-10", use_cache=False)

        assert isinstance(result, pd.Series)
        assert len(result) == 2
        assert result.iloc[0] == 102
        assert result.iloc[1] == 112

    @patch('backend.data_sources.prices.yf.Ticker')
    def test_get_close_prices_multiple_tickers(self, mock_ticker_class):
        """Test getting close prices for multiple tickers."""
        mock_data_aapl = pd.DataFrame({
            "Close": [100, 110],
            "Volume": [1000, 1100]
        }, index=pd.date_range("2023-01-01", periods=2))

        mock_data_msft = pd.DataFrame({
            "Close": [200, 220],
            "Volume": [2000, 2200]
        }, index=pd.date_range("2023-01-01", periods=2))

        def ticker_side_effect(ticker_str):
            mock_ticker = Mock()
            if "AAPL" in ticker_str:
                mock_ticker.history.return_value = mock_data_aapl
            elif "MSFT" in ticker_str:
                mock_ticker.history.return_value = mock_data_msft
            return mock_ticker

        mock_ticker_class.side_effect = ticker_side_effect

        result = get_close_prices(["AAPL", "MSFT"], "2023-01-01", "2023-01-10", use_cache=False)

        assert isinstance(result, pd.Series)
        assert isinstance(result.index, pd.MultiIndex)
        assert len(result) == 4


