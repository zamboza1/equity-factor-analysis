"""
Tests for returns computation and ReturnSeries ADT.

Tests cover:
- Return computation (log and simple)
- ReturnSeries invariants (sorted dates, equal lengths, no duplicates)
- Series alignment
- Edge cases (empty series, missing data, etc.)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backend.entities import Asset, ReturnSeries
from backend.analytics.returns import compute_returns, create_return_series, align_series


class TestComputeReturns:
    """Tests for compute_returns function."""

    def test_log_returns_basic(self):
        """Test log returns on simple price series."""
        prices = pd.Series([100, 110, 121, 110], index=pd.date_range("2023-01-01", periods=4))
        returns = compute_returns(prices, method="log")

        # First value should be NaN
        assert pd.isna(returns.iloc[0])

        # Log return from 100 to 110 = ln(110/100) ≈ 0.0953
        assert abs(returns.iloc[1] - np.log(110/100)) < 1e-6

    def test_simple_returns_basic(self):
        """Test simple returns on simple price series."""
        prices = pd.Series([100, 110, 121, 110], index=pd.date_range("2023-01-01", periods=4))
        returns = compute_returns(prices, method="simple")

        # First value should be NaN
        assert pd.isna(returns.iloc[0])

        # Simple return from 100 to 110 = (110/100) - 1 = 0.1
        assert abs(returns.iloc[1] - 0.1) < 1e-6

    def test_returns_negative_prices_raises(self):
        """Test that negative prices raise ValueError."""
        prices = pd.Series([100, -50], index=pd.date_range("2023-01-01", periods=2))
        with pytest.raises(ValueError, match="prices must be positive"):
            compute_returns(prices)

    def test_returns_zero_prices_raises(self):
        """Test that zero prices raise ValueError."""
        prices = pd.Series([100, 0], index=pd.date_range("2023-01-01", periods=2))
        with pytest.raises(ValueError, match="prices must be positive"):
            compute_returns(prices)

    def test_returns_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        prices = pd.Series([100, 110], index=pd.date_range("2023-01-01", periods=2))
        with pytest.raises(ValueError, match="method must be"):
            compute_returns(prices, method="invalid")


class TestReturnSeries:
    """Tests for ReturnSeries ADT."""

    def test_basic_creation(self):
        """Test creating a ReturnSeries from dates and returns."""
        dates = pd.date_range("2023-01-01", periods=5)
        returns = pd.Series([0.01, 0.02, -0.01, 0.03], index=dates[1:])
        # Add NaN for first date to match compute_returns behavior
        returns = pd.concat([pd.Series([np.nan], index=[dates[0]]), returns])

        series = ReturnSeries(dates, returns)
        assert len(series) == 5
        assert len(series.dates) == 5
        assert len(series.returns) == 5

    def test_unsorted_dates_get_sorted(self):
        """Test that unsorted dates are automatically sorted."""
        dates = pd.date_range("2023-01-01", periods=5)
        returns = pd.Series([0.01, 0.02, -0.01, 0.03], index=dates[1:])
        returns = pd.concat([pd.Series([np.nan], index=[dates[0]]), returns])

        # Reverse dates
        reversed_dates = dates[::-1]
        series = ReturnSeries(reversed_dates, returns)

        # Dates should be sorted
        assert series.dates.is_monotonic_increasing

    def test_duplicate_dates_get_deduplicated(self):
        """Test that duplicate dates are removed (keeping first)."""
        dates = pd.date_range("2023-01-01", periods=5)
        # Add duplicate
        dates_with_dup = dates.tolist() + [dates[2]]
        dates_with_dup = pd.DatetimeIndex(dates_with_dup)

        # Need 6 return values for 6 dates (one duplicate)
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.05, 0.04], index=dates_with_dup)

        series = ReturnSeries(dates_with_dup, returns)
        assert len(series) == 5  # Duplicate removed
        assert not series.dates.has_duplicates

    def test_returns_with_asset(self):
        """Test ReturnSeries with associated Asset."""
        asset = Asset(ticker="AAPL", name="Apple Inc.", asset_type="equity")
        dates = pd.date_range("2023-01-01", periods=5)
        returns = pd.Series([0.01, 0.02, -0.01, 0.03], index=dates[1:])
        returns = pd.concat([pd.Series([np.nan], index=[dates[0]]), returns])

        series = ReturnSeries(dates, returns, asset=asset)
        assert series.asset == asset
        assert series.asset.ticker == "AAPL"

    def test_empty_series_raises(self):
        """Test that empty series raises error."""
        dates = pd.DatetimeIndex([])
        returns = pd.Series([], dtype=float)
        with pytest.raises(ValueError):
            ReturnSeries(dates, returns)

    def test_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise error."""
        dates = pd.date_range("2023-01-01", periods=5)
        returns = pd.Series([0.01, 0.02], index=dates[:2])
        with pytest.raises(ValueError):
            ReturnSeries(dates, returns)


class TestCreateReturnSeries:
    """Tests for create_return_series convenience function."""

    def test_create_from_prices(self):
        """Test creating ReturnSeries from price data."""
        prices = pd.Series([100, 110, 121, 110], index=pd.date_range("2023-01-01", periods=4))
        series = create_return_series(prices)

        # First return should be dropped (NaN)
        assert len(series) == 3
        assert series.dates[0] == pd.Timestamp("2023-01-02")

    def test_create_with_asset(self):
        """Test creating ReturnSeries with Asset."""
        asset = Asset(ticker="MSFT", name="Microsoft", asset_type="equity")
        prices = pd.Series([100, 110], index=pd.date_range("2023-01-01", periods=2))
        series = create_return_series(prices, asset=asset)

        assert series.asset == asset

    def test_empty_prices_raises(self):
        """Test that empty prices raise error."""
        prices = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        with pytest.raises(ValueError):
            create_return_series(prices)


class TestAlignSeries:
    """Tests for align_series function."""

    def test_align_two_series(self):
        """Test aligning two series with overlapping dates."""
        dates1 = pd.date_range("2023-01-01", periods=5)  # 01-01 to 01-05
        dates2 = pd.date_range("2023-01-03", periods=5)  # 01-03 to 01-07

        returns1 = pd.Series([0.01, 0.02, -0.01, 0.03, 0.02], index=dates1)
        returns2 = pd.Series([0.02, -0.01, 0.03, 0.01, 0.02], index=dates2)

        series1 = ReturnSeries(dates1, returns1)
        series2 = ReturnSeries(dates2, returns2)

        aligned = align_series([series1, series2])

        # Common dates: 2023-01-03 to 2023-01-05 (3 dates)
        assert len(aligned) == 2
        assert len(aligned[0].dates) == 3
        assert len(aligned[1].dates) == 3
        assert (aligned[0].dates == aligned[1].dates).all()
        # Verify the common dates
        expected_common = pd.date_range("2023-01-03", periods=3)
        assert (aligned[0].dates == expected_common).all()

    def test_align_three_series(self):
        """Test aligning three series."""
        dates1 = pd.date_range("2023-01-01", periods=5)
        dates2 = pd.date_range("2023-01-02", periods=5)
        dates3 = pd.date_range("2023-01-03", periods=5)

        returns1 = pd.Series([0.01] * 5, index=dates1)
        returns2 = pd.Series([0.02] * 5, index=dates2)
        returns3 = pd.Series([0.03] * 5, index=dates3)

        series1 = ReturnSeries(dates1, returns1)
        series2 = ReturnSeries(dates2, returns2)
        series3 = ReturnSeries(dates3, returns3)

        aligned = align_series([series1, series2, series3])

        # Common dates: 2023-01-03 to 2023-01-05 (3 dates)
        assert len(aligned) == 3
        assert all(len(s.dates) == 3 for s in aligned)

    def test_align_empty_list_raises(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            align_series([])

    def test_align_no_overlap_raises(self):
        """Test that series with no overlap raise error."""
        dates1 = pd.date_range("2023-01-01", periods=3)
        dates2 = pd.date_range("2023-02-01", periods=3)  # No overlap

        returns1 = pd.Series([0.01] * 3, index=dates1)
        returns2 = pd.Series([0.02] * 3, index=dates2)

        series1 = ReturnSeries(dates1, returns1)
        series2 = ReturnSeries(dates2, returns2)

        with pytest.raises(ValueError, match="no common dates"):
            align_series([series1, series2])

    def test_align_single_series(self):
        """Test that single series returns unchanged."""
        dates = pd.date_range("2023-01-01", periods=5)
        returns = pd.Series([0.01] * 5, index=dates)
        series = ReturnSeries(dates, returns)

        aligned = align_series([series])
        assert len(aligned) == 1
        assert aligned[0] is series  # Should be same object (no copy needed)


class TestAsset:
    """Tests for Asset entity."""

    def test_valid_asset(self):
        """Test creating a valid Asset."""
        asset = Asset(ticker="AAPL", name="Apple Inc.", asset_type="equity")
        assert asset.ticker == "AAPL"
        assert asset.name == "Apple Inc."
        assert asset.asset_type == "equity"

    def test_invalid_asset_type_raises(self):
        """Test that invalid asset_type raises error."""
        with pytest.raises(ValueError, match="invalid asset_type"):
            Asset(ticker="AAPL", name="Apple", asset_type="invalid")

    def test_empty_ticker_raises(self):
        """Test that empty ticker raises error."""
        with pytest.raises(ValueError, match="ticker cannot be empty"):
            Asset(ticker="", name="Apple", asset_type="equity")

    def test_empty_name_raises(self):
        """Test that empty name raises error."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Asset(ticker="AAPL", name="", asset_type="equity")

