"""
Functions for computing and aligning return series.

This module provides pure functions for return calculations and time series
alignment, following functional programming principles where possible.
"""

from typing import List
import pandas as pd
import numpy as np
from backend.entities import ReturnSeries, Asset


def compute_returns(prices: pd.Series, method: str = "log") -> pd.Series:
    """
    Compute returns from a price series.

    Preconditions:
        - prices is a pd.Series with numeric values
        - prices index is DatetimeIndex or convertible to dates
        - prices contains no negative values
        - method is either "log" or "simple"

    Postconditions:
        - Returns a pd.Series with same index as prices (minus first row)
        - Returns are computed as log returns or simple returns
        - First observation is NaN (no prior price to compute return from)

    Args:
        prices: Price series (must be positive)
        method: "log" for log returns, "simple" for simple returns

    Returns:
        Series of returns, same index as prices (first value is NaN)

    Raises:
        ValueError: If prices contains negative values or method is invalid
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pd.Series")

    if (prices <= 0).any():
        raise ValueError("prices must be positive")

    if method not in ["log", "simple"]:
        raise ValueError(f"method must be 'log' or 'simple', got {method}")

    if method == "log":
        returns = np.log(prices / prices.shift(1))
    else:  # simple
        returns = (prices / prices.shift(1)) - 1

    return returns


def create_return_series(
    prices: pd.Series,
    asset: Asset = None,
    method: str = "log"
) -> ReturnSeries:
    """
    Create a ReturnSeries from a price series.

    This is a convenience function that combines compute_returns and
    ReturnSeries construction.

    Preconditions:
        - prices is a pd.Series with numeric values
        - prices index is DatetimeIndex or convertible to dates
        - prices contains no negative values

    Postconditions:
        - Returns a ReturnSeries with sorted, deduplicated dates
        - First return value is NaN (no prior price)

    Args:
        prices: Price series
        asset: Optional Asset object
        method: "log" or "simple" returns

    Returns:
        ReturnSeries object
    """
    returns = compute_returns(prices, method=method)
    # Drop NaN from first observation
    returns = returns.dropna()

    if len(returns) == 0:
        raise ValueError("computed returns series is empty after dropping NaN")

    return ReturnSeries(returns.index, returns, asset=asset)


def align_series(series_list: List[ReturnSeries]) -> List[ReturnSeries]:
    """
    Align multiple ReturnSeries to a common date index.

    This function finds the intersection of all date indices and returns
    new ReturnSeries objects with aligned dates. Missing values in the
    aligned period are preserved as NaN.

    Preconditions:
        - series_list is non-empty
        - All series in list are valid ReturnSeries

    Postconditions:
        - All returned series have the same date index
        - Date index is the intersection of all input series dates
        - Original series are not modified

    Args:
        series_list: List of ReturnSeries to align

    Returns:
        List of ReturnSeries with aligned dates

    Raises:
        ValueError: If series_list is empty or intersection is empty
    """
    if not series_list:
        raise ValueError("series_list cannot be empty")

    if len(series_list) == 1:
        return series_list

    # Find intersection of all date indices
    common_dates = series_list[0].dates
    for series in series_list[1:]:
        common_dates = common_dates.intersection(series.dates)

    if len(common_dates) == 0:
        raise ValueError("no common dates found across series")

    # Create aligned series
    aligned = []
    for series in series_list:
        aligned_returns = series.returns.reindex(common_dates)
        aligned_series = ReturnSeries(
            common_dates,
            aligned_returns,
            asset=series.asset
        )
        aligned.append(aligned_series)

    return aligned


