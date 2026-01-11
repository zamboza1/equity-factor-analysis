"""
Price data download and caching.

This module handles downloading price data from yfinance with caching
to avoid repeated network calls.
"""

from datetime import datetime, date
from typing import Optional, List, Union
import pandas as pd
import yfinance as yf
from backend.cache import DataCache
from backend.errors import DataError


def get_prices(
    tickers: Union[str, List[str]],
    start: Union[str, date, datetime],
    end: Union[str, date, datetime],
    interval: str = "1d",
    cache: Optional[DataCache] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Download price data for one or more tickers.

    This function downloads OHLCV data from yfinance and optionally caches
    the results to disk. If cached data exists and use_cache is True,
    returns cached data instead of downloading.

    Preconditions:
        - tickers is a non-empty string or list of strings
        - start and end are valid dates (start < end)
        - interval is a valid yfinance interval ("1d", "1m", "5m", etc.)
        - If cache is provided, it's a valid DataCache instance

    Postconditions:
        - Returns DataFrame with MultiIndex (ticker, date) or single-level
          index (date) if single ticker
        - Columns include: Open, High, Low, Close, Volume
        - Dates are sorted in ascending order
        - No duplicate dates per ticker

    Args:
        tickers: Single ticker string or list of tickers
        start: Start date (string "YYYY-MM-DD" or date/datetime)
        end: End date (string "YYYY-MM-DD" or date/datetime)
        interval: Data interval ("1d" for daily, "1m" for 1-minute, etc.)
        cache: Optional DataCache instance for caching
        use_cache: Whether to use cache if available

    Returns:
        DataFrame with price data. For single ticker, index is date.
        For multiple tickers, MultiIndex is (ticker, date).

    Raises:
        DataError: If download fails or returns empty data
    """
    # Normalize inputs
    if isinstance(tickers, str):
        ticker_list = [tickers]
        single_ticker = True
    else:
        ticker_list = list(tickers)
        single_ticker = len(ticker_list) == 1

    if not ticker_list:
        raise ValueError("tickers cannot be empty")

    # Convert dates to strings for yfinance
    if isinstance(start, (date, datetime)):
        start_str = start.strftime("%Y-%m-%d")
    else:
        start_str = str(start)

    if isinstance(end, (date, datetime)):
        end_str = end.strftime("%Y-%m-%d")
    else:
        end_str = str(end)

    # Build cache query params
    query_params = {
        "tickers": sorted(ticker_list),  # Sort for consistent hashing
        "start": start_str,
        "end": end_str,
        "interval": interval
    }

    # Check cache first
    if use_cache and cache is not None:
        cached_data = cache.get(query_params)
        if cached_data is not None:
            return cached_data

    # Download data
    try:
        ticker_str = " ".join(ticker_list)
        ticker_obj = yf.Ticker(ticker_str) if single_ticker else yf.Tickers(ticker_str)

        if single_ticker:
            data = ticker_obj.history(start=start_str, end=end_str, interval=interval)
            if data.empty:
                raise DataError(f"No data returned for {ticker_list[0]}")
            # Ensure proper column names
            data.columns = [col.replace(" ", "") for col in data.columns]
        else:
            # Multiple tickers - download each separately and combine
            data_dict = {}
            for ticker in ticker_list:
                ticker_data = yf.Ticker(ticker).history(
                    start=start_str, end=end_str, interval=interval
                )
                if not ticker_data.empty:
                    ticker_data.columns = [col.replace(" ", "") for col in ticker_data.columns]
                    data_dict[ticker] = ticker_data

            if not data_dict:
                raise DataError(f"No data returned for any ticker in {ticker_list}")

            # Combine into MultiIndex DataFrame
            data = pd.concat(data_dict, names=["Ticker", "Date"])

        # Sort by date
        if single_ticker:
            data = data.sort_index()
        else:
            data = data.sort_index(level=1)  # Sort by date (second level)

        # Cache the result
        if cache is not None:
            cache.set(query_params, data)

        return data

    except Exception as e:
        if isinstance(e, DataError):
            raise
        raise DataError(f"Failed to download price data: {e}") from e


def get_close_prices(
    tickers: Union[str, List[str]],
    start: Union[str, date, datetime],
    end: Union[str, date, datetime],
    interval: str = "1d",
    cache: Optional[DataCache] = None,
    use_cache: bool = True
) -> pd.Series:
    """
    Get close prices only (convenience function).

    This is a thin wrapper around get_prices that extracts the Close column.

    Preconditions:
        - Same as get_prices

    Postconditions:
        - Returns Series with same index structure as get_prices
        - Contains only Close prices

    Args:
        tickers: Single ticker string or list of tickers
        start: Start date
        end: End date
        interval: Data interval
        cache: Optional DataCache instance
        use_cache: Whether to use cache

    Returns:
        Series of close prices
    """
    data = get_prices(tickers, start, end, interval, cache, use_cache)

    if isinstance(data.index, pd.MultiIndex):
        # MultiIndex case - extract Close
        return data["Close"]
    else:
        # Single index case
        return data["Close"]


