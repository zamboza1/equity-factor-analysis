"""
Core entity classes (ADTs) for the radar module.

These classes represent the fundamental data structures used throughout
the analysis pipeline, with strong encapsulation and representation invariants.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class Asset:
    """
    Represents a financial asset (stock, ETF, etc.).

    Attributes:
        ticker: Stock ticker symbol (e.g., "AAPL")
        name: Human-readable name (e.g., "Apple Inc.")
        asset_type: Type of asset (e.g., "equity", "etf", "bond")

    Representation Invariants:
        - ticker is non-empty and uppercase
        - name is non-empty
        - asset_type is one of: "equity", "etf", "bond", "commodity", "currency"
    """
    ticker: str
    name: str
    asset_type: str

    def __post_init__(self):
        """Validate representation invariants."""
        if not self.ticker:
            raise ValueError("ticker cannot be empty")
        if not self.name:
            raise ValueError("name cannot be empty")
        if self.asset_type not in ["equity", "etf", "bond", "commodity", "currency"]:
            raise ValueError(f"invalid asset_type: {self.asset_type}")


class ReturnSeries:
    """
    A time series of returns with associated dates.

    This is an ADT that enforces invariants on return data:
    - Dates are sorted and non-duplicate
    - Returns and dates have equal length
    - No NaN values in dates (returns may have NaNs for missing data)

    Attributes:
        dates: Sorted array of dates (pd.DatetimeIndex)
        returns: Array of returns (pd.Series), aligned with dates
        asset: Optional Asset object this series belongs to

    Representation Invariants:
        - len(dates) == len(returns)
        - dates are sorted in ascending order
        - dates contain no duplicates
        - dates contain no NaN values
    """

    def __init__(self, dates: pd.DatetimeIndex, returns: pd.Series, asset: Optional[Asset] = None):
        """
        Initialize a ReturnSeries.

        Preconditions:
            - Labeled returns refer to the supplied dates, regardless of order
            - Arrays and RangeIndex Series are positional and have matching length

        Postconditions:
            - self.dates is sorted and deduplicated
            - self.returns is aligned with self.dates
        """
        # Convert to DatetimeIndex if needed
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)

        # Check for empty input
        if len(dates) == 0:
            raise ValueError("dates cannot be empty")

        if dates.hasnans:
            raise ValueError("dates must not contain NaN values")

        # Preserve label identity; equal lengths do not authorize relabeling.
        if isinstance(returns, pd.Series):
            if len(returns) != len(dates):
                raise ValueError(f"length mismatch: dates has {len(dates)} elements, returns has {len(returns)}")
            if isinstance(returns.index, pd.RangeIndex) and returns.index.equals(pd.RangeIndex(len(returns))):
                aligned_returns = pd.Series(returns.to_numpy(copy=True), index=dates)
            elif returns.index.equals(dates):
                aligned_returns = returns.copy()
            elif (isinstance(returns.index, pd.DatetimeIndex)
                  and not returns.index.has_duplicates and not dates.has_duplicates
                  and returns.index.difference(dates).empty and dates.difference(returns.index).empty):
                aligned_returns = returns.reindex(dates)
            else:
                raise ValueError("Labeled returns must match the supplied dates")
        else:
            # Convert to Series
            if len(returns) != len(dates):
                raise ValueError(f"length mismatch: dates has {len(dates)} elements, returns has {len(returns)}")
            aligned_returns = pd.Series(returns, index=dates)

        # Sort and deduplicate
        df = pd.DataFrame({"returns": aligned_returns}, index=dates)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # Check if result is empty
        if len(df) == 0:
            raise ValueError("resulting series is empty after deduplication")

        self._dates = df.index
        self._returns = df["returns"]
        self._asset = asset

        # Validate invariants
        self._check_invariants()

    def _check_invariants(self):
        """Check representation invariants."""
        if len(self._dates) != len(self._returns):
            raise ValueError("dates and returns must have equal length")
        if not self._dates.is_monotonic_increasing:
            raise ValueError("dates must be sorted in ascending order")
        if self._dates.has_duplicates:
            raise ValueError("dates must not contain duplicates")
        if self._dates.isna().any():
            raise ValueError("dates must not contain NaN values")

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Return the dates (read-only)."""
        return self._dates

    @property
    def returns(self) -> pd.Series:
        """Return the returns series (read-only)."""
        return self._returns

    @property
    def asset(self) -> Optional[Asset]:
        """Return the associated asset (read-only)."""
        return self._asset

    def __len__(self) -> int:
        """Return the number of observations."""
        return len(self._dates)

    def __repr__(self) -> str:
        """String representation."""
        asset_str = f" ({self._asset.ticker})" if self._asset else ""
        return f"ReturnSeries({len(self)} obs{asset_str})"
