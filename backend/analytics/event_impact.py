"""
Macro event impact analysis.

This module analyzes price movements around macroeconomic events
(e.g., CPI releases, FOMC announcements) to measure market impact.

Uses a hybrid approach:
- For events within 7 days: Uses 1-minute intraday data
- For older events: Uses daily data (event day vs prior day)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from pathlib import Path
from backend.data_sources.macro_events import MacroEvent
from backend.data_sources.prices import get_prices, get_close_prices
from backend.analytics.returns import compute_returns
from backend.errors import DataError


@dataclass
class EventImpact:
    """
    Represents the impact of a macro event on an asset.

    Attributes:
        event: MacroEvent object
        ticker: Asset ticker
        pre_event_return: Return in pre-event window
        post_event_return: Return in post-event window
        cumulative_return: Cumulative return over full window
        window_start: Start of analysis window
        window_end: End of analysis window
        n_observations: Number of observations used
        data_type: "intraday" or "daily" indicating data granularity
    """
    event: MacroEvent
    ticker: str
    pre_event_return: float
    post_event_return: float
    cumulative_return: float
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_observations: int
    data_type: str = "intraday"

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EventImpact({self.ticker}, {self.event.name}, "
            f"cum_return={self.cumulative_return:.4f}, {self.data_type})"
        )


class EventImpactStudy:
    """
    Analyzes price movements around macro events.

    This class fetches minute-level price data around events and computes
    returns in pre-event and post-event windows to measure impact.

    Representation Invariants:
        - pre_window_minutes >= 0
        - post_window_minutes > 0
    """

    def __init__(
        self,
        pre_window_minutes: int = 30,
        post_window_minutes: int = 120
    ):
        """
        Initialize event impact study.

        Preconditions:
            - pre_window_minutes >= 0
            - post_window_minutes > 0

        Postconditions:
            - Parameters are set

        Args:
            pre_window_minutes: Minutes before event to analyze
            post_window_minutes: Minutes after event to analyze
        """
        if pre_window_minutes < 0:
            raise ValueError("pre_window_minutes must be non-negative")
        if post_window_minutes <= 0:
            raise ValueError("post_window_minutes must be positive")

        self.pre_window_minutes = pre_window_minutes
        self.post_window_minutes = post_window_minutes

    def fetch_intraday_data(
        self,
        ticker: str,
        event: MacroEvent,
        cache: Optional[object] = None,
        use_cache: bool = True
    ) -> pd.Series:
        """
        Fetch minute-level price data around an event.

        This attempts to download from yfinance first, but falls back to
        reading from cache if available.

        Preconditions:
            - ticker is a valid ticker string
            - event has valid timestamp

        Postconditions:
            - Returns Series of close prices indexed by minute timestamps
            - Series covers [event - pre_window, event + post_window]
            - Prices are sorted by timestamp

        Args:
            ticker: Asset ticker
            event: MacroEvent object
            cache: Optional cache object

        Returns:
            Series of close prices, indexed by minute timestamps

        Raises:
            DataError: If data cannot be fetched from either source
        """
        # Calculate window
        window_start = event.timestamp - pd.Timedelta(minutes=self.pre_window_minutes)
        window_end = event.timestamp + pd.Timedelta(minutes=self.post_window_minutes)

        # Try downloading first
        try:
            data = get_close_prices(
                ticker,
                start=window_start,
                end=window_end,
                interval="1m",
                cache=cache,
                use_cache=True
            )

            # Check if we got reasonable data
            if len(data) > 0:
                return data.sort_index()

        except Exception:
            # If download failed, raise error
            raise DataError(
                f"Failed to fetch intraday data for {ticker} around {event.timestamp}."
            )



    def analyze_event(
        self,
        ticker: str,
        event: MacroEvent,
        cache: Optional[object] = None,
        use_cache: bool = True
    ) -> EventImpact:
        """
        Analyze impact of a single event on an asset.

        Uses hybrid approach:
        - For events within 7 days: Uses 1-minute intraday data
        - For older events: Uses daily data (event day vs prior day)

        Preconditions:
            - ticker is valid
            - event has valid timestamp

        Postconditions:
            - Returns EventImpact with pre/post returns
            - Returns are computed as cumulative returns in each window

        Args:
        Args:
            ticker: Asset ticker
            event: MacroEvent object
            cache: Optional cache
            use_cache: Whether to use cache
            
        Returns:
            EventImpact object
        """
        # Check if event is recent enough for intraday data (7 days)
        now = pd.Timestamp.now(tz=event.timestamp.tz if event.timestamp.tz else None)
        days_ago = (now - event.timestamp).days
        
        if days_ago <= 7:
            # Try intraday data first
            try:
                return self._analyze_event_intraday(ticker, event, cache, use_cache)
            except DataError:
                # Fall back to daily
                pass
        
        # Use daily data for older events
        return self._analyze_event_daily(ticker, event, cache, use_cache)

    def _analyze_event_intraday(
        self,
        ticker: str,
        event: MacroEvent,
        cache: Optional[object],
        use_cache: bool
    ) -> EventImpact:
        """Analyze event using intraday (1-minute) data."""
        # Fetch intraday data
        prices = self.fetch_intraday_data(
            ticker, event, cache, use_cache
        )

        if len(prices) < 2:
            raise DataError(f"Insufficient intraday data for {ticker} around {event.timestamp}")

        # Compute returns
        returns = compute_returns(prices, method="log")
        returns = returns.dropna()

        if len(returns) == 0:
            raise DataError(f"No valid returns computed for {ticker}")

        # Split into pre-event and post-event
        event_time = event.timestamp
        pre_returns = returns[returns.index < event_time]
        post_returns = returns[returns.index >= event_time]

        # Compute cumulative returns
        pre_cum_return = pre_returns.sum() if len(pre_returns) > 0 else 0.0
        post_cum_return = post_returns.sum() if len(post_returns) > 0 else 0.0
        total_cum_return = returns.sum()

        return EventImpact(
            event=event,
            ticker=ticker,
            pre_event_return=pre_cum_return,
            post_event_return=post_cum_return,
            cumulative_return=total_cum_return,
            window_start=prices.index[0],
            window_end=prices.index[-1],
            n_observations=len(returns),
            data_type="intraday"
        )

    def _analyze_event_daily(
        self,
        ticker: str,
        event: MacroEvent,
        cache: Optional[object],
        use_cache: bool
    ) -> EventImpact:
        """
        Analyze event using daily data.
        
        For daily analysis:
        - pre_event_return: Return from day before event to event day open (approximated as 0)
        - post_event_return: Event day close vs previous day close
        - This captures the full day's reaction to the event
        """
        event_date = event.timestamp.date()
        
        # Get 5 days of data around the event to ensure we have enough
        start_date = (event.timestamp - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = (event.timestamp + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        
        try:
            prices = get_close_prices(
                ticker, start_date, end_date, 
                interval="1d", cache=cache, use_cache=use_cache
            )
            
            if len(prices) < 2:
                raise DataError(f"Insufficient daily data for {ticker} around {event.timestamp}")
            
            # Find event day and prior day
            prices_df = prices.to_frame() if isinstance(prices, pd.Series) else prices
            prices_df = prices_df.sort_index()
            
            # Get date index
            dates = prices_df.index.date if hasattr(prices_df.index, 'date') else prices_df.index
            
            # Find the event day (or closest trading day after)
            event_idx = None
            for i, d in enumerate(dates):
                if d >= event_date:
                    event_idx = i
                    break
            
            if event_idx is None or event_idx < 1:
                raise DataError(f"Cannot find event day data for {ticker}")
            
            # Get prices
            prior_price = float(prices_df.iloc[event_idx - 1].iloc[0] if hasattr(prices_df.iloc[event_idx - 1], 'iloc') else prices_df.iloc[event_idx - 1])
            event_price = float(prices_df.iloc[event_idx].iloc[0] if hasattr(prices_df.iloc[event_idx], 'iloc') else prices_df.iloc[event_idx])
            
            # Try to get next day for post-event
            if event_idx + 1 < len(prices_df):
                next_price = float(prices_df.iloc[event_idx + 1].iloc[0] if hasattr(prices_df.iloc[event_idx + 1], 'iloc') else prices_df.iloc[event_idx + 1])
            else:
                next_price = event_price
            
            # Compute returns
            # Pre-event: prior day close to event day close (full day move)
            event_day_return = np.log(event_price / prior_price)
            # Post-event: event day to next day (continuation)
            next_day_return = np.log(next_price / event_price) if next_price != event_price else 0.0
            
            # For daily data, we approximate:
            # - pre_event = 0 (we don't know intraday timing)
            # - post_event = full event day return (captures the reaction)
            pre_return = 0.0
            post_return = event_day_return
            
            window_start = prices_df.index[event_idx - 1]
            window_end = prices_df.index[min(event_idx + 1, len(prices_df) - 1)]
            
            return EventImpact(
                event=event,
                ticker=ticker,
                pre_event_return=pre_return,
                post_event_return=post_return,
                cumulative_return=event_day_return + next_day_return,
                window_start=window_start,
                window_end=window_end,
                n_observations=2,
                data_type="daily"
            )
            
        except Exception as e:
            raise DataError(f"Failed to analyze daily event impact: {e}")

    def analyze_multiple_events(
        self,
        ticker: str,
        events: List[MacroEvent],
        cache: Optional[object] = None,
        use_cache: bool = True
    ) -> List[EventImpact]:
        """
        Analyze impact of multiple events on an asset.

        Args:
            ticker: Asset ticker
            events: List of MacroEvent objects
            cache: Optional cache
            use_cache: Whether to use cache
        Args:
            ticker: Asset ticker
            events: List of MacroEvent objects
            cache: Optional cache
            use_cache: Whether to use cache

        Returns:
            List of EventImpact objects (one per event)
        """
        impacts = []
        for event in events:
            try:
                impact = self.analyze_event(
                    ticker, event, cache, use_cache
                )
                impacts.append(impact)
            except DataError:
                # Skip events where data is unavailable
                continue

        return impacts

    def summarize_impacts(self, impacts: List[EventImpact]) -> pd.DataFrame:
        """
        Create summary table of event impacts.

        Args:
            impacts: List of EventImpact objects

        Returns:
            DataFrame with summary statistics
        """
        if not impacts:
            return pd.DataFrame()

        data = {
            "ticker": [i.ticker for i in impacts],
            "event_name": [i.event.name for i in impacts],
            "event_date": [i.event.timestamp.date() for i in impacts],
            "pre_return": [i.pre_event_return for i in impacts],
            "post_return": [i.post_event_return for i in impacts],
            "cumulative_return": [i.cumulative_return for i in impacts],
            "n_observations": [i.n_observations for i in impacts],
            "data_type": [getattr(i, 'data_type', 'daily') for i in impacts],
        }

        df = pd.DataFrame(data)
        return df


