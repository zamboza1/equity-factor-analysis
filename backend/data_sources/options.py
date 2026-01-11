"""
Options chain data sources.

This module handles fetching option chain data from yfinance.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import pandas as pd
import yfinance as yf
from backend.errors import DataError


@dataclass
class OptionChain:
    """
    Represents an option chain for a given expiry.

    Attributes:
        ticker: Underlying ticker
        expiry: Expiry date
        strikes: Array of strike prices
        call_ivs: Array of call implied volatilities
        put_ivs: Array of put implied volatilities
        spot_price: Current spot price
        expiry_date: Expiry date (pd.Timestamp)
    """
    ticker: str
    expiry: str
    strikes: pd.Series
    call_ivs: pd.Series
    put_ivs: pd.Series
    spot_price: float
    expiry_date: pd.Timestamp

    def __repr__(self) -> str:
        """String representation."""
        return f"OptionChain({self.ticker}, expiry={self.expiry}, {len(self.strikes)} strikes)"


def get_option_chain(
    ticker: str,
    expiry: Optional[str] = None
) -> OptionChain:
    """
    Fetch option chain data.

    This attempts to download from yfinance.

    Postconditions:
        - Returns OptionChain with strikes and IVs
        - Strikes are sorted
        - IVs are non-negative (or NaN for missing)

    Args:
        ticker: Underlying ticker
        expiry: Expiry date string (YYYY-MM-DD). If None, uses nearest expiry.

    Returns:
        OptionChain object

    Raises:
        DataError: If data cannot be fetched
    """
    # Try downloading first
    try:
        ticker_obj = yf.Ticker(ticker)
        option_chain = ticker_obj.option_chain(expiry) if expiry else ticker_obj.option_chain()

        # Extract data
        calls = option_chain.calls
        puts = option_chain.puts

        # Get spot price with multiple fallbacks
        info = ticker_obj.info
        spot_price = info.get("regularMarketPrice", 
                      info.get("currentPrice", 
                      info.get("previousClose", 0.0)))
        
        if spot_price == 0.0:
            # Try history if info fails
            hist = ticker_obj.history(period="1d")
            if not hist.empty:
                spot_price = float(hist["Close"].iloc[-1])

        if expiry:
            expiry_date = pd.to_datetime(expiry)
        else:
            # Use first available expiry
            avail_expiries = ticker_obj.options
            if avail_expiries:
                expiry_date = pd.to_datetime(avail_expiries[0])
            else:
                expiry_date = pd.Timestamp.now()

        # Extract strikes and IVs
        call_strikes = calls["strike"].values if not calls.empty else np.array([])
        call_ivs = calls.get("impliedVolatility", pd.Series([])).values if not calls.empty else np.array([])
        call_ivs = pd.Series(call_ivs).fillna(0).values

        put_strikes = puts["strike"].values if not puts.empty else np.array([])
        put_ivs = puts.get("impliedVolatility", pd.Series([])).values if not puts.empty else np.array([])
        put_ivs = pd.Series(put_ivs).fillna(0).values

        # Combine strikes (union)
        all_strikes = sorted(set(list(call_strikes) + list(put_strikes)))
        
        if not all_strikes:
            raise DataError(f"No options strikes found for {ticker}")

        # Map IVs to strikes
        call_iv_map = dict(zip(call_strikes, call_ivs))
        put_iv_map = dict(zip(put_strikes, put_ivs))

        call_ivs_aligned = [call_iv_map.get(s, 0.0) for s in all_strikes]
        put_ivs_aligned = [put_iv_map.get(s, 0.0) for s in all_strikes]

        return OptionChain(
            ticker=ticker,
            expiry=expiry or "nearest",
            strikes=pd.Series(all_strikes),
            call_ivs=pd.Series(call_ivs_aligned, index=all_strikes),
            put_ivs=pd.Series(put_ivs_aligned, index=all_strikes),
            spot_price=float(spot_price),
            expiry_date=expiry_date
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise DataError(f"Failed to fetch option chain for {ticker}: {e}")



