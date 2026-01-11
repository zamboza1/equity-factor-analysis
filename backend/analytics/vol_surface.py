"""
Volatility surface cleaning and interpolation.

This module provides functions to clean option chain data and interpolate
implied volatility smiles.
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from backend.data_sources.options import OptionChain
from backend.errors import DataError


def clean_iv_smile(
    strikes: pd.Series,
    ivs: pd.Series,
    spot_price: float,
    max_iv: float = 5.0,
    min_iv: float = 0.001,
    outlier_z_threshold: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Clean implied volatility smile by removing outliers.

    Preconditions:
        - strikes and ivs have same length
        - spot_price > 0
        - max_iv > min_iv > 0

    Postconditions:
        - Returns cleaned strikes and ivs
        - All IVs are between min_iv and max_iv
        - Outliers (high z-score) are removed
        - Strikes are sorted

    Args:
        strikes: Strike prices
        ivs: Implied volatilities
        spot_price: Current spot price
        max_iv: Maximum allowed IV (default 500%)
        min_iv: Minimum allowed IV (default 0.1%)
        outlier_z_threshold: Z-score threshold for outlier removal

    Returns:
        Tuple of (cleaned_strikes, cleaned_ivs)
    """
    # Combine into DataFrame for easier manipulation
    df = pd.DataFrame({"strike": strikes.values, "iv": ivs.values})
    df = df.sort_values("strike")

    # Remove NaN
    df = df.dropna()

    if len(df) == 0:
        raise DataError("No valid IV data after removing NaN")

    # Filter by IV bounds
    df = df[(df["iv"] >= min_iv) & (df["iv"] <= max_iv)]

    if len(df) < 3:
        raise DataError(f"Insufficient data after filtering: {len(df)} points")

    # Remove outliers using z-score
    iv_mean = df["iv"].mean()
    iv_std = df["iv"].std()

    if iv_std > 0:
        z_scores = np.abs((df["iv"] - iv_mean) / iv_std)
        df = df[z_scores < outlier_z_threshold]

    if len(df) < 3:
        raise DataError(f"Insufficient data after outlier removal: {len(df)} points")

    return df["strike"].values, df["iv"].values


def interpolate_iv_smile(
    strikes: np.ndarray,
    ivs: np.ndarray,
    target_strikes: Optional[np.ndarray] = None,
    method: str = "linear"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate implied volatility smile.

    Preconditions:
        - strikes and ivs have same length
        - strikes are sorted
        - method is "linear" or "cubic"

    Postconditions:
        - Returns interpolated strikes and IVs
        - If target_strikes provided, uses those; otherwise uses original range

    Args:
        strikes: Strike prices (sorted)
        ivs: Implied volatilities
        target_strikes: Optional target strikes for interpolation
        method: Interpolation method ("linear" or "cubic")

    Returns:
        Tuple of (interpolated_strikes, interpolated_ivs)
    """
    if len(strikes) < 2:
        raise DataError("Need at least 2 points for interpolation")

    # Remove duplicates
    unique_mask = np.concatenate([[True], np.diff(strikes) > 0])
    strikes_unique = strikes[unique_mask]
    ivs_unique = ivs[unique_mask]

    if len(strikes_unique) < 2:
        raise DataError("Insufficient unique points for interpolation")

    # Create interpolator
    if method == "cubic" and len(strikes_unique) >= 4:
        kind = "cubic"
    else:
        kind = "linear"

    interpolator = interp1d(
        strikes_unique,
        ivs_unique,
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate"
    )

    # Interpolate
    if target_strikes is None:
        target_strikes = strikes_unique
    else:
        target_strikes = np.sort(target_strikes)

    interpolated_ivs = interpolator(target_strikes)

    # Ensure non-negative
    interpolated_ivs = np.maximum(interpolated_ivs, 0.0)

    return target_strikes, interpolated_ivs


def compute_moneyness(strikes: np.ndarray, spot_price: float) -> np.ndarray:
    """
    Compute moneyness (strike / spot).

    Args:
        strikes: Strike prices
        spot_price: Spot price

    Returns:
        Array of moneyness values
    """
    return strikes / spot_price

