"""
Correlation anomaly detection radar.

This module tracks rolling correlations between asset pairs and flags
anomalies when correlations deviate significantly from historical norms.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from backend.entities import ReturnSeries
from backend.analytics.returns import align_series


@dataclass
class CorrelationAnomaly:
    """
    Represents a correlation anomaly detection.

    Attributes:
        date: Date of the anomaly
        pair: Tuple of (asset1, asset2) tickers
        correlation: Rolling correlation value at this date
        z_score: Z-score of correlation vs historical distribution
        historical_mean: Historical mean correlation
        historical_std: Historical standard deviation
    """
    date: pd.Timestamp
    pair: Tuple[str, str]
    correlation: float
    z_score: float
    historical_mean: float
    historical_std: float

    def __repr__(self) -> str:
        """String representation."""
        pair_str = f"{self.pair[0]}-{self.pair[1]}"
        return (
            f"CorrelationAnomaly({pair_str}, date={self.date.date()}, "
            f"corr={self.correlation:.3f}, z={self.z_score:.2f})"
        )


class CorrelationRadar:
    """
    Tracks rolling correlations and detects anomalies.

    This class computes rolling correlations between asset pairs and
    flags when correlations deviate significantly (high z-score) from
    their historical distribution.

    Representation Invariants:
        - window_size > 0
        - min_history > window_size
        - z_threshold > 0
    """

    def __init__(
        self,
        window_size: int = 60,
        min_history: int = 120,
        z_threshold: float = 2.0
    ):
        """
        Initialize correlation radar.

        Preconditions:
            - window_size > 0
            - min_history > window_size
            - z_threshold > 0

        Postconditions:
            - Parameters are set

        Args:
            window_size: Rolling window size for correlation (days)
            min_history: Minimum history required before flagging anomalies
            z_threshold: Z-score threshold for flagging anomalies
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if min_history <= window_size:
            raise ValueError("min_history must be greater than window_size")
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")

        self.window_size = window_size
        self.min_history = min_history
        self.z_threshold = z_threshold

    def compute_rolling_correlation(
        self,
        series1: ReturnSeries,
        series2: ReturnSeries
    ) -> pd.Series:
        """
        Compute rolling correlation between two return series.

        Preconditions:
            - series1 and series2 are valid ReturnSeries
            - Series have overlapping dates

        Postconditions:
            - Returns Series with rolling correlations
            - Index is aligned with input series dates
            - First (window_size - 1) values are NaN

        Args:
            series1: First return series
            series2: Second return series

        Returns:
            Series of rolling correlations, indexed by date
        """
        # Align series
        aligned = align_series([series1, series2])
        returns1 = aligned[0].returns
        returns2 = aligned[1].returns

        # Compute rolling correlation
        rolling_corr = returns1.rolling(
            window=self.window_size,
            min_periods=self.window_size
        ).corr(returns2)

        return rolling_corr

    def detect_anomalies(
        self,
        series1: ReturnSeries,
        series2: ReturnSeries,
        asset1_name: Optional[str] = None,
        asset2_name: Optional[str] = None
    ) -> List[CorrelationAnomaly]:
        """
        Detect correlation anomalies between two assets.

        This computes rolling correlations and flags dates where the
        correlation z-score exceeds the threshold.

        Preconditions:
            - series1 and series2 are valid ReturnSeries
            - Series have at least min_history overlapping dates
            - asset1_name and asset2_name are tickers (if provided)

        Postconditions:
            - Returns list of CorrelationAnomaly objects
            - Anomalies are sorted by date
            - All anomalies have |z_score| >= z_threshold

        Args:
            series1: First return series
            series2: Second return series
            asset1_name: Name/ticker for first asset (defaults to series1.asset.ticker)
            asset2_name: Name/ticker for second asset (defaults to series2.asset.ticker)

        Returns:
            List of CorrelationAnomaly objects

        Raises:
            ValueError: If insufficient history
        """
        # Get asset names
        if asset1_name is None:
            asset1_name = series1.asset.ticker if series1.asset else "Asset1"
        if asset2_name is None:
            asset2_name = series2.asset.ticker if series2.asset else "Asset2"

        pair = (asset1_name, asset2_name)

        # Compute rolling correlation
        rolling_corr = self.compute_rolling_correlation(series1, series2)

        # Need at least min_history observations
        valid_corr = rolling_corr.dropna()
        if len(valid_corr) < self.min_history:
            raise ValueError(
                f"Insufficient history: {len(valid_corr)} observations, "
                f"need at least {self.min_history}"
            )

        # Split into historical and current periods
        # Use first (min_history - window_size) observations as history
        history_size = self.min_history - self.window_size
        if len(valid_corr) <= history_size:
            # Not enough data to have both history and current
            return []

        historical_corr = valid_corr.iloc[:history_size]
        current_corr = valid_corr.iloc[history_size:]

        # Compute historical statistics
        hist_mean = historical_corr.mean()
        hist_std = historical_corr.std()

        # Avoid division by zero
        if hist_std == 0:
            return []

        # Compute z-scores for current period
        z_scores = (current_corr - hist_mean) / hist_std

        # Find anomalies (|z_score| >= threshold)
        anomaly_mask = np.abs(z_scores) >= self.z_threshold

        # Build anomaly list
        anomalies = []
        for date, is_anomaly in anomaly_mask.items():
            if is_anomaly:
                anomalies.append(CorrelationAnomaly(
                    date=date,
                    pair=pair,
                    correlation=current_corr.loc[date],
                    z_score=z_scores.loc[date],
                    historical_mean=hist_mean,
                    historical_std=hist_std
                ))

        return sorted(anomalies, key=lambda x: x.date)

    def scan_pairs(
        self,
        series_list: List[ReturnSeries],
        asset_names: Optional[List[str]] = None
    ) -> List[CorrelationAnomaly]:
        """
        Scan all pairs of assets for correlation anomalies.

        Preconditions:
            - series_list has at least 2 series
            - If asset_names provided, len(asset_names) == len(series_list)

        Postconditions:
            - Returns list of all anomalies across all pairs
            - Each pair is scanned once (no duplicates)

        Args:
            series_list: List of ReturnSeries to scan
            asset_names: Optional list of asset names/tickers

        Returns:
            List of all CorrelationAnomaly objects
        """
        if len(series_list) < 2:
            return []

        if asset_names is None:
            asset_names = [
                s.asset.ticker if s.asset else f"Asset{i}"
                for i, s in enumerate(series_list)
            ]

        if len(asset_names) != len(series_list):
            raise ValueError("asset_names length must match series_list length")

        all_anomalies = []

        # Scan all pairs (no duplicates)
        for i in range(len(series_list)):
            for j in range(i + 1, len(series_list)):
                try:
                    anomalies = self.detect_anomalies(
                        series_list[i],
                        series_list[j],
                        asset_names[i],
                        asset_names[j]
                    )
                    all_anomalies.extend(anomalies)
                except ValueError:
                    # Skip pairs with insufficient data
                    continue

        # Sort by date, then by absolute z-score
        all_anomalies.sort(key=lambda x: (x.date, -abs(x.z_score)))

        return all_anomalies

    def get_top_anomalies(
        self,
        anomalies: List[CorrelationAnomaly],
        n: int = 10
    ) -> List[CorrelationAnomaly]:
        """
        Get top N anomalies by absolute z-score.

        Args:
            anomalies: List of CorrelationAnomaly objects
            n: Number of top anomalies to return

        Returns:
            Top N anomalies sorted by |z_score| (descending)
        """
        sorted_anomalies = sorted(anomalies, key=lambda x: abs(x.z_score), reverse=True)
        return sorted_anomalies[:n]


