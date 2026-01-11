"""
Tests for correlation anomaly radar.

Tests cover:
- Rolling correlation computation
- Z-score calculation
- Anomaly detection
- Edge cases (insufficient data, etc.)
"""

import pytest
import pandas as pd
import numpy as np
from backend.analytics.corr_radar import CorrelationRadar, CorrelationAnomaly
from backend.entities import ReturnSeries, Asset


class TestCorrelationRadar:
    """Tests for CorrelationRadar class."""

    def test_radar_init(self):
        """Test initializing correlation radar."""
        radar = CorrelationRadar(window_size=60, min_history=120, z_threshold=2.0)
        assert radar.window_size == 60
        assert radar.min_history == 120
        assert radar.z_threshold == 2.0

    def test_radar_invalid_window_size_raises(self):
        """Test that invalid window_size raises error."""
        with pytest.raises(ValueError, match="window_size must be positive"):
            CorrelationRadar(window_size=0)

    def test_radar_invalid_min_history_raises(self):
        """Test that min_history <= window_size raises error."""
        with pytest.raises(ValueError, match="min_history must be greater"):
            CorrelationRadar(window_size=60, min_history=50)

    def test_radar_invalid_z_threshold_raises(self):
        """Test that invalid z_threshold raises error."""
        with pytest.raises(ValueError, match="z_threshold must be positive"):
            CorrelationRadar(z_threshold=-1.0)

    def test_compute_rolling_correlation(self):
        """Test computing rolling correlation."""
        # Create two correlated series
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Series 1: random walk
        series1_returns = np.random.normal(0, 0.01, len(dates))
        # Series 2: correlated with series1 (0.7 correlation) + noise
        series2_returns = 0.7 * series1_returns + np.random.normal(0, 0.007, len(dates))

        series1 = ReturnSeries(dates, pd.Series(series1_returns, index=dates))
        series2 = ReturnSeries(dates, pd.Series(series2_returns, index=dates))

        radar = CorrelationRadar(window_size=20, min_history=40, z_threshold=2.0)
        rolling_corr = radar.compute_rolling_correlation(series1, series2)

        # Should have some NaN values at start (window_size - 1)
        assert pd.isna(rolling_corr.iloc[:19]).all()
        # Should have valid correlations after window
        assert not pd.isna(rolling_corr.iloc[20:]).all()
        # Correlations should be between -1 and 1
        valid_corr = rolling_corr.dropna()
        assert (valid_corr >= -1).all() and (valid_corr <= 1).all()

    def test_detect_anomalies_insufficient_history_raises(self):
        """Test that insufficient history raises error."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        returns1 = pd.Series(np.random.normal(0, 0.01, len(dates)), index=dates)
        returns2 = pd.Series(np.random.normal(0, 0.01, len(dates)), index=dates)

        series1 = ReturnSeries(dates, returns1)
        series2 = ReturnSeries(dates, returns2)

        radar = CorrelationRadar(window_size=20, min_history=100, z_threshold=2.0)
        with pytest.raises(ValueError, match="Insufficient history"):
            radar.detect_anomalies(series1, series2)

    def test_detect_anomalies_with_known_anomaly(self):
        """Test detecting a known correlation anomaly."""
        # Create series with a correlation break
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        np.random.seed(42)

        # First 150 days: high correlation (0.8)
        series1_base = np.random.normal(0, 0.01, 150)
        series2_base = 0.8 * series1_base + np.random.normal(0, 0.006, 150)

        # Last 50 days: negative correlation (-0.7) - this is the anomaly
        series1_anomaly = np.random.normal(0, 0.01, 50)
        series2_anomaly = -0.7 * series1_anomaly + np.random.normal(0, 0.007, 50)

        series1_returns = np.concatenate([series1_base, series1_anomaly])
        series2_returns = np.concatenate([series2_base, series2_anomaly])

        series1 = ReturnSeries(dates, pd.Series(series1_returns, index=dates))
        series2 = ReturnSeries(dates, pd.Series(series2_returns, index=dates))

        radar = CorrelationRadar(window_size=20, min_history=120, z_threshold=2.0)
        anomalies = radar.detect_anomalies(series1, series2, "ASSET1", "ASSET2")

        # Should detect anomalies in the last period (correlation flipped)
        assert len(anomalies) > 0
        # All anomalies should have high |z_score|
        assert all(abs(a.z_score) >= 2.0 for a in anomalies)
        # Anomalies should be in the later period
        assert all(a.date >= dates[150] for a in anomalies)

    def test_detect_anomalies_no_anomalies(self):
        """Test that stable correlation produces no anomalies."""
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        np.random.seed(42)

        # Stable correlation throughout
        series1_returns = np.random.normal(0, 0.01, len(dates))
        series2_returns = 0.6 * series1_returns + np.random.normal(0, 0.008, len(dates))

        series1 = ReturnSeries(dates, pd.Series(series1_returns, index=dates))
        series2 = ReturnSeries(dates, pd.Series(series2_returns, index=dates))

        radar = CorrelationRadar(window_size=20, min_history=120, z_threshold=3.0)  # High threshold
        anomalies = radar.detect_anomalies(series1, series2)

        # With stable correlation and high threshold, should have few/no anomalies
        # (might have some due to noise, but should be minimal)
        assert len(anomalies) < 10  # Very few if any

    def test_scan_pairs(self):
        """Test scanning multiple pairs."""
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        np.random.seed(42)

        # Create 3 series
        returns1 = np.random.normal(0, 0.01, len(dates))
        returns2 = 0.7 * returns1 + np.random.normal(0, 0.007, len(dates))
        returns3 = np.random.normal(0, 0.01, len(dates))  # Uncorrelated

        series1 = ReturnSeries(dates, pd.Series(returns1, index=dates))
        series2 = ReturnSeries(dates, pd.Series(returns2, index=dates))
        series3 = ReturnSeries(dates, pd.Series(returns3, index=dates))

        radar = CorrelationRadar(window_size=20, min_history=120, z_threshold=2.5)
        anomalies = radar.scan_pairs([series1, series2, series3], ["A", "B", "C"])

        # Should scan 3 pairs: A-B, A-C, B-C
        # Anomalies might be detected depending on correlation structure
        assert isinstance(anomalies, list)
        # All anomalies should have valid pairs
        for anomaly in anomalies:
            assert len(anomaly.pair) == 2
            assert anomaly.pair[0] in ["A", "B", "C"]
            assert anomaly.pair[1] in ["A", "B", "C"]

    def test_scan_pairs_insufficient_series(self):
        """Test that scan_pairs with < 2 series returns empty."""
        radar = CorrelationRadar()
        anomalies = radar.scan_pairs([ReturnSeries(
            pd.date_range("2023-01-01", periods=10),
            pd.Series([0.01] * 10, index=pd.date_range("2023-01-01", periods=10))
        )])
        assert anomalies == []

    def test_get_top_anomalies(self):
        """Test getting top N anomalies by z-score."""
        anomalies = [
            CorrelationAnomaly(
                date=pd.Timestamp("2023-01-01"),
                pair=("A", "B"),
                correlation=0.8,
                z_score=2.5,
                historical_mean=0.5,
                historical_std=0.1
            ),
            CorrelationAnomaly(
                date=pd.Timestamp("2023-01-02"),
                pair=("A", "C"),
                correlation=0.9,
                z_score=4.0,  # Highest
                historical_mean=0.3,
                historical_std=0.15
            ),
            CorrelationAnomaly(
                date=pd.Timestamp("2023-01-03"),
                pair=("B", "C"),
                correlation=-0.7,
                z_score=-3.0,  # Second highest |z|
                historical_mean=0.2,
                historical_std=0.3
            ),
        ]

        radar = CorrelationRadar()
        top = radar.get_top_anomalies(anomalies, n=2)

        assert len(top) == 2
        # Should be sorted by |z_score| descending
        assert abs(top[0].z_score) == 4.0
        assert abs(top[1].z_score) == 3.0

    def test_correlation_anomaly_repr(self):
        """Test CorrelationAnomaly string representation."""
        anomaly = CorrelationAnomaly(
            date=pd.Timestamp("2023-01-15"),
            pair=("AAPL", "MSFT"),
            correlation=0.85,
            z_score=2.5,
            historical_mean=0.5,
            historical_std=0.14
        )

        repr_str = repr(anomaly)
        assert "AAPL-MSFT" in repr_str
        assert "2.5" in repr_str


