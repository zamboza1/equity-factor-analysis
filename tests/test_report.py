"""
Tests for report generation.

Tests cover:
- Report file creation
- Sections present
- Chart generation
"""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from backend.reporting.report import Report
from backend.analytics.factor_model import FactorFitResult
from backend.analytics.corr_radar import CorrelationAnomaly
from backend.analytics.event_impact import EventImpact
from backend.data_sources.macro_events import MacroEvent
from backend.analytics.sabr import SABRFitResult, SABRParams
from backend.entities import Asset


class TestReport:
    """Tests for Report class."""

    def test_report_generation(self):
        """Test generating a basic report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Report(output_dir=tmpdir)

            # Create sample data
            asset = Asset("AAPL", "Apple Inc.", "equity")
            factor_result = FactorFitResult(
                betas={"Market": 1.2, "Value": 0.3},
                alpha=0.001,
                r_squared=0.65,
                t_stats={"Market": 5.0, "Value": 2.0, "alpha": 1.0},
                p_values={"Market": 0.001, "Value": 0.05, "alpha": 0.3},
                n_observations=100,
                factor_names=["Market", "Value"]
            )

            report_path = report.generate_report(
                ticker="AAPL",
                asset=asset,
                factor_result=factor_result,
                anomalies=[],
                event_impacts=[]
            )

            # Check report file exists
            assert Path(report_path).exists()

            # Check content
            content = Path(report_path).read_text()
            assert "AAPL" in content
            assert "Factor Exposures" in content
            assert "R²" in content

    def test_report_with_anomalies(self):
        """Test report with correlation anomalies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Report(output_dir=tmpdir)

            anomaly = CorrelationAnomaly(
                date=pd.Timestamp("2024-01-15"),
                pair=("AAPL", "MSFT"),
                correlation=0.85,
                z_score=2.5,
                historical_mean=0.5,
                historical_std=0.14
            )

            report_path = report.generate_report(
                ticker="AAPL",
                asset=None,
                factor_result=None,
                anomalies=[anomaly],
                event_impacts=[]
            )

            content = Path(report_path).read_text()
            assert "Correlation Anomalies" in content
            assert "AAPL-MSFT" in content

    def test_report_with_events(self):
        """Test report with event impacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Report(output_dir=tmpdir)

            event = MacroEvent(
                name="CPI",
                timestamp=pd.Timestamp("2024-01-11 08:30:00", tz="US/Eastern"),
                timezone="US/Eastern",
                source="BLS"
            )

            impact = EventImpact(
                event=event,
                ticker="SPY",
                pre_event_return=0.001,
                post_event_return=0.005,
                cumulative_return=0.006,
                window_start=pd.Timestamp("2024-01-11 08:00:00"),
                window_end=pd.Timestamp("2024-01-11 10:30:00"),
                n_observations=150
            )

            report_path = report.generate_report(
                ticker="SPY",
                asset=None,
                factor_result=None,
                anomalies=[],
                event_impacts=[impact]
            )

            content = Path(report_path).read_text()
            assert "Macro Event Impact" in content
            assert "CPI" in content

    def test_report_with_sabr(self):
        """Test report with SABR fit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Report(output_dir=tmpdir)

            sabr_result = SABRFitResult(
                params=SABRParams(alpha=0.2, beta=0.5, rho=-0.3, nu=0.5),
                fit_error=0.01,
                n_iterations=50,
                success=True
            )

            strikes = np.array([80, 90, 100, 110, 120])
            ivs = np.array([0.25, 0.23, 0.20, 0.22, 0.24])

            report_path = report.generate_report(
                ticker="SPY",
                asset=None,
                factor_result=None,
                anomalies=[],
                event_impacts=[],
                sabr_result=sabr_result,
                sabr_strikes=strikes,
                sabr_ivs=ivs,
                sabr_forward=100.0,
                sabr_time_to_expiry=0.25
            )

            content = Path(report_path).read_text()
            assert "SABR" in content
            assert "alpha" in content.lower()

    def test_report_assets_directory(self):
        """Test that assets directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Report(output_dir=tmpdir)
            assets_dir = Path(tmpdir) / "assets"

            # Generate report with charts
            event = MacroEvent(
                name="CPI",
                timestamp=pd.Timestamp("2024-01-11 08:30:00", tz="US/Eastern"),
                timezone="US/Eastern",
                source="BLS"
            )
            impact = EventImpact(
                event=event,
                ticker="SPY",
                pre_event_return=0.001,
                post_event_return=0.005,
                cumulative_return=0.006,
                window_start=pd.Timestamp("2024-01-11 08:00:00"),
                window_end=pd.Timestamp("2024-01-11 10:30:00"),
                n_observations=150
            )

            report.generate_report(
                ticker="SPY",
                asset=None,
                factor_result=None,
                anomalies=[],
                event_impacts=[impact]
            )

            # Assets directory should exist
            assert assets_dir.exists()
            assert assets_dir.is_dir()


