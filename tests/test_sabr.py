"""
Tests for SABR model and volatility surface.

Tests cover:
- SABR implied vol function
- SABR calibration
- Vol surface cleaning
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from backend.analytics.sabr import (
    SABRModel, SABRParams, SABRFitResult,
    sabr_implied_vol
)
from backend.analytics.vol_surface import (
    clean_iv_smile, interpolate_iv_smile, compute_moneyness
)
from backend.errors import CalibrationError, DataError


class TestSABRImpliedVol:
    """Tests for sabr_implied_vol function."""

    def test_sabr_atm(self):
        """Test SABR formula at the money."""
        forward = 100.0
        strike = 100.0
        time_to_expiry = 0.25  # 3 months
        alpha = 0.2
        beta = 0.5
        rho = -0.3
        nu = 0.5

        iv = sabr_implied_vol(forward, strike, time_to_expiry, alpha, beta, rho, nu)

        # ATM vol should be approximately alpha / forward^(1-beta)
        expected_approx = alpha / (forward ** (1 - beta))
        assert abs(iv - expected_approx) < 0.1

    def test_sabr_positive(self):
        """Test that SABR IV is always positive."""
        forward = 100.0
        strikes = np.array([80, 90, 100, 110, 120])
        time_to_expiry = 0.25
        alpha = 0.2
        beta = 0.5
        rho = -0.3
        nu = 0.5

        for strike in strikes:
            iv = sabr_implied_vol(forward, strike, time_to_expiry, alpha, beta, rho, nu)
            assert iv >= 0

    def test_sabr_invalid_inputs(self):
        """Test that invalid inputs return 0 or handle gracefully."""
        # Zero forward
        iv = sabr_implied_vol(0, 100, 0.25, 0.2, 0.5, -0.3, 0.5)
        assert iv == 0.0

        # Zero strike
        iv = sabr_implied_vol(100, 0, 0.25, 0.2, 0.5, -0.3, 0.5)
        assert iv == 0.0

        # Zero time
        iv = sabr_implied_vol(100, 100, 0, 0.2, 0.5, -0.3, 0.5)
        assert iv == 0.0


class TestSABRModel:
    """Tests for SABRModel class."""

    def test_sabr_model_init(self):
        """Test initializing SABR model."""
        model = SABRModel(beta=0.5, beta_fixed=True)
        assert model.beta == 0.5
        assert model.beta_fixed is True

    def test_sabr_model_invalid_beta_raises(self):
        """Test that invalid beta raises error."""
        with pytest.raises(ValueError, match="beta must be between"):
            SABRModel(beta=1.5)

    def test_sabr_calibrate_synthetic_smile(self):
        """Test calibrating SABR to synthetic smile."""
        # Create synthetic smile using known SABR parameters
        forward = 100.0
        time_to_expiry = 0.25
        true_params = SABRParams(alpha=0.2, beta=0.5, rho=-0.3, nu=0.5)

        strikes = np.array([80, 90, 95, 100, 105, 110, 120])
        ivs = np.array([
            sabr_implied_vol(
                forward, k, time_to_expiry,
                true_params.alpha, true_params.beta,
                true_params.rho, true_params.nu
            )
            for k in strikes
        ])

        # Add small noise
        np.random.seed(42)
        ivs += np.random.normal(0, 0.01, len(ivs))

        # Calibrate
        model = SABRModel(beta=0.5, beta_fixed=True)
        result = model.calibrate(strikes, ivs, forward, time_to_expiry)

        assert result.success
        # Parameters may not recover exactly (calibration is sensitive)
        # But fit error should be small (model fits the data well)
        assert result.fit_error < 0.05
        # Parameters should be in reasonable ranges
        assert 0 < result.params.alpha < 2.0
        assert -1 <= result.params.rho <= 1
        assert 0 < result.params.nu < 5.0

    def test_sabr_calibrate_insufficient_strikes_raises(self):
        """Test that insufficient strikes raise error."""
        forward = 100.0
        strikes = np.array([100, 110])  # Only 2 strikes
        ivs = np.array([0.2, 0.22])

        model = SABRModel()
        with pytest.raises(CalibrationError, match="Need at least 3"):
            model.calibrate(strikes, ivs, forward, 0.25)

    def test_sabr_calibrate_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise error."""
        forward = 100.0
        strikes = np.array([100, 110, 120])
        ivs = np.array([0.2, 0.22])  # Different length

        model = SABRModel()
        with pytest.raises(ValueError, match="must have same length"):
            model.calibrate(strikes, ivs, forward, 0.25)


class TestVolSurface:
    """Tests for volatility surface cleaning."""

    def test_clean_iv_smile_basic(self):
        """Test basic IV smile cleaning."""
        strikes = pd.Series([80, 90, 100, 110, 120])
        ivs = pd.Series([0.2, 0.22, 0.25, 0.23, 0.21])
        spot = 100.0

        cleaned_strikes, cleaned_ivs = clean_iv_smile(strikes, ivs, spot)

        assert len(cleaned_strikes) == len(cleaned_ivs)
        assert all(0.001 <= iv <= 5.0 for iv in cleaned_ivs)

    def test_clean_iv_smile_removes_outliers(self):
        """Test that outliers are removed."""
        strikes = pd.Series([80, 90, 100, 110, 120])
        ivs = pd.Series([0.2, 0.22, 0.25, 0.23, 10.0])  # Outlier at end
        spot = 100.0

        cleaned_strikes, cleaned_ivs = clean_iv_smile(strikes, ivs, spot)

        # Outlier should be removed
        assert len(cleaned_ivs) < len(ivs)
        assert all(iv < 5.0 for iv in cleaned_ivs)

    def test_clean_iv_smile_insufficient_data_raises(self):
        """Test that insufficient data raises error."""
        strikes = pd.Series([100, 110])
        ivs = pd.Series([0.2, 0.22])
        spot = 100.0

        with pytest.raises(DataError, match="Insufficient"):
            clean_iv_smile(strikes, ivs, spot)

    def test_interpolate_iv_smile(self):
        """Test IV smile interpolation."""
        strikes = np.array([80, 90, 100, 110, 120])
        ivs = np.array([0.25, 0.23, 0.20, 0.22, 0.24])

        interp_strikes, interp_ivs = interpolate_iv_smile(strikes, ivs)

        assert len(interp_strikes) == len(interp_ivs)
        assert all(iv >= 0 for iv in interp_ivs)

    def test_interpolate_iv_smile_target_strikes(self):
        """Test interpolation with target strikes."""
        strikes = np.array([80, 90, 100, 110, 120])
        ivs = np.array([0.25, 0.23, 0.20, 0.22, 0.24])
        target = np.array([85, 95, 105, 115])

        interp_strikes, interp_ivs = interpolate_iv_smile(strikes, ivs, target_strikes=target)

        assert len(interp_strikes) == len(target)
        assert np.allclose(interp_strikes, target)

    def test_compute_moneyness(self):
        """Test moneyness computation."""
        strikes = np.array([80, 100, 120])
        spot = 100.0

        moneyness = compute_moneyness(strikes, spot)

        assert np.allclose(moneyness, [0.8, 1.0, 1.2])

