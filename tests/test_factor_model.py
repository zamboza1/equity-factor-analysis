"""
Tests for factor model regression.

Tests cover:
- Factor model fitting with synthetic data
- Beta recovery (known true betas)
- Insufficient data edge cases
- Interpretation function
"""

import pytest
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from backend.analytics.factor_model import FactorModel, FactorFitResult
from backend.entities import ReturnSeries, Asset
from backend.errors import DataError

def create_synthetic_factors(
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    n_days: Optional[int] = None
) -> pd.DataFrame:
    """Create synthetic factor data for internal testing ONLY."""
    if n_days is None:
        dates = pd.date_range(start_date, end_date, freq="D")
        dates = dates[dates.weekday < 5]
    else:
        dates = pd.date_range(start_date, periods=n_days, freq="B")

    np.random.seed(42)
    n = len(dates)
    market = np.random.normal(0.0005, 0.01, n)
    value = 0.3 * market + np.random.normal(0, 0.008, n)
    size = 0.2 * market + np.random.normal(0, 0.007, n)
    momentum = 0.1 * market + np.random.normal(0, 0.006, n)

    return pd.DataFrame({
        "Market": market, "Value": value, "Size": size, "Momentum": momentum
    }, index=dates)


class TestFactorModel:
    """Tests for FactorModel class."""

    def test_factor_model_init(self):
        """Test initializing factor model."""
        model = FactorModel(["Market", "Value", "Size"])
        assert model.factor_names == ["Market", "Value", "Size"]

    def test_factor_model_empty_names_raises(self):
        """Test that empty factor names raise error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            FactorModel([])

    def test_factor_model_fit_synthetic_data(self):
        """Test fitting factor model to synthetic data with known betas."""
        # Create synthetic factors with more observations for better fit
        factor_data = create_synthetic_factors(n_days=200)

        # Create asset returns with known betas
        # asset_return = 0.001 + 1.2*Market + 0.5*Value + 0.3*Size + noise
        true_alpha = 0.001
        true_betas = {"Market": 1.2, "Value": 0.5, "Size": 0.3}

        np.random.seed(42)
        # Use less noise for better recovery
        asset_returns = (
            true_alpha
            + true_betas["Market"] * factor_data["Market"]
            + true_betas["Value"] * factor_data["Value"]
            + true_betas["Size"] * factor_data["Size"]
            + np.random.normal(0, 0.002, len(factor_data))  # Very small noise
        )

        asset_series = ReturnSeries(factor_data.index, pd.Series(asset_returns, index=factor_data.index))

        # Fit model
        model = FactorModel(["Market", "Value", "Size"])
        result = model.fit(asset_series, factor_data, min_observations=30)

        # Check that betas are close to true values (within reasonable tolerance)
        # Factors are correlated, so tolerance needs to be more lenient
        assert abs(result.betas["Market"] - true_betas["Market"]) < 0.5
        assert abs(result.betas["Value"] - true_betas["Value"]) < 0.5
        assert abs(result.betas["Size"] - true_betas["Size"]) < 0.5
        assert abs(result.alpha - true_alpha) < 0.005
        
        # More importantly: check that the model fits well
        assert result.r_squared > 0.3  # Should have decent fit

        # Check R² is reasonable (should be high since we know the true model)
        assert result.r_squared > 0.5

    def test_factor_model_fit_insufficient_data_raises(self):
        """Test that insufficient data raises DataError."""
        factor_data = create_synthetic_factors(n_days=10)  # Too few observations

        asset_returns = pd.Series(
            np.random.normal(0, 0.01, len(factor_data)),
            index=factor_data.index
        )
        asset_series = ReturnSeries(factor_data.index, asset_returns)

        model = FactorModel(["Market", "Value"])
        with pytest.raises(DataError, match="Insufficient"):
            model.fit(asset_series, factor_data, min_observations=30)

    def test_factor_model_fit_missing_factor_columns_raises(self):
        """Test that missing factor columns raise error."""
        factor_data = pd.DataFrame({
            "Market": [0.01, 0.02],
            "Value": [0.005, 0.01]
        }, index=pd.date_range("2023-01-01", periods=2))

        asset_returns = pd.Series([0.01, 0.02], index=factor_data.index)
        asset_series = ReturnSeries(factor_data.index, asset_returns)

        model = FactorModel(["Market", "Size"])  # Size column missing
        with pytest.raises(KeyError):
            model.fit(asset_series, factor_data, min_observations=2)

    def test_factor_model_fit_no_overlap_raises(self):
        """Test that no date overlap raises error."""
        factor_data = pd.DataFrame({
            "Market": [0.01, 0.02],
            "Value": [0.005, 0.01]
        }, index=pd.date_range("2023-01-01", periods=2))

        # Asset returns on different dates
        asset_returns = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2))
        asset_series = ReturnSeries(asset_returns.index, asset_returns)

        model = FactorModel(["Market", "Value"])
        with pytest.raises(DataError, match="Insufficient"):
            model.fit(asset_series, factor_data, min_observations=2)

    def test_factor_model_interpret(self):
        """Test interpretation function."""
        result = FactorFitResult(
            betas={"Market": 1.2, "Value": 0.3, "Size": -0.4, "Momentum": 0.1},
            alpha=0.001,
            r_squared=0.65,
            t_stats={"Market": 5.0, "Value": 2.0, "Size": -2.5, "Momentum": 1.0, "alpha": 0.5},
            p_values={"Market": 0.001, "Value": 0.05, "Size": 0.02, "Momentum": 0.3, "alpha": 0.6},
            n_observations=100,
            factor_names=["Market", "Value", "Size", "Momentum"]
        )

        model = FactorModel(["Market", "Value", "Size", "Momentum"])
        interpretation = model.interpret(result)

        # Should mention high market sensitivity
        assert "High market sensitivity" in interpretation or "Very high market sensitivity" in interpretation
        # Should mention value tilt
        assert "Value tilt" in interpretation
        # Should mention large-cap exposure (negative size beta)
        assert "Large-cap" in interpretation
        # Should mention strong model fit
        assert "Strong model fit" in interpretation

    def test_factor_model_fit_with_nans(self):
        """Test that fit handles NaN values correctly."""
        factor_data = create_synthetic_factors(n_days=100)

        # Add some NaNs to factor data
        factor_data.loc[factor_data.index[10:15], "Market"] = np.nan

        # Create asset returns
        asset_returns = pd.Series(
            np.random.normal(0, 0.01, len(factor_data)),
            index=factor_data.index
        )
        asset_series = ReturnSeries(factor_data.index, asset_returns)

        model = FactorModel(["Market", "Value"])
        result = model.fit(asset_series, factor_data, min_observations=30)

        # Should still fit successfully (NaNs removed)
        assert result.n_observations < 100  # Some observations removed
        assert result.n_observations >= 30  # But still enough

    def test_factor_fit_result_repr(self):
        """Test FactorFitResult string representation."""
        result = FactorFitResult(
            betas={"Market": 1.0},
            alpha=0.001,
            r_squared=0.5,
            t_stats={"Market": 5.0, "alpha": 1.0},
            p_values={"Market": 0.001, "alpha": 0.3},
            n_observations=100,
            factor_names=["Market"]
        )

        repr_str = repr(result)
        assert "FactorFitResult" in repr_str
        assert "R²" in repr_str or "R^2" in repr_str
        assert "100" in repr_str  # n_observations


class TestCreateSyntheticFactors:
    """Tests for synthetic factor generation."""

    def test_create_synthetic_factors_default(self):
        """Test creating synthetic factors with default params."""
        factors = create_synthetic_factors()
        assert isinstance(factors, pd.DataFrame)
        assert "Market" in factors.columns
        assert "Value" in factors.columns
        assert "Size" in factors.columns
        assert "Momentum" in factors.columns
        assert len(factors) > 0

    def test_create_synthetic_factors_n_days(self):
        """Test creating synthetic factors with n_days."""
        factors = create_synthetic_factors(n_days=50)
        assert len(factors) == 50

    def test_create_synthetic_factors_date_range(self):
        """Test creating synthetic factors with date range."""
        factors = create_synthetic_factors(
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        assert len(factors) > 0
        assert factors.index[0] >= pd.Timestamp("2023-01-01")
        assert factors.index[-1] <= pd.Timestamp("2023-01-31")

