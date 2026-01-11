"""
Factor model regression and analysis.

This module implements factor model regression (e.g., Fama-French style)
to estimate factor exposures (betas) for assets.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from backend.entities import ReturnSeries
from backend.errors import DataError


@dataclass
class FactorFitResult:
    """
    Results from factor model regression.

    Attributes:
        betas: Dictionary mapping factor names to beta coefficients
        alpha: Intercept (alpha) coefficient
        r_squared: R-squared of the regression
        t_stats: Dictionary mapping factor names to t-statistics
        p_values: Dictionary mapping factor names to p-values
        n_observations: Number of observations used
        factor_names: List of factor names in order
    """
    betas: Dict[str, float]
    alpha: float
    r_squared: float
    t_stats: Dict[str, float]
    p_values: Dict[str, float]
    n_observations: int
    factor_names: List[str]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FactorFitResult(R²={self.r_squared:.3f}, "
            f"n={self.n_observations}, factors={len(self.factor_names)})"
        )


class FactorModel:
    """
    Factor model for estimating asset exposures to risk factors.

    This class fits a linear regression model:
        asset_return = alpha + beta1*factor1 + beta2*factor2 + ... + error

    Representation Invariants:
        - factor_names is non-empty
        - All factor data has same date index
    """

    def __init__(self, factor_names: List[str]):
        """
        Initialize factor model.

        Preconditions:
            - factor_names is non-empty
            - All names are strings

        Postconditions:
            - self.factor_names is set

        Args:
            factor_names: List of factor names (e.g., ["Market", "Value", "Size"])
        """
        if not factor_names:
            raise ValueError("factor_names cannot be empty")
        self.factor_names = list(factor_names)

    def fit(
        self,
        asset_returns: ReturnSeries,
        factor_data: pd.DataFrame,
        min_observations: int = 30
    ) -> FactorFitResult:
        """
        Fit factor model to asset returns.

        Preconditions:
            - asset_returns is a valid ReturnSeries
            - factor_data is a DataFrame with columns matching factor_names
            - factor_data index is DatetimeIndex
            - min_observations >= 2 (at least 2 for regression)

        Postconditions:
            - Returns FactorFitResult with regression statistics
            - All betas, t-stats, p-values are computed
            - R² is between 0 and 1 (or NaN if fit fails)

        Args:
            asset_returns: ReturnSeries for the asset
            factor_data: DataFrame with factor returns, indexed by date
            min_observations: Minimum number of observations required

        Returns:
            FactorFitResult with regression results

        Raises:
            DataError: If insufficient data or factor columns missing
        """
        # Align asset returns with factor data
        common_dates = asset_returns.dates.intersection(factor_data.index)

        if len(common_dates) < min_observations:
            raise DataError(
                f"Insufficient overlapping data: {len(common_dates)} observations, "
                f"need at least {min_observations}"
            )

        # Extract aligned data
        asset_aligned = asset_returns.returns.reindex(common_dates).dropna()
        
        # Check for Risk-Free rate to compute excess returns
        if "RF" in factor_data.columns:
            rf_aligned = factor_data["RF"].reindex(asset_aligned.index)
            # Subtract RF to get excess returns (Asset - RF)
            # Fama-French methodology requires regressing Excess Returns
            asset_aligned = asset_aligned - rf_aligned
        
        factor_aligned = factor_data[self.factor_names].reindex(asset_aligned.index)

        # Remove rows with any NaN in factors or asset returns
        # (RF subtraction might introduce NaNs if RF is missing)
        valid_mask = ~factor_aligned.isna().any(axis=1) & ~asset_aligned.isna()
        asset_aligned = asset_aligned[valid_mask]
        factor_aligned = factor_aligned[valid_mask]

        if len(asset_aligned) < min_observations:
            raise DataError(
                f"Insufficient valid data after alignment: {len(asset_aligned)} observations, "
                f"need at least {min_observations}"
            )

        # Prepare regression data
        y = asset_aligned.values
        X = factor_aligned.values

        # Add constant for intercept (alpha)
        X_with_const = add_constant(X)

        # Fit OLS regression
        try:
            model = OLS(y, X_with_const)
            results = model.fit()
        except Exception as e:
            raise DataError(f"Regression failed: {e}") from e

        # Extract coefficients
        # First coefficient is intercept (alpha), rest are betas
        coefs = results.params
        alpha = float(coefs[0])
        betas = {
            factor_name: float(coefs[i + 1])
            for i, factor_name in enumerate(self.factor_names)
        }

        # Extract statistics
        t_stats = {
            factor_name: float(results.tvalues[i + 1])
            for i, factor_name in enumerate(self.factor_names)
        }
        # Add alpha t-stat
        t_stats["alpha"] = float(results.tvalues[0])

        p_values = {
            factor_name: float(results.pvalues[i + 1])
            for i, factor_name in enumerate(self.factor_names)
        }
        # Add alpha p-value
        p_values["alpha"] = float(results.pvalues[0])

        r_squared = float(results.rsquared)

        return FactorFitResult(
            betas=betas,
            alpha=alpha,
            r_squared=r_squared,
            t_stats=t_stats,
            p_values=p_values,
            n_observations=len(asset_aligned),
            factor_names=self.factor_names
        )

    def interpret(self, result: FactorFitResult) -> str:
        """
        Generate plain-English interpretation of factor model results.

        This provides a human-readable summary of the factor exposures.

        Preconditions:
            - result is a valid FactorFitResult

        Postconditions:
            - Returns a string with interpretation

        Args:
            result: FactorFitResult from fit()

        Returns:
            Plain-English interpretation string
        """
        lines = []

        # Market beta interpretation
        if "Market" in result.betas:
            market_beta = result.betas["Market"]
            if abs(market_beta) < 0.3:
                lines.append(f"Low market sensitivity (β={market_beta:.2f})")
            elif abs(market_beta) < 0.7:
                lines.append(f"Moderate market sensitivity (β={market_beta:.2f})")
            elif abs(market_beta) < 1.3:
                lines.append(f"High market sensitivity (β={market_beta:.2f})")
            else:
                lines.append(f"Very high market sensitivity (β={market_beta:.2f})")

        # Value tilt
        if "Value" in result.betas:
            value_beta = result.betas["Value"]
            if value_beta > 0.2:
                lines.append(f"Value tilt (β={value_beta:.2f})")
            elif value_beta < -0.2:
                lines.append(f"Growth tilt (β={value_beta:.2f})")

        # Size exposure
        if "Size" in result.betas:
            size_beta = result.betas["Size"]
            if size_beta > 0.2:
                lines.append(f"Small-cap exposure (β={size_beta:.2f})")
            elif size_beta < -0.2:
                lines.append(f"Large-cap exposure (β={size_beta:.2f})")

        # Momentum
        if "Momentum" in result.betas:
            momentum_beta = result.betas["Momentum"]
            if momentum_beta > 0.2:
                lines.append(f"Momentum exposure (β={momentum_beta:.2f})")
            elif momentum_beta < -0.2:
                lines.append(f"Contrarian exposure (β={momentum_beta:.2f})")

        # R-squared summary
        if result.r_squared < 0.3:
            lines.append(f"Low model fit (R²={result.r_squared:.2f})")
        elif result.r_squared < 0.6:
            lines.append(f"Moderate model fit (R²={result.r_squared:.2f})")
        else:
            lines.append(f"Strong model fit (R²={result.r_squared:.2f})")

        return "; ".join(lines) if lines else "No significant factor exposures detected"


