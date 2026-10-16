"""
SABR (Stochastic Alpha Beta Rho) volatility model.

This module implements the SABR model for fitting implied volatility smiles.
Think of the SABR model as a way to "connect the dots" between different option prices
to understand how the market expects volatility to change as the stock price moves.

The SABR model uses 4 specialized parameters:
- alpha (α): Volatility scale; ATM IV also depends on forward, beta and maturity.
- beta (β): The "Backbone" - determines how volatility scales with the stock price.
- rho (ρ): The "Skew" - measures if volatility tends to rise when the stock falls (fear).
- nu (ν): The "Smile Curvature" - measures the risk of extreme price jumps.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from backend.errors import CalibrationError


@dataclass
class SABRParams:
    """
    SABR model parameters.

    Attributes:
        alpha: Volatility level
        beta: Skew parameter (0 <= beta <= 1)
        rho: Correlation (-1 <= rho <= 1)
        nu: Volatility of volatility (nu >= 0)
    """
    alpha: float  # Volatility scale
    beta: float   # Backbone (scaling factor, 0 to 1)
    rho: float    # Skew (spot-vol correlation, -1 to 1)
    nu: float     # Vol-of-vol (smile curvature)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SABRParams(α={self.alpha:.4f}, β={self.beta:.2f}, "
            f"ρ={self.rho:.2f}, ν={self.nu:.4f})"
        )


@dataclass
class SABRFitResult:
    """
    Results from SABR model calibration.

    Attributes:
        params: SABRParams object
        fit_error: Root mean squared error of fit
        n_iterations: Number of optimization iterations
        success: Whether optimization converged; not a guarantee of fit/pricing quality
    """
    params: SABRParams
    fit_error: float
    n_iterations: int
    success: bool

    def __repr__(self) -> str:
        """String representation."""
        return f"SABRFitResult({self.params}, RMSE={self.fit_error:.6f})"


def sabr_implied_vol(
    forward: float,
    strike: float,
    time_to_expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float
) -> float:
    """
    Compute SABR implied volatility.

    This implements the Hagan et al. (2002) SABR formula for implied volatility.

    Preconditions:
        - forward > 0
        - strike > 0
        - time_to_expiry > 0
        - 0 <= beta <= 1
        - -1 < rho < 1
        - nu >= 0
        - alpha > 0

    Postconditions:
        - Returns implied volatility (annualized)
        - Result is non-negative

    Args:
        forward: Forward price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        alpha: Volatility level
        beta: Skew parameter
        rho: Correlation
        nu: Volatility of volatility

    Returns:
        Implied volatility (annualized)
    """
    inputs = [forward, strike, time_to_expiry, alpha, beta, rho, nu]
    if not np.isfinite(inputs).all():
        raise ValueError("SABR inputs must be finite")
    if min(forward, strike, time_to_expiry, alpha) <= 0:
        raise ValueError("forward, strike, maturity and alpha must be positive")
    if not 0 <= beta <= 1 or not -1 < rho < 1 or nu < 0:
        raise ValueError("SABR requires beta in [0,1], rho in (-1,1), and nu >= 0")

    # Hagan et al. lognormal approximation. Alpha is a scale parameter, not IV.
    # The same expression includes ATM, beta=1 and nu=0 through the small-z limit.
    log_m = np.log(forward / strike)
    one_minus_beta = 1 - beta
    geometric_scale = np.exp(one_minus_beta * (np.log(forward) + np.log(strike)) / 2)
    z = nu * geometric_scale * log_m / alpha
    if abs(z) < 1e-5:
        z_over_x = 1 - rho * z / 2 + (2 - 3 * rho**2) * z**2 / 12
    else:
        # asinh difference is algebraically x(z), without cancellation in its log numerator.
        rho_scale = np.sqrt(1 - rho**2)
        x = np.arcsinh((z - rho) / rho_scale) + np.arcsinh(rho / rho_scale)
        z_over_x = z / x
    log_correction = one_minus_beta**2 * log_m**2
    denominator = geometric_scale * (1 + log_correction / 24 + log_correction**2 / 1920)
    time_correction = (
        one_minus_beta**2 * alpha**2 / (24 * geometric_scale**2)
        + rho * beta * nu * alpha / (4 * geometric_scale)
        + (2 - 3 * rho**2) * nu**2 / 24
    )
    vol = alpha / denominator * z_over_x * (1 + time_to_expiry * time_correction)
    if not np.isfinite(vol) or vol <= 0:
        raise ValueError("SABR approximation produced an invalid volatility")
    return float(vol)


class SABRModel:
    """
    SABR model for volatility smile calibration.

    This class fits SABR parameters to observed implied volatilities
    by minimizing squared error.
    """

    def __init__(
        self,
        beta: float = 0.5,
        beta_fixed: bool = False
    ):
        """
        Initialize SABR model.

        Preconditions:
            - 0 <= beta <= 1

        Postconditions:
            - Model is initialized with beta parameter

        Args:
            beta: Initial/fixed beta value (typically 0.5 or 1.0)
            beta_fixed: Whether to fix beta during calibration
        """
        if not (0 <= beta <= 1):
            raise ValueError("beta must be between 0 and 1")

        self.beta = beta
        self.beta_fixed = beta_fixed

    def calibrate(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        forward: float,
        time_to_expiry: float,
        initial_params: Optional[SABRParams] = None
    ) -> SABRFitResult:
        """
        Calibrate SABR model to observed implied volatilities.

        Preconditions:
            - len(strikes) == len(ivs)
            - forward > 0
            - time_to_expiry > 0
            - All IVs are positive

        Postconditions:
            - Returns SABRFitResult with calibrated parameters
            - Parameters satisfy constraints (beta in [0,1], rho in [-1,1], nu >= 0)

        Args:
            strikes: Strike prices
            ivs: Observed implied volatilities
            forward: Forward price
            time_to_expiry: Time to expiry in years
            initial_params: Optional initial parameter guess

        Returns:
            SABRFitResult object

        Raises:
            CalibrationError: If calibration fails
        """
        if len(strikes) != len(ivs):
            raise ValueError("strikes and ivs must have same length")

        if len(strikes) < 3:
            raise CalibrationError("Need at least 3 strikes for calibration")

        strikes = np.asarray(strikes, dtype=float)
        ivs = np.asarray(ivs, dtype=float)
        if strikes.ndim != 1 or ivs.ndim != 1:
            raise CalibrationError("strikes and ivs must be one-dimensional")
        if not np.isfinite([forward, time_to_expiry]).all() or min(forward, time_to_expiry) <= 0:
            raise CalibrationError("Forward and maturity must be finite and positive")
        if not np.isfinite(strikes).all() or not np.isfinite(ivs).all() or (strikes <= 0).any() or (ivs <= 0).any():
            raise CalibrationError("Strikes and IVs must be finite and positive")
        required = 3 if self.beta_fixed else 4
        if len(np.unique(strikes)) < required:
            raise CalibrationError(f"Need at least {required} distinct strikes for calibration")

        # Initial parameter guess
        if initial_params is None:
            # Rough initial guess
            avg_iv = np.mean(ivs)
            alpha_init = avg_iv * (forward ** (1 - self.beta))
            rho_init = -0.3  # Typical negative correlation
            nu_init = 0.5  # Moderate vol-of-vol
        else:
            alpha_init = initial_params.alpha
            rho_init = initial_params.rho
            nu_init = initial_params.nu
            if not self.beta_fixed:
                self.beta = initial_params.beta

        # Objective function: sum of squared errors
        def objective(params):
            if self.beta_fixed:
                alpha, rho, nu = params
                beta = self.beta
            else:
                alpha, beta, rho, nu = params

            # Constraints
            if alpha <= 0 or nu < 0:
                return 1e10
            if not (0 <= beta <= 1):
                return 1e10
            if not (-1 <= rho <= 1):
                return 1e10

            # Compute model IVs
            try:
                model_ivs = np.array([
                    sabr_implied_vol(forward, k, time_to_expiry, alpha, beta, rho, nu)
                    for k in strikes
                ])
            except ValueError:
                return 1e10

            # Compute error
            errors = model_ivs - ivs
            return np.sum(errors ** 2)

        # Bounds
        if self.beta_fixed:
            bounds = [(1e-6, 10.0), (-0.99, 0.99), (1e-6, 5.0)]  # alpha, rho, nu
            x0 = [alpha_init, rho_init, nu_init]
        else:
            bounds = [(1e-6, 10.0), (0.0, 1.0), (-0.99, 0.99), (1e-6, 5.0)]  # alpha, beta, rho, nu
            x0 = [alpha_init, self.beta, rho_init, nu_init]

        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-10}
            )

            if not result.success:
                raise CalibrationError(f"Optimization failed: {result.message}")

            # Extract parameters
            if self.beta_fixed:
                alpha, rho, nu = result.x
                beta = self.beta
            else:
                alpha, beta, rho, nu = result.x

            # Compute fit error (RMSE)
            model_ivs = np.array([
                sabr_implied_vol(forward, k, time_to_expiry, alpha, beta, rho, nu)
                for k in strikes
            ])
            rmse = np.sqrt(np.mean((model_ivs - ivs) ** 2))

            return SABRFitResult(
                params=SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu),
                fit_error=rmse,
                n_iterations=result.nit,
                success=True
            )

        except Exception as e:
            raise CalibrationError(f"SABR calibration failed: {e}") from e
