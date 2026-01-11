"""
Chart generation for reports.

This module creates matplotlib charts for factor models, correlations,
event impacts, and volatility surfaces.
"""

from typing import Optional, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from backend.analytics.factor_model import FactorFitResult
from backend.analytics.corr_radar import CorrelationAnomaly
from backend.analytics.event_impact import EventImpact
from backend.analytics.sabr import SABRFitResult, sabr_implied_vol


def plot_factor_residuals(
    asset_returns: pd.Series,
    fitted_returns: pd.Series,
    save_path: str
) -> None:
    """
    Plot factor model residuals (actual vs fitted returns).

    Args:
        asset_returns: Actual asset returns
        fitted_returns: Fitted returns from factor model
        save_path: Path to save chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Align series
    common_idx = asset_returns.index.intersection(fitted_returns.index)
    asset_aligned = asset_returns.reindex(common_idx)
    fitted_aligned = fitted_returns.reindex(common_idx)

    # Plot cumulative returns
    cum_actual = (1 + asset_aligned).cumprod()
    cum_fitted = (1 + fitted_aligned).cumprod()

    ax.plot(cum_actual.index, cum_actual.values, label="Actual", linewidth=2)
    ax.plot(cum_fitted.index, cum_fitted.values, label="Fitted", linewidth=2, linestyle="--")

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Factor Model: Actual vs Fitted Returns")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rolling_correlation(
    dates: pd.DatetimeIndex,
    correlations: pd.Series,
    anomalies: List[CorrelationAnomaly],
    pair_name: str,
    save_path: str
) -> None:
    """
    Plot rolling correlation with anomaly markers.

    Args:
        dates: Date index
        correlations: Rolling correlation series
        anomalies: List of CorrelationAnomaly objects
        pair_name: Asset pair name (e.g., "AAPL-MSFT")
        save_path: Path to save chart
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot correlation
    ax.plot(dates, correlations.values, label="Rolling Correlation", linewidth=2, color="blue")

    # Mark anomalies
    if anomalies:
        anomaly_dates = [a.date for a in anomalies]
        anomaly_corrs = [a.correlation for a in anomalies]
        ax.scatter(
            anomaly_dates, anomaly_corrs,
            color="red", s=100, marker="x", zorder=5,
            label=f"Anomalies (|z| >= 2.0)"
        )

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Rolling Correlation: {pair_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_event_impact(
    impacts: List[EventImpact],
    save_path: str
) -> None:
    """
    Plot event impact (pre/post returns) for multiple events.

    Args:
        impacts: List of EventImpact objects
        save_path: Path to save chart
    """
    if not impacts:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by event name
    event_names = list(set(i.event.name for i in impacts))
    tickers = list(set(i.ticker for i in impacts))

    x_pos = np.arange(len(event_names))
    width = 0.35

    # Plot pre and post returns for each event
    for ticker in tickers:
        ticker_impacts = [i for i in impacts if i.ticker == ticker]
        pre_returns = [i.pre_event_return for i in ticker_impacts]
        post_returns = [i.post_event_return for i in ticker_impacts]

        if len(pre_returns) == len(event_names):
            ax.bar(x_pos - width/2, pre_returns, width, label=f"{ticker} Pre", alpha=0.7)
            ax.bar(x_pos + width/2, post_returns, width, label=f"{ticker} Post", alpha=0.7)

    ax.set_xlabel("Event")
    ax.set_ylabel("Return")
    ax.set_title("Event Impact: Pre vs Post Event Returns")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(event_names, rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sabr_fit(
    strikes: np.ndarray,
    observed_ivs: np.ndarray,
    sabr_result: SABRFitResult,
    forward: float,
    time_to_expiry: float,
    save_path: str
) -> None:
    """
    Plot SABR fit to implied volatility smile.

    Args:
        strikes: Strike prices
        observed_ivs: Observed implied volatilities
        sabr_result: SABRFitResult from calibration
        forward: Forward price
        time_to_expiry: Time to expiry in years
        save_path: Path to save chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot observed IVs
    ax.scatter(strikes, observed_ivs, label="Observed IV", s=50, alpha=0.7, color="blue")

    # Plot SABR fit
    strike_range = np.linspace(strikes.min(), strikes.max(), 100)
    sabr_ivs = np.array([
        sabr_implied_vol(
            forward, k, time_to_expiry,
            sabr_result.params.alpha, sabr_result.params.beta,
            sabr_result.params.rho, sabr_result.params.nu
        )
        for k in strike_range
    ])

    ax.plot(strike_range, sabr_ivs, label="SABR Fit", linewidth=2, color="red")

    # Mark ATM
    ax.axvline(x=forward, color="gray", linestyle="--", alpha=0.5, label="ATM")

    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(f"SABR Fit (RMSE={sabr_result.fit_error:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_report_assets_dir(report_dir: Path) -> Path:
    """
    Create assets directory for report charts.

    Args:
        report_dir: Report directory path

    Returns:
        Path to assets directory
    """
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir


