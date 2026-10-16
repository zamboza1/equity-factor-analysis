# Model conventions

## Data and return intervals

Yahoo Finance supplies adjusted price history through yfinance. Price and factor
ranges include the start and exclude the end. Prices are sorted by their actual
timestamps before returns are calculated. Duplicate or invalid price dates are
rejected. Explicit missing prices do not create a return across that interval;
absent date rows are not checked against an exchange-session calendar.
Labeled return series retain their date labels when reordered.

Daily factor data comes from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html).
Provider percentages are divided by 100. Daily period labels and date indices are
normalized before joining. The factors are Market (Mkt−RF), Value (HML), Size
(SMB) and Momentum (MOM). Revised provider histories can change a later rerun;
the application is not a point-in-time data archive.

## Factor regression

The fitted equation is `asset arithmetic return − RF = alpha + factors × betas + error`.
RF is subtracted once; Mkt−RF is already an excess return. Observations align by
date and rows missing required inputs are excluded. A fit requires at least 30
valid observations, positive residual degrees of freedom, a full-rank design and
finite reported statistics. Constant excess returns and constant or collinear
regressors are rejected.

Inference uses ordinary, nonrobust OLS standard errors. The two-sided p-value
tests zero exposure under the model assumptions. It does not measure the
probability that a result is random, and it is not adjusted for serial dependence,
heteroskedasticity or multiple testing. R² describes the fitted sample.
[statsmodels documents the rank-deficiency issue](https://www.statsmodels.org/stable/pitfalls.html#rank-deficient-exog-perfect-multicollinearity)
that motivates checking the design before fitting.

## Correlation alerts

The CLI uses log returns, a 30-observation rolling correlation and at least 60
valid rolling-correlation observations. The first 30 valid correlations form a
fixed reference mean and sample standard deviation. Later deviations are divided
by that standard deviation; absolute scores of at least two trigger an alert.

These overlapping windows are not independent observations. Scores are
descriptive, with no calibrated null distribution or multiple-testing correction.
The same trading date can produce several pair alerts. A baseline with zero
dispersion cannot produce a meaningful standardized score. Missing, failed and
skipped comparisons must not be interpreted as normal correlations.

## Macro-event returns

The bundled CSV contains eight sourced CPI, employment and FOMC release times
from part of October–December 2024. Source URLs accompany the rows. It is not a
complete or current economic calendar. Up to ten covered events within the
requested interval are analyzed.

For recent events, the application requests minute prices, including extended
hours, and clips returned bars to the requested window. Older or unavailable
intraday observations use daily close-to-close log-return proxies. Values are
decimal log returns: `0.01` is a one-percent log return. Unavailable intraday
pre-event measurements are not displayed as measured zero bars. Prices after the
analysis end are excluded. Daily cumulative returns extend from the prior close
through the next available session, or through the event close when the bounded
window has no later price. The daily fetch allows seven calendar days after the
release while respecting the selected end date. These returns are not causal or
benchmark-adjusted event effects.

## SABR smile

The optional fit uses the current nearest-expiry call-IV snapshot, independently
of the historical price range. The implementation uses the Hagan lognormal
implied-volatility approximation, checked against independent values and
[QuantLib's implementation](https://github.com/lballabio/QuantLib/blob/master/ql/termstructures/volatility/sabr.cpp).
Small-z limits handle ATM, beta=1 and nu=0. Invalid maturities and parameters are
rejected. Beta is fixed at 0.5 in the CLI; alpha, rho and nu are calibrated.

Alpha is a volatility scale, rho is the model forward/volatility shock
correlation, and nu is volatility of volatility. Spot approximates the forward;
no financing or dividend curve is estimated. Maturity is fractional ACT/365.
Date-only expiries assume 16:00 America/New_York rather than contract-specific
settlement. Expired contracts are rejected instead of receiving a fallback tenor.

Only one expiry is fitted. Quote cleaning is limited to the implemented strike
and IV filters; the fit is unweighted. There is no bid/ask, liquidity, static
arbitrage or cross-expiry validation. Optimizer convergence and IV RMSE measure
numerical fit, not economic validity, crash probability or forecast performance.
