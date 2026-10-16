# Verification

Checked October 16, 2026 for version 0.1.0, using the source changes accompanying
this record. The starting repository revision was
`8c4d7e9ef70d8f191612f686abc96dc0421cd753`.
The tested code, resources and tests have source-manifest SHA-256
`ef30a314f8b44cd3a51236a84b9d232f21cac723e2690df5213c990249f4fe3c`.

## Environment

macOS 14.6.1 arm64; Python 3.12.7; NumPy 1.26.4; pandas 2.2.2; SciPy 1.13.1;
statsmodels 0.14.2; FastAPI 0.128.0; yfinance 0.2.65; pandas-datareader 0.10.0;
pytest 7.4.4. Existing dependencies were reused. Project wheels were built without
dependency installation and installed into separate temporary directories.

Browser checks used the Codex in-app browser on dedicated loopback ports, with
desktop and 390 × 844 viewports. Its engine version was not exposed. These checks
do not establish compatibility with other browsers or physical mobile devices.

## Automated and package checks

The saved correction baseline passed 114 tests. Eighteen added data-contract
cases initially produced 15 failures, covering unidentified regressions, invalid
observations, date alignment and cache recovery. A separate Friday-event case
reproduced an exclusive-end fetch that omitted the next trading session.

The final suite passed **134 tests**, with five dependency deprecation warnings.
Checks cover independent SABR values, factor inference, returns, date boundaries,
per-job reporting, cache replacement and installed resources. The rebuilt wheel
contains the analytics modules, templates, calendar and watchlist. Outside the
source directory, its React page, calendar loader and both command entry points
worked. The installed wheel also completed the live browser workflow below.

## Browser checks

Synthetic provider inputs exercised the actual application and analysis worker.
These were fixtures, not market observations.

| Scenario | Observed result |
| --- | --- |
| Complete analysis, charts and full-report navigation | Factors, correlation alerts, dated events and SABR chart rendered; report link and return navigation worked. |
| Change submitted inputs | Previous results cleared. The baseline had retained the previous ticker's results. |
| Pending analysis | Ticker, dates, shortcuts, options and submit controls disabled. A 72.77-second fixture completed automatically. |
| Invalid ticker, reversed dates, excessive date range | Readable validation; HTTP 422/400 where applicable. |
| Short history, provider failure, collinear factors | Specific errors rendered; subsequent submissions remained available. |
| No events or unavailable options | Empty coverage and options failure were explicit; available analysis remained visible. |
| Narrow layout | Document width remained 390 px; the factor canvas was 308 px wide; wide tables scrolled within their containers. |

No browser errors appeared in the checked flows. Console warnings identified the
existing Tailwind CDN and in-browser Babel development setup. Screenshots were
visually inspected during testing; no exported screenshot archive accompanies
this repository.

The main fixture's coefficients matched independent least squares within
`9.5e-16`; t-statistics within `1.8e-13`. Forty-three correlation-score comparisons
were within `1.9e-10`. A synthetic SABR smile fitted with IV RMSE `1.69e-9`.

## Live data

AAPL, January 1, 2024 through January 1, 2025 exclusive, completed with 251 factor
observations, R² `0.3415703674`, 42 successful comparison scans, 3,017 pair/date
alerts and eight covered macro events. Independent calculations using recorded
Yahoo prices and raw French factors agreed with coefficients within `2.5e-16`
and ten displayed AAPL–GOOGL alert scores within `2.0e-14`.

All eight daily event calculations agreed within `1e-6` log-return units with an
independent full-year price series. Separate adjusted-price download windows
differed slightly at floating-point precision. After the Friday correction, the
November 1 cumulative return included November 4 (`−0.0174143`), and December 6
included December 9 (`0.0151496`). Both the React table and full report displayed
the corrected results; the dated chart loaded.

A separate live current-options run produced a **poor near-expiry SABR fit**:
RMSE `1.369819`, alpha `10` and nu `5` at their bounds, maturity `0.000323` years.
Its parameters and RMSE were visible. This does not validate quote quality or
pricing accuracy. The final event-only rerun disabled options; the SABR code was
unchanged between those runs.

## Limits

Live checks cover one historical equity interval and one current option snapshot.
Recent intraday event handling was tested with fixtures, not live release data.
There was no exhaustive browser, accessibility, concurrency, provider-outage or
deployment test. No exchange-calendar completeness check is performed. Static
Markdown reports round displayed numbers; JSON companions retain precision.

The partial calendar, ordinary OLS inference, descriptive correlation thresholds,
current options, quote filtering and restart behavior remain as documented in the
[README](../README.md) and [model conventions](model-conventions.md).
