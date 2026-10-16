# Equity Factor Analysis

A local equity research tool for factor exposures, rolling correlations, macro-event returns and option volatility smiles. Run an analysis in the browser or CLI and save a report with its dates, assumptions and data coverage.

## How It Works

**Factors:** OLS estimates Market, Value, Size and Momentum exposures using daily arithmetic excess returns and the Kenneth French research factors. Reports include coefficients, t-statistics, p-values, R² and the observation count.

**Correlations:** Scan a bundled watchlist using 30-observation rolling correlations of log returns. Alerts describe departures from an initial empirical baseline. Reports distinguish successful, skipped and failed comparisons.

**Macro events:** Match the selected interval to a sourced calendar and calculate event-window returns. The bundled calendar is a partial October–December 2024 sample. Daily close-to-close proxies are used when intraday data is unavailable.

**Options:** Optionally fit a Hagan lognormal SABR smile to the nearest available current call-IV snapshot. The chart shows observed and fitted IV, with parameters, maturity and RMSE.

**Application:** FastAPI runs the analysis workers; React displays their results. Each run owns its report and chart files. Invalid inputs and worker failures are shown explicitly.

## Run It Yourself

Use Python 3.11 or later.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
analysis-web --port 8000
```

Open [the local application](http://127.0.0.1:8000). It has no authentication and binds to loopback. The browser loads React, Chart.js, Tailwind and Babel from public CDNs, so an internet connection is required.

```sh
analysis analyze AAPL --start 2024-01-01 --end 2025-01-01
```

The end date is exclusive. Add `--fit-sabr` to include current options; that snapshot is independent of the historical price interval. `analysis --help` lists the other commands. From a source checkout, `bash start_web.sh 8000` starts the development server with reload enabled and refuses an occupied port.

Reports, JSON results, charts and worker logs are written to `reports/` in the working directory; downloaded prices are cached in `.cache/`. Both are excluded from Git. Only use trusted local cache files. Active web jobs are held in memory and do not resume after a server restart; completed reports remain on disk.

## Model Limits

Factor estimates describe the fitted sample, not future performance. Missing observations are excluded; non-finite or unidentified fits are rejected. Default OLS inference does not adjust for heteroskedasticity, serial dependence or multiple comparisons.

Correlation thresholds are descriptive alerts, not calibrated confidence intervals. Event returns are neither causal effects nor market-adjusted abnormal returns. Missing calendar or provider coverage does not establish that nothing happened.

SABR uses β=0.5, spot as a forward approximation, ACT/365 and 16:00 Eastern for date-only expiries. It fits one expiry, not a complete volatility surface. Convergence and low RMSE do not establish liquidity, arbitrage-free prices or forecast accuracy. See [Model conventions](docs/model-conventions.md) for definitions and sources.

## Testing

```sh
python -m pytest tests -q
python -m pip wheel . --no-deps --wheel-dir dist
```

Tests use synthetic inputs and mocked provider boundaries. Separate live-provider and browser checks are recorded in [Verification](docs/verification.md).

## License

[MIT](LICENSE) - Built for learning. Not financial advice.
