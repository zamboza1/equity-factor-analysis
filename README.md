# Equity Factor Analysis

An interactive equity risk analyzer that uses historical market prices and Fama-French factors to model beta exposures, correlation anomalies, macro event impacts, and volatility smiles. It features a Python FastAPI backend for statistical regression and SABR calibration, and a React frontend to visualize risk metrics.

## How It Works
**Backend:** Python FastAPI handling the data pipeline. It conducts price fetching (yfinance), regression modeling, and SABR analytics.

**Frontend:** React + Tailwind CSS. A web interface to explore risk metrics, factor betas, and volatility smiles.

## Engineering Standards
**Architecture:** Modular design separating `backend` logic from `frontend` presentation.


**Resilience:** Implements error handling for background processes.

**Mathematics:** Implements standard OLS regressions for factor modeling and non-linear least squares for SABR calibration.

## Run It Yourself
```bash
./start_web.sh
```

- **Options Data**: Uses yfinance. Some stocks might not have data available.

## Testing

```bash
pytest  # Runs comprehensive test suite
```

## License

MIT - Built for learning. Not financial advice.
