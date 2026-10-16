"""
Command-line interface for the analysis tool.

This module provides CLI commands for analyzing assets, running event studies,
and fitting SABR models.
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime, date
from typing import List, Optional
import pandas as pd
import numpy as np

# Suppress yfinance warnings about intraday data (expected behavior)
warnings.filterwarnings("ignore", message=".*1m data not available.*")
warnings.filterwarnings("ignore", message=".*possibly delisted.*")

from backend.entities import Asset, ReturnSeries
from backend.analytics.returns import create_return_series
from backend.data_sources.prices import get_close_prices


from backend.data_sources.macro_events import load_events_from_csv, filter_events_by_name
from backend.data_sources.options import get_option_chain, option_time_to_expiry
from backend.analytics.factor_model import FactorModel
from backend.analytics.corr_radar import CorrelationRadar
from backend.analytics.event_impact import EventImpactStudy
from backend.entities import Asset
from backend.analytics.vol_surface import clean_iv_smile
from backend.analytics.sabr import SABRModel
from backend.cache import DataCache
from backend.reporting.report import Report
from backend.errors import DataError, CalibrationError


def analyze_command(args):
    """Run full analysis on a ticker."""
    ticker = args.ticker
    start_date = args.start
    end_date = args.end

    print(f"Analyzing {ticker} from {start_date} to {end_date}...")

    # Initialize cache
    cache = DataCache(".cache")

    try:
        # Create asset
        asset = Asset(ticker=ticker, name=ticker, asset_type="equity")

        # Download prices
        print("  Downloading price data...")
        prices = get_close_prices(ticker, start_date, end_date, cache=cache)
        asset_returns = create_return_series(prices, asset=asset)
        # Academic factors and RF are arithmetic daily returns. Preserve the
        # existing log-return convention separately for correlation analysis.
        factor_returns = create_return_series(prices, asset=asset, method="simple")

        # Factor analysis
        print("  Running factor analysis...")
        from backend.data_sources.factors import get_factor_data
        
        factor_data = get_factor_data(start_date, end_date, cache=cache)
        
        # Ensure factor data index matches asset returns index (timezone handling)
        if asset_returns.dates.tz is not None and factor_data.index.tz is None:
            factor_data.index = factor_data.index.tz_localize(asset_returns.dates.tz)
        elif asset_returns.dates.tz is None and factor_data.index.tz is not None:
             factor_data.index = factor_data.index.tz_localize(None)
        
        # Align returns with factor data
        common_index = asset_returns.dates.intersection(factor_data.index)
        
        if len(common_index) < 30:
            print(f"    Warning: Only {len(common_index)} overlapping days found")
            
        asset_returns_adj = factor_returns.returns.loc[common_index]
        factor_data = factor_data.loc[common_index] # Reindex specific columns
        
        # Update asset_returns object
        factor_returns = ReturnSeries(common_index, asset_returns_adj, asset=asset_returns.asset)
        
        factor_model = FactorModel(["Market", "Value", "Size", "Momentum"])
        factor_result = factor_model.fit(factor_returns, factor_data)

        print(f"    R² = {factor_result.r_squared:.3f}")
        for factor_name in factor_result.factor_names:
            beta = factor_result.betas[factor_name]
            print(f"    {factor_name} β = {beta:.3f}")

        # Correlation analysis (scan against watchlist assets)
        print("  Scanning for correlation anomalies...")
        anomalies = []
        scan_coverage = {'attempted': 0, 'successful': 0, 'skipped': 0, 'failed': 0}
        try:
            # Load watchlist for comparison - use multiple lists for broad coverage
            import yaml
            watchlist_path = Path(__file__).parent / "data" / "watchlists.yaml"
            
            comparison_tickers = []
            
            if watchlist_path.exists():
                with open(watchlist_path) as f:
                    watchlists = yaml.safe_load(f)
                    # Combine default, sector ETFs, and macro assets
                    comparison_tickers.extend(watchlists.get("default", []))
                    comparison_tickers.extend(watchlists.get("sector_etfs", []))
                    comparison_tickers.extend(watchlists.get("macro", []))
                    # Add some S&P 100 stocks for single-stock correlations
                    sp100 = watchlists.get("sp100", [])
                    comparison_tickers.extend(sp100[:20])  # Top 20 S&P 100
                    # Deduplicate
                    comparison_tickers = list(dict.fromkeys(comparison_tickers))
            else:
                 print("    Warning: watchlists.yaml not found, skipping correlation scan")
            
            # Exclude self and VIX (no direct price data)
            comparison_tickers = [t for t in comparison_tickers if t != ticker and t != "VIX"]
            
            print(f"    Comparing against {len(comparison_tickers)} assets...")
            
            successful_comparisons = 0
            if len(comparison_tickers) > 0:
                # Get comparison asset returns
                for comp_ticker in comparison_tickers:
                    scan_coverage['attempted'] += 1
                    try:
                        comp_prices = get_close_prices(comp_ticker, start_date, end_date, cache=cache)
                        
                        # correlation points = len(prices) - window_size + 1
                        # We need min_history correlation points
                        # So len(prices) >= min_history + window_size - 1
                        required_history = 60 + 30
                        
                        if len(comp_prices) < required_history:
                            scan_coverage['skipped'] += 1
                            continue
                            
                        comp_asset = Asset(ticker=comp_ticker, name=comp_ticker, asset_type="equity")
                        comp_returns = create_return_series(comp_prices, asset=comp_asset)
                        
                        # window_size=30 (1.5 months), min_history=60 (3 months valid correlations)
                        radar = CorrelationRadar(window_size=30, min_history=60, z_threshold=2.0)
                        if len(asset_returns.dates.intersection(comp_returns.dates)) < radar.min_history + radar.window_size - 1:
                            scan_coverage['skipped'] += 1
                            continue
                        pair_anomalies = radar.detect_anomalies(asset_returns, comp_returns)
                        anomalies.extend(pair_anomalies)
                        successful_comparisons += 1
                        scan_coverage['successful'] += 1
                        
                        if len(pair_anomalies) > 0:
                            print(f"    ✓ {comp_ticker}: {len(pair_anomalies)} anomalies")
                    except Exception as e:
                        scan_coverage['failed'] += 1
                        print(f'    Failed comparison {comp_ticker}: {e}')
                        continue
            
            print(f"    Scanned {successful_comparisons} pairs, found {len(anomalies)} total anomalies")
        except Exception as e:
            # If correlation analysis fails, continue without it
            scan_coverage['error'] = str(e)
            print(f"    Correlation analysis failed: {e}")

        # Event impacts from the verified calendar within the selected range.
        print("  Analyzing macro event impacts...")
        from backend.data_sources.macro_events import get_events_for_analysis
        
        # Uses daily data for older events and intraday data when available.
        recent_events = get_events_for_analysis(start_date, end_date)
        
        # All macro event types
        all_macro_events = [e for e in recent_events if e.name in ["CPI", "NFP", "FOMC"]]
        
        # Sort by most recent first
        all_macro_events.sort(key=lambda x: x.timestamp, reverse=True)

        event_study = EventImpactStudy()
        event_impacts = []
        
        # Analyze up to 10 most recent events (uses hybrid intraday/daily approach)
        for event in all_macro_events[:10]:
            try:
                impact = event_study.analyze_event(ticker, event, cache=cache, analysis_end=end_date)
                event_impacts.append(impact)
                data_source = "intraday" if impact.data_type == "intraday" else "daily"
                print(f"    ✓ {event.name} on {event.timestamp.strftime('%Y-%m-%d')} ({data_source} data)")
            except (DataError, Exception) as e:
                # Skip if data unavailable
                continue
        
        if len(event_impacts) == 0:
            print("    No verified event impacts available for this range (partial calendar)")
        else:
            print(f"    Analyzed {len(event_impacts)} events total")

        # Current options are fetched only when explicitly requested.
        sabr_result = None
        sabr_status = "disabled" if not args.fit_sabr else "unavailable"
        options_snapshot_time = None
        sabr_strikes = None
        sabr_ivs = None
        sabr_forward = None
        sabr_time_to_expiry = None

        should_try_sabr = args.fit_sabr
        
        
        if should_try_sabr:
            print("  Fitting SABR model...")
            try:
                option_chain = get_option_chain(ticker)  # Live only
                options_snapshot_time = pd.Timestamp.now(tz="UTC").isoformat()
                # Clean IV smile
                strikes, call_ivs = clean_iv_smile(
                    option_chain.strikes,
                    option_chain.call_ivs,
                    option_chain.spot_price
                )
                # Fit SABR
                forward = option_chain.spot_price
                time_to_expiry = option_time_to_expiry(option_chain.expiry_date)
                sabr_model = SABRModel(beta=0.5, beta_fixed=True)
                sabr_result = sabr_model.calibrate(
                    strikes, call_ivs, forward, time_to_expiry
                )
                sabr_status = "completed; optimizer convergence does not establish pricing validity"
                sabr_strikes = strikes
                sabr_ivs = call_ivs
                sabr_forward = forward
                sabr_time_to_expiry = time_to_expiry
                print(f"    ✓ SABR fit RMSE = {sabr_result.fit_error:.6f}")
            except (DataError, CalibrationError, Exception) as e:
                sabr_status = f"failed: {e}"
                print(f"    ✗ SABR fit failed: {e}")


        # Generate report
        print("  Generating report...")
        report = Report()
        report_path = report.generate_report(
            ticker=ticker,
            asset=asset,
            factor_result=factor_result,
            anomalies=anomalies,
            event_impacts=event_impacts,
            sabr_result=sabr_result,
            sabr_strikes=sabr_strikes,
            sabr_ivs=sabr_ivs,
            sabr_forward=sabr_forward,
            sabr_time_to_expiry=sabr_time_to_expiry,
            run_id=getattr(args, 'run_id', None),
            analysis_metadata={
                'Requested price/factor range': f'{start_date} through {end_date} (end exclusive)',
                'Aligned factor dates (before missing-data filtering)': f'{common_index.min().date()} through {common_index.max().date()}',
                'Factor return convention': 'arithmetic daily excess returns; RF subtracted once',
                'Correlation return convention': 'log daily returns',
                'SABR requested': bool(args.fit_sabr),
                'SABR status': sabr_status,
                'Correlation scan coverage': scan_coverage,
                'Calendar coverage': 'Bundled calendar is a partial October–December 2024 sample; missing coverage is not evidence of no events',
                'Options timing': 'Current snapshot only; independent of the selected historical price range',
                'Options assumptions': 'Spot used as forward; date-only expiries approximated at 16:00 America/New_York, ACT/365',
                'Options retrieved at': options_snapshot_time or 'Not retrieved',
                'Options expiry': str(option_chain.expiry_date) if sabr_result else 'Not calibrated',
            }
        )

        print(f"\n✓ Analysis complete!")
        print(f"  Report saved to: {report_path}")

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


def event_study_command(args):
    """Run event study for specific events."""
    event_name = args.event
    tickers = args.assets.split(",") if args.assets else ["SPY"]

    print(f"Running event study for {event_name} on {', '.join(tickers)}...")

    cache = DataCache(".cache")
    events = load_events_from_csv()
    filtered_events = filter_events_by_name(events, event_name)

    if not filtered_events:
        print(f"No events found for {event_name}")
        return

    print(f"Found {len(filtered_events)} {event_name} events")

    event_study = EventImpactStudy()
    all_impacts = []

    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        impacts = event_study.analyze_multiple_events(
            ticker,
            filtered_events[:5],  # Limit to first 5
            cache=cache
        )
        all_impacts.extend(impacts)
        print(f"  Analyzed {len(impacts)} events")

    if all_impacts:
        summary = event_study.summarize_impacts(all_impacts)
        print("\n" + summary.to_string())
    else:
        print("No event impacts computed (intraday data unavailable)")


def fit_sabr_command(args):
    """Fit SABR model to option chain."""
    ticker = args.ticker
    expiry = args.expiry

    print(f"Fitting SABR model for {ticker} (expiry: {expiry or 'nearest'})...")

    try:
        option_chain = get_option_chain(ticker, expiry=expiry)

        print(f"  Spot price: ${option_chain.spot_price:.2f}")
        print(f"  Expiry: {option_chain.expiry_date.date()}")
        print(f"  Strikes: {len(option_chain.strikes)}")

        # Clean IV smile
        strikes, call_ivs = clean_iv_smile(
            option_chain.strikes,
            option_chain.call_ivs,
            option_chain.spot_price
        )

        print(f"  Valid strikes after cleaning: {len(strikes)}")

        # Fit SABR
        forward = option_chain.spot_price
        time_to_expiry = option_time_to_expiry(option_chain.expiry_date)

        sabr_model = SABRModel(beta=0.5, beta_fixed=True)
        result = sabr_model.calibrate(strikes, call_ivs, forward, time_to_expiry)

        print(f"\n✓ SABR calibration complete!")
        print(f"  α (alpha): {result.params.alpha:.4f}")
        print(f"  β (beta): {result.params.beta:.2f}")
        print(f"  ρ (rho): {result.params.rho:.2f}")
        print(f"  ν (nu): {result.params.nu:.4f}")
        print(f"  RMSE: {result.fit_error:.6f}")

    except (DataError, CalibrationError) as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stock Factor Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run full analysis on a ticker")
    analyze_parser.add_argument("ticker", help="Ticker symbol")
    analyze_parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    analyze_parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD, default: today)")
    analyze_parser.add_argument("--fit-sabr", action="store_true", help="Fit SABR model")
    analyze_parser.add_argument("--run-id", help=argparse.SUPPRESS)

    # Event study command
    event_parser = subparsers.add_parser("event-study", help="Run event study")
    event_parser.add_argument("--event", default="CPI", help="Event name (default: CPI)")
    event_parser.add_argument("--assets", help="Comma-separated asset list (default: SPY)")

    # Fit SABR command
    sabr_parser = subparsers.add_parser("fit-sabr", help="Fit SABR model to option chain")
    sabr_parser.add_argument("ticker", help="Ticker symbol")
    sabr_parser.add_argument("--expiry", help="Expiry date (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.command == "analyze":
        if args.end is None:
            args.end = date.today().strftime("%Y-%m-%d")
        analyze_command(args)
    elif args.command == "event-study":
        event_study_command(args)
    elif args.command == "fit-sabr":
        fit_sabr_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
