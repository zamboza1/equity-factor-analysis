"""
Markdown report generation.

This module generates Markdown reports with factor analysis,
correlation anomalies, event impacts, and SABR fits.
"""

from datetime import datetime
from pathlib import Path
from uuid import uuid4
import json
from typing import List, Optional
import pandas as pd
import numpy as np
from backend.analytics.factor_model import FactorFitResult, FactorModel
from backend.analytics.corr_radar import CorrelationAnomaly, CorrelationRadar
from backend.analytics.event_impact import EventImpact, EventImpactStudy
from backend.analytics.sabr import SABRFitResult
from backend.entities import Asset, ReturnSeries
from backend.reporting.charts import (
    plot_factor_residuals, plot_rolling_correlation,
    plot_event_impact, plot_sabr_fit, create_report_assets_dir
)


class Report:
    """
    Generates markdown reports with analysis results.

    This class assembles factor models, correlation anomalies, event impacts,
    and SABR fits into a Markdown report with charts.
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        ticker: str,
        asset: Optional[Asset],
        factor_result: Optional[FactorFitResult],
        anomalies: List[CorrelationAnomaly],
        event_impacts: List[EventImpact],
        sabr_result: Optional[SABRFitResult] = None,
        sabr_strikes: Optional[np.ndarray] = None,
        sabr_ivs: Optional[np.ndarray] = None,
        sabr_forward: Optional[float] = None,
        sabr_time_to_expiry: Optional[float] = None,
        run_id: Optional[str] = None,
        analysis_metadata: Optional[dict] = None
    ) -> str:
        """
        Generate complete markdown report.

        Args:
            ticker: Asset ticker
            asset: Asset object
            factor_result: Factor model fit result
            anomalies: List of correlation anomalies
            event_impacts: List of event impacts
            sabr_result: Optional SABR fit result
            sabr_strikes: Optional strike prices for SABR plot
            sabr_ivs: Optional observed IVs for SABR plot
            sabr_forward: Optional forward price for SABR
            sabr_time_to_expiry: Optional time to expiry for SABR

        Returns:
            Path to generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        identity = run_id or f'{timestamp}_{uuid4().hex[:8]}'
        if not identity or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in identity):
            raise ValueError('Invalid run ID')
        report_filename = f"{ticker}_{identity}.md"
        report_path = self.output_dir / report_filename

        # Create assets directory
        assets_dir = create_report_assets_dir(self.output_dir) / identity
        assets_dir.mkdir(parents=True, exist_ok=False)

        # Generate report content
        content = self._generate_header(ticker, asset)
        if analysis_metadata:
            content += '\n'.join(f'**{key}:** {value}  ' for key,value in analysis_metadata.items()) + '\n\n'
        content += self._generate_factor_section(factor_result, assets_dir, ticker)
        content += self._generate_correlation_section(anomalies, assets_dir)
        content += self._generate_event_section(event_impacts, assets_dir)
        content += self._generate_sabr_section(
            sabr_result, sabr_strikes, sabr_ivs,
            sabr_forward, sabr_time_to_expiry, assets_dir
        )
        content += self._generate_footer()

        # Keep numerical precision and ticker identity outside Markdown parsing.
        # The Markdown remains the human-readable artifact, with an atomic
        # structured companion used by newly completed browser jobs.
        top = CorrelationRadar().get_top_anomalies(anomalies, n=10) if anomalies else []
        structured = {
            'ticker': ticker,
            'r_squared': factor_result.r_squared if factor_result else None,
            'n_observations': factor_result.n_observations if factor_result else None,
            'factors': [{'Factor': name, 'Beta': factor_result.betas[name],
                         't-stat': factor_result.t_stats[name],
                         'p-value': factor_result.p_values[name]}
                        for name in factor_result.factor_names] if factor_result else [],
            'anomalies': len(anomalies),
            'anomalies_data': [{'Date': a.date.strftime('%Y-%m-%d'),
                               'Pair': '-'.join(a.pair), 'Correlation': a.correlation,
                               'Z-Score': a.z_score} for a in top],
            'events': len(event_impacts),
            'events_data': [{'Ticker': e.ticker, 'Event': e.event.name,
                            'Date': e.event.timestamp.strftime('%Y-%m-%d'),
                            'Post-Return': e.post_event_return,
                            'Cumulative': e.cumulative_return, 'Data': e.data_type}
                           for e in event_impacts],
            'sabr': vars(sabr_result.params) if sabr_result else None,
            'sabr_rmse': sabr_result.fit_error if sabr_result else None,
            'sabr_maturity': sabr_time_to_expiry,
            'analysis_metadata': analysis_metadata or {},
            'event_chart_url': f'/assets/{identity}/event_impact.png' if (assets_dir/'event_impact.png').exists() else None,
            'sabr_chart_url': f'/assets/{identity}/sabr_fit.png' if (assets_dir/'sabr_fit.png').exists() else None,
        }
        json_temporary = report_path.with_suffix('.json.tmp')
        json_temporary.write_text(json.dumps(structured, indent=2, default=float, allow_nan=False))
        json_temporary.replace(report_path.with_suffix('.json'))

        # Write report
        temporary_path = report_path.with_suffix('.tmp')
        with open(temporary_path, "w") as f:
            f.write(content)
        temporary_path.replace(report_path)

        return str(report_path)

    def _generate_header(self, ticker: str, asset: Optional[Asset]) -> str:
        """Generate report header."""
        asset_name = asset.name if asset else ticker
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""# Risk Analysis Report: {asset_name}

**Ticker:** {ticker}  
**Generated:** {timestamp}

---

"""

    def _generate_factor_section(
        self,
        factor_result: Optional[FactorFitResult],
        assets_dir: Path,
        ticker: str
    ) -> str:
        """Generate factor model section."""
        if factor_result is None:
            return "## Factor Exposures\n\n*No factor analysis available.*\n\n---\n\n"

        section = "## Factor Exposures\n\n"
        
        # Explanation
        section += "> **What are Factor Betas?** Factor betas measure how much a stock's returns "
        section += "move with common risk factors. They are computed via OLS regression of the stock's "
        section += "returns against factor returns. A beta of 1.0 is a fitted one-unit excess-return response to a one-unit factor change, holding the other factors fixed.\n\n"
        
        section += "**Factor Definitions (Academic Fama-French):**\n"
        section += "- **Market (Mkt-RF)**: Excess return on the market portfolio (CRSP universe minus Risk-Free Rate)\n"
        section += "- **Value (HML)**: High Minus Low - return spread between value and growth stocks\n"
        section += "- **Size (SMB)**: Small Minus Big - return spread between small-cap and large-cap stocks\n"
        section += "- **Momentum (MOM)**: Difference between recent winners and recent losers\n\n"
        
        section += "### Factor Betas\n\n"
        section += "| Factor | Beta | t-statistic | p-value | Significant? |\n"
        section += "|--------|------|-------------|----------|---------------|\n"
        
        for factor_name in factor_result.factor_names:
            beta = factor_result.betas[factor_name]
            t_stat = factor_result.t_stats[factor_name]
            p_val = factor_result.p_values[factor_name]
            significant = "✓ Yes" if p_val < 0.05 else "No"
            section += f"| {factor_name} | {beta:.4f} | {t_stat:.2f} | {p_val:.4f} | {significant} |\n"
        
        section += "\n"
        section += f"**R²:** {factor_result.r_squared:.3f} "
        section += f"*(proportion of variance explained by the factor model)*  \n"
        section += f"**Observations:** {factor_result.n_observations}  \n\n"
        
        # Interpretation
        model = FactorModel(factor_result.factor_names)
        interpretation = model.interpret(factor_result)
        section += f"**Interpretation:** {interpretation}\n\n"
        
        section += "> *Statistical significance uses p-value < 0.05 for a two-sided test "
        section += "against zero exposure under the fitted OLS assumptions.*\n\n"
        section += "---\n\n"

        return section

    def _generate_correlation_section(
        self,
        anomalies: List[CorrelationAnomaly],
        assets_dir: Path
    ) -> str:
        """Generate correlation anomaly section."""
        section = "## Correlation Anomalies\n\n"
        
        # Explanation
        section += "> **What are Correlation Anomalies?** These are dates when the rolling correlation "
        section += "between this stock and another asset deviated from the initial empirical baseline. "
        section += "A high positive z-score means unusually high correlation; a negative z-score means "
        section += "unusually low or negative correlation.\n\n"
        
        section += "**Comparison Assets:** SPY, QQQ, TLT, GLD, sector ETFs (XLF, XLE, XLK), "
        section += "and major S&P 100 stocks.\n\n"
        
        if not anomalies:
            section += "*No correlation alerts recorded.*\n\n"
            section += "No alerts were recorded. Missing or skipped comparisons do not establish normal correlations; consult the scan coverage.\n\n"
            section += "---\n\n"
            return section

        # Get top 10 anomalies
        radar = CorrelationRadar()
        top_anomalies = radar.get_top_anomalies(anomalies, n=10)

        section += f"### Top {len(top_anomalies)} Anomalies\n\n"
        section += f"**Selection:** Showing top 10 by |z-score| out of **{len(anomalies)} total** anomalies detected.\n\n"
        section += "Scores rank departures from the baseline; they do not identify a cause or a market regime.\n\n"
        section += "| Date | Pair | Correlation | Z-Score | Interpretation |\n"
        section += "|------|------|-------------|----------|----------------|\n"
        
        for anomaly in top_anomalies:
            if anomaly.z_score > 2:
                interp = "Unusually high correlation"
            elif anomaly.z_score < -2:
                interp = "Unusually low correlation"
            else:
                interp = "Moderate deviation"
            section += f"| {anomaly.date.strftime('%Y-%m-%d')} | {anomaly.pair[0]}-{anomaly.pair[1]} | {anomaly.correlation:.3f} | {anomaly.z_score:.2f} | {interp} |\n"
        
        section += "\n"
        section += "> *Z-score threshold: ±2.0 (a descriptive threshold, not a calibrated confidence interval). "
        section += "Rolling window: 30 trading days.*\n\n"
        section += "---\n\n"

        return section

    def _generate_event_section(
        self,
        event_impacts: List[EventImpact],
        assets_dir: Path
    ) -> str:
        """Generate event impact section."""
        section = "## Macro Event Impact Analysis\n\n"
        
        if not event_impacts:
            section += "*No event impact data available.*\n\n"
            section += "No verified event impacts are available for this range. The bundled calendar covers a partial October–December 2024 sample; missing calendar or price coverage is not evidence of no events.\n\n"
            section += "---\n\n"
            return section
        
        # Create summary table
        study = EventImpactStudy()
        summary_df = study.summarize_impacts(event_impacts)

        section += "### Event Impact Summary\n\n"
        section += "Returns are decimal log returns. Daily results are close-to-close proxies, not causal or market-adjusted effects.\n\n"
        section += "| Ticker | Event | Date | Post-Return | Cumulative | Data |\n"
        section += "|--------|-------|------|-------------|------------|------|\n"
        
        for _, row in summary_df.iterrows():
            data_type = row.get('data_type', 'daily') if 'data_type' in row else 'daily'
            section += f"| {row['ticker']} | {row['event_name']} | {row['event_date']} "
            section += f"| {row['post_return']:.4f} | {row['cumulative_return']:.4f} | {data_type} |\n"
        
        section += "\n"

        # Plot if we have data
        if len(event_impacts) > 0:
            try:
                chart_path = assets_dir / "event_impact.png"
                plot_event_impact(event_impacts, str(chart_path))
                section += f"![Event Impact](assets/{assets_dir.name}/event_impact.png)\n\n"
            except Exception:
                pass

        section += "---\n\n"

        return section

    def _generate_sabr_section(
        self,
        sabr_result: Optional[SABRFitResult],
        sabr_strikes: Optional[np.ndarray],
        sabr_ivs: Optional[np.ndarray],
        sabr_forward: Optional[float],
        sabr_time_to_expiry: Optional[float],
        assets_dir: Path
    ) -> str:
        """Generate SABR fit section."""
        section = "## Options Volatility Smile (SABR)\n\n"
        
        if sabr_result is None or sabr_strikes is None or sabr_ivs is None:
            section += "*No SABR analysis available.*\n\n"
            section += "Consult SABR status above: the analysis may be disabled, unavailable or failed.\n\n"
            section += "---\n\n"
            return section

        section += "> **What is SABR?** The SABR (Stochastic Alpha Beta Rho) model is a stochastic "
        section += "volatility model used to price options. It captures the volatility smile observed "
        section += "in real option markets - the phenomenon where implied volatility varies across strikes.\n\n"
        
        section += "### SABR Parameters\n\n"
        section += "| Parameter | Value | Description |\n"
        section += "|-----------|-------|-------------|\n"
        section += f"| α (alpha) | {sabr_result.params.alpha:.4f} | Volatility scale; ATM IV also depends on forward, beta and the time correction |\n"
        section += f"| β (beta) | {sabr_result.params.beta:.2f} | CEV exponent (0-1), fixed at 0.5 for this calibration |\n"
        section += f"| ρ (rho) | {sabr_result.params.rho:.2f} | Correlation of model forward and volatility shocks; affects skew |\n"
        section += f"| ν (nu) | {sabr_result.params.nu:.4f} | Volatility of volatility; affects smile curvature |\n\n"
        
        section += f"**Fit Error (RMSE):** {sabr_result.fit_error:.6f} "
        section += "*(lower is better - measures how well the model fits observed IV)*\n\n"

        # Plot if we have forward and time
        if sabr_forward is not None and sabr_time_to_expiry is not None:
            try:
                chart_path = assets_dir / "sabr_fit.png"
                plot_sabr_fit(
                    sabr_strikes, sabr_ivs, sabr_result,
                    sabr_forward, sabr_time_to_expiry, str(chart_path)
                )
                section += f"![SABR Fit](assets/{assets_dir.name}/sabr_fit.png)\n\n"
            except Exception:
                pass

        section += "---\n\n"

        return section

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return """
## Methodology Notes

### Factor Analysis
- Uses OLS regression of stock excess returns against Fama-French research factors
- Factors: Market (Mkt-RF), Value (HML), Size (SMB), Momentum (MOM)
- Data Source: Ken French Data Library
- R² describes the fitted sample; it does not measure forecast accuracy

### Correlation Anomalies
- Rolling window: 30 trading days
- Z-score threshold: ±2.0 (descriptive alert threshold; not a calibrated confidence interval)
- Comparison universe: 40+ assets including ETFs, sector funds, and S&P 100 stocks
- Anomalies indicate unusual co-movement, not necessarily causation

### Macro Event Impact
- Events within 7 days: Uses 1-minute intraday data (when available)
- Older events: Uses daily close-to-close returns
- Event times are approximate based on scheduled release times
- Actual market reaction may differ from scheduled time

### SABR Model
- Calibrated to nearest available option expiry
- β (beta) fixed at 0.5 as a modeling choice
- Parameters are point-in-time; the volatility surface changes daily
- Review residuals and quote quality; no universal RMSE cutoff or arbitrage check is applied

### Date Considerations
- All dates in US/Eastern timezone
- Trading days only (excludes weekends/holidays)
- At least 30 valid aligned observations are required for factors; sample length alone does not establish stability

---

*Report generated by Equity Factor Analysis*
"""
