"""
FastAPI web interface for viewing reports.

This module provides a web interface to view generated reports and run analyses.
Implements OWASP security guidelines including input validation and sanitization.
"""

import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, validator
import re
import html
from pathlib import Path
import json
import subprocess
from typing import List, Optional, Dict
from datetime import datetime

app = FastAPI(title="Equity Factor Analysis")

# Security: CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Restrict to localhost
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Setup templates
template_env = Environment(loader=FileSystemLoader("frontend/templates"))

# Mount reports directory
reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)
assets_dir = reports_dir / "assets"
assets_dir.mkdir(exist_ok=True)

app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


# Input validation models
class AnalysisRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    fit_sabr: bool = False

    @validator('tickers')
    def validate_tickers(cls, v):
        if len(v) > 50:  # Limit max tickers per request (DoS protection)
            raise ValueError("Too many tickers (max 50)")
        validated = []
        for ticker in v:
            # Clean and validate ticker
            ticker = ticker.strip().upper()
            # Allow alphanumeric, dots (BRK.A), and hyphens (BF-B)
            if not ticker or len(ticker) > 10:
                raise ValueError(f"Invalid ticker: {ticker}")
            # Remove invalid characters but allow . and -
            import re
            if not re.match(r'^[A-Z0-9.\-]+$', ticker):
                raise ValueError(f"Invalid ticker format: {ticker}")
            validated.append(ticker)
        return validated


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    # Remove any path components
    filename = Path(filename).name
    # Only allow alphanumeric, dash, underscore, dot
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    return filename


def parse_report_metadata(report_path: Path) -> dict:
    """Parse report metadata from filename and content."""
    stem = report_path.stem
    parts = stem.split("_")
    
    metadata = {
        "filename": report_path.name,
        "ticker": parts[0] if parts else "Unknown",
        "date": report_path.stat().st_mtime,
        "date_str": datetime.fromtimestamp(report_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "sections": []
    }
    
    return metadata


def parse_markdown_report(content: str) -> Dict:
    """Parse markdown report and extract structured data."""
    result = {
        "r_squared": 0.0,
        "factors": [],
        "anomalies": 0,
        "anomalies_data": [],
        "events": 0,
        "events_data": [],
        "sabr": None
    }
    
    # Parse R²
    r2_match = re.search(r'\*\*R²:\*\*\s*([\d.]+)', content)
    if r2_match:
        result["r_squared"] = float(r2_match.group(1))
    
    # Parse factor table - look for rows with Factor | Beta | t-stat | p-value format
    # Match both old format (t-stat) and new format (t-statistic)
    factor_rows = re.findall(
        r'\|\s*(\w+)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|',
        content
    )
    for row in factor_rows:
        factor_name = row[0]
        # Skip header rows and separator rows
        if factor_name.lower() in ['factor', '---', '-----', ':---']:
            continue
        try:
            result["factors"].append({
                "Factor": factor_name,
                "Beta": round(float(row[1]), 4),
                "t-stat": round(float(row[2]), 2),
                "p-value": round(float(row[3]), 4)
            })
        except (ValueError, IndexError):
            continue
    
    # Count anomalies - look for anomaly table rows
    anomaly_rows = re.findall(
        r'\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\w+-\w+)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|',
        content
    )
    # Parse total anomalies from text
    # Matches "**841 total** anomalies detected"
    total_anomalies_match = re.search(r'\*\*(\d+)\s+total\*\*\s+anomalies', content)
    if total_anomalies_match:
        result["anomalies"] = int(total_anomalies_match.group(1))
    else:
        # Fallback to row count if text match fails
        result["anomalies"] = len(anomaly_rows)

    result["anomalies_data"] = [
        {"Date": row[0], "Pair": row[1], "Correlation": row[2], "Z-Score": row[3]}
        for row in anomaly_rows[:10]
    ]
    
    # Count events - look for event table rows (new format with data type)
    event_rows = re.findall(
        r'\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*(\w+)\s*\|',
        content
    )
    result["events"] = len(event_rows)
    result["events_data"] = [
        {
            "Ticker": row[0],
            "Event": row[1],
            "Date": row[2],
            "Post-Return": row[3],
            "Cumulative": row[4],
            "Data": row[5]
        }
        for row in event_rows
    ]
    
    # Parse SABR - look for alpha, beta, rho, nu values in various formats
    # Format 1: Bullet points (old format)
    sabr_alpha = re.search(r'\*\*α \(alpha\):\*\*\s*([\d.]+)', content)
    sabr_beta = re.search(r'\*\*β \(beta\):\*\*\s*([\d.]+)', content)
    sabr_rho = re.search(r'\*\*ρ \(rho\):\*\*\s*([-\d.]+)', content)
    sabr_nu = re.search(r'\*\*ν \(nu\):\*\*\s*([\d.]+)', content)
    
    # Format 2: Table format (new format)
    if not sabr_alpha:
        sabr_alpha = re.search(r'\| α \(alpha\)\s*\|\s*([\d.]+)', content)
    if not sabr_beta:
        sabr_beta = re.search(r'\| β \(beta\)\s*\|\s*([\d.]+)', content)
    if not sabr_rho:
        sabr_rho = re.search(r'\| ρ \(rho\)\s*\|\s*([-\d.]+)', content)
    if not sabr_nu:
        sabr_nu = re.search(r'\| ν \(nu\)\s*\|\s*([\d.]+)', content)
    
    if sabr_alpha and sabr_beta and sabr_rho and sabr_nu:
        result["sabr"] = {
            "alpha": float(sabr_alpha.group(1)),
            "beta": float(sabr_beta.group(1)),
            "rho": float(sabr_rho.group(1)),
            "nu": float(sabr_nu.group(1))
        }
    
    return result


def markdown_to_html(content: str) -> str:
    """Convert markdown to HTML with dark tech styling and XSS protection."""
    # Escape HTML first to prevent XSS
    content = html.escape(content)
    
    # Headers
    content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
    content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
    content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
    
    # Bold (restore after escaping)
    content = content.replace('&lt;strong&gt;', '<strong>').replace('&lt;/strong&gt;', '</strong>')
    content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
    
    # Tables with dark tech styling
    lines = content.split("\n")
    html_lines = []
    in_table = False
    is_header = False
    
    for line in lines:
        if "|" in line and not line.strip().startswith("<"):
            if not in_table:
                html_lines.append('<div class="table-responsive my-4"><table>')
                in_table = True
                is_header = True
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if cells and not all(c.startswith("-") or c.startswith(":") for c in cells):
                if is_header:
                    html_lines.append("<thead><tr>")
                    for cell in cells:
                        html_lines.append(f"<th>{cell}</th>")
                    html_lines.append("</tr></thead><tbody>")
                    is_header = False
                else:
                    html_lines.append("<tr>")
                    for cell in cells:
                        # Format numbers
                        if re.match(r'^-?\d+\.?\d*$', cell):
                            cell = f"{float(cell):.4f}" if '.' in cell else cell
                        html_lines.append(f"<td>{cell}</td>")
                    html_lines.append("</tr>")
        else:
            if in_table:
                html_lines.append("</tbody></table></div>")
                in_table = False
                is_header = False
            html_lines.append(line)
    
    if in_table:
        html_lines.append("</tbody></table></div>")
    
    content = "\n".join(html_lines)
    
    # Lists
    content = re.sub(r'^- (.+)$', r'<li>\1</li>', content, flags=re.MULTILINE)
    # Wrap consecutive list items
    content = re.sub(r'(<li>.*?</li>(?:\s*<li>.*?</li>)*)', r'<ul>\1</ul>', content, flags=re.DOTALL)
    
    # Images (with strict protocol validation)
    def image_replacer(match):
        alt = match.group(1)
        src = match.group(2)
        # Prevent javascript: or data: XSS vectors
        if re.search(r'^\s*(javascript:|data:|vbscript:)', src, re.IGNORECASE):
            return "" # Block common XSS vectors
        return f'<img src="{src}" alt="{alt}">'
    
    content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', image_replacer, content)
    
    # Line breaks
    content = content.replace("\n\n", "<br><br>")
    
    return content


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with React UI."""
    react_template = Path("frontend/templates/react_index.html")
    if react_template.exists():
        content = react_template.read_text()
        return HTMLResponse(content=content)
    
    # Fallback to old template
    reports = sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    report_list = [parse_report_metadata(r) for r in reports[:20]]
    template = template_env.get_template("index.html")
    return HTMLResponse(template.render(
        reports=report_list,
        report_count=len(reports),
        today=datetime.now().strftime("%Y-%m-%d")
    ))


@app.get("/report/{filename}", response_class=HTMLResponse)
async def view_report(request: Request, filename: str):
    """View a specific report with security validation."""
    # Sanitize filename to prevent path traversal
    filename = sanitize_filename(filename)
    report_path = reports_dir / filename
    
    if not report_path.exists() or not filename.endswith('.md'):
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Read and convert markdown
    content = report_path.read_text()
    html_content = markdown_to_html(content)
    
    metadata = parse_report_metadata(report_path)
    
    template = template_env.get_template("report.html")
    return HTMLResponse(template.render(
        filename=html.escape(filename),
        ticker=html.escape(metadata["ticker"]),
        date_str=html.escape(metadata["date_str"]),
        content=html_content
    ))


@app.post("/api/run-analysis")
async def run_analysis(request: AnalysisRequest):
    """Trigger a new analysis with input validation."""
    try:
        # Additional validation
        start_date_obj = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(request.end_date, "%Y-%m-%d")
        today = datetime.now().date()
        min_date = datetime(2000, 1, 1).date()
        
        # Validate date range
        if start_date_obj.date() < min_date:
            raise HTTPException(status_code=400, detail=f"Start date cannot be before {min_date}")
        
        if end_date_obj.date() > today:
            raise HTTPException(status_code=400, detail="End date cannot be in the future")
        
        if start_date_obj > end_date_obj:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        if (end_date_obj - start_date_obj).days > 365 * 5:  # Max 5 years
            raise HTTPException(status_code=400, detail="Date range cannot exceed 5 years")
        
        # Run analysis for each ticker
        for ticker in request.tickers:
            cmd = [
                sys.executable, "-W", "ignore", "-m", "backend.cli", "analyze",
                ticker,
                "--start", request.start_date,
                "--end", request.end_date
            ]
            if request.fit_sabr:
                cmd.append("--fit-sabr")
            
            subprocess.Popen(
                cmd,
                cwd=Path(__file__).parent.parent,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        return {
            "status": "started",
            "message": f"Analysis running for {', '.join(request.tickers)}",
            "tickers": request.tickers
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/analysis-results/{ticker}")
async def get_analysis_results(ticker: str):
    """Get parsed analysis results for a ticker."""
    # Find latest report for this ticker
    reports = sorted(
        [r for r in reports_dir.glob("*.md") if r.stem.startswith(ticker)],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if not reports:
        raise HTTPException(status_code=404, detail="No reports found for this ticker")
    
    # Parse report content
    report_path = reports[0]
    content = report_path.read_text()
    parsed_data = parse_markdown_report(content)
    
    metadata = parse_report_metadata(report_path)
    
    return {
        "ticker": ticker,
        "report_path": report_path.name,
        "r_squared": parsed_data["r_squared"],
        "factors": parsed_data["factors"],
        "anomalies": parsed_data["anomalies"],
        "anomalies_data": parsed_data["anomalies_data"],
        "events": parsed_data["events"],
        "events_data": parsed_data["events_data"],
        "sabr": parsed_data["sabr"],
        "date_str": metadata["date_str"]
    }


@app.get("/api/reports")
async def list_reports():
    """API endpoint to list all reports."""
    reports = sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "reports": [parse_report_metadata(r) for r in reports]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
