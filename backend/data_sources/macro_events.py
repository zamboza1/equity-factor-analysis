"""
Macro event data sources.

This module handles loading macro event timestamps (CPI releases, FOMC meetings, etc.)
from CSV files.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from pathlib import Path
from backend.errors import DataError


@dataclass
class MacroEvent:
    """
    Represents a macroeconomic event.

    Attributes:
        name: Event name (e.g., "CPI", "FOMC")
        timestamp: Event timestamp (pd.Timestamp)
        timezone: Timezone string (e.g., "US/Eastern")
        source: Data source (e.g., "BLS", "Fed")
    """
    name: str
    timestamp: pd.Timestamp
    timezone: str
    source: str

    def __repr__(self) -> str:
        """String representation."""
        return f"MacroEvent({self.name}, {self.timestamp}, {self.source})"


def load_events_from_csv(csv_path: Optional[str] = None) -> List[MacroEvent]:
    """
    Load macro events from CSV file.

    Expected CSV columns:
        - event_name: Name of event
        - event_date: Date string (YYYY-MM-DD)
        - event_time_et: Time string (HH:MM) in Eastern Time
        - timezone: Timezone string
        - source: Data source

    Preconditions:
        - csv_path points to valid CSV file (if provided)
        - CSV has required columns

    Postconditions:
        - Returns list of MacroEvent objects
        - Events are sorted by timestamp

    Args:
        csv_path: Path to CSV file. If None, uses default.

    Returns:
        List of MacroEvent objects

    Raises:
        DataError: If file doesn't exist or is invalid
    """
    if csv_path is None:
        default_path = Path(__file__).parent.parent.parent / "data" / "macro_events.csv"
        csv_path = str(default_path)

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise DataError(f"Macro events CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_file)

        # Validate required columns
        required_cols = ["event_name", "event_date", "event_time_et", "timezone", "source"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataError(f"Missing required columns: {missing_cols}")

        events = []
        for _, row in df.iterrows():
            # Parse date and time
            date_str = str(row["event_date"])
            time_str = str(row["event_time_et"])

            # Combine date and time
            datetime_str = f"{date_str} {time_str}"
            timestamp = pd.to_datetime(datetime_str)

            # Convert to specified timezone if needed
            timezone = str(row["timezone"])
            if timezone:
                timestamp = timestamp.tz_localize(timezone)

            event = MacroEvent(
                name=str(row["event_name"]),
                timestamp=timestamp,
                timezone=timezone,
                source=str(row["source"])
            )
            events.append(event)

        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp)

        return events

    except Exception as e:
        raise DataError(f"Failed to load macro events: {e}") from e


def filter_events_by_name(events: List[MacroEvent], event_name: str) -> List[MacroEvent]:
    """
    Filter events by name.

    Args:
        events: List of MacroEvent objects
        event_name: Event name to filter by

    Returns:
        Filtered list of events
    """
    return [e for e in events if e.name == event_name]


def filter_events_by_date_range(
    events: List[MacroEvent],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> List[MacroEvent]:
    """
    Filter events by date range.

    Args:
        events: List of MacroEvent objects
        start_date: Start timestamp (inclusive)
        end_date: End timestamp (inclusive)

    Returns:
        Filtered list of events
    """
    # Normalize timezones for comparison
    filtered = []
    for e in events:
        event_ts = e.timestamp
        # Convert to same timezone if needed
        if event_ts.tz is not None and start_date.tz is None:
            start_ts = start_date.tz_localize(event_ts.tz)
            end_ts = end_date.tz_localize(event_ts.tz)
        elif event_ts.tz is None and start_date.tz is not None:
            event_ts = event_ts.tz_localize(start_date.tz)
            start_ts = start_date
            end_ts = end_date
        else:
            start_ts = start_date
            end_ts = end_date

        if start_ts <= event_ts <= end_ts:
            filtered.append(e)

    return filtered


def generate_recent_events(days_back: int = 30) -> List[MacroEvent]:
    """
    Automatically generate recent macro events based on known schedules.
    
    CPI: 2nd Tuesday of each month at 8:30 AM ET
    FOMC: Scheduled meetings (approximate - 8 per year)
    NFP: First Friday of each month at 8:30 AM ET
    
    This ensures events are always within yfinance's 30-day intraday data window.
    
    Args:
        days_back: How many days back to generate events (default 30)
    
    Returns:
        List of MacroEvent objects for recent events
    """
    from datetime import datetime, timedelta
    import pytz
    
    events = []
    eastern = pytz.timezone("US/Eastern")
    # Use pd.Timestamp for consistent timezone handling
    today_ts = pd.Timestamp.now(tz=eastern)
    today = today_ts.to_pydatetime()
    start_date = (today_ts - pd.Timedelta(days=days_back)).to_pydatetime()
    
    # Generate events for all months that overlap with the date range
    # Start from 2 months before start_date to catch events near the boundary
    current_month = (start_date - timedelta(days=60)).replace(day=1)
    end_month = today.replace(day=1) + timedelta(days=32)  # Next month to be safe
    
    # Generate CPI events (2nd Tuesday of each month)
    while current_month <= end_month:
        # Find 2nd Tuesday
        first_day = current_month.replace(day=1)
        # Find first Tuesday
        days_until_tuesday = (1 - first_day.weekday()) % 7  # 1 = Tuesday
        if days_until_tuesday == 0:
            days_until_tuesday = 7
        first_tuesday = first_day + timedelta(days=days_until_tuesday)
        second_tuesday = first_tuesday + timedelta(days=7)  # 2nd Tuesday
        
        # Convert to timezone-aware for comparison
        second_tuesday_aware = eastern.localize(second_tuesday.replace(tzinfo=None)) if second_tuesday.tzinfo is None else second_tuesday
        
        # Create event timestamp
        event_time = eastern.localize(
            datetime.combine(second_tuesday_aware.date(), datetime.strptime("08:30", "%H:%M").time())
        )
        event_ts = pd.Timestamp(event_time)
        
        # Check if within date range
        if start_date <= event_ts.to_pydatetime() <= today:
            events.append(MacroEvent(
                name="CPI",
                timestamp=event_ts,
                timezone="US/Eastern",
                source="BLS"
            ))
        
        # Move to next month
        if current_month.month == 12:
            current_month = current_month.replace(year=current_month.year + 1, month=1)
        else:
            current_month = current_month.replace(month=current_month.month + 1)
    
    # Generate NFP events (first Friday of each month)
    current_month = (start_date - timedelta(days=60)).replace(day=1)
    while current_month <= end_month:
        first_day = current_month.replace(day=1)
        days_until_friday = (4 - first_day.weekday()) % 7  # 4 = Friday
        if days_until_friday == 0:
            days_until_friday = 7
        first_friday = first_day + timedelta(days=days_until_friday)
        
        # Convert to timezone-aware
        first_friday_aware = eastern.localize(first_friday.replace(tzinfo=None)) if first_friday.tzinfo is None else first_friday
        
        # Create event timestamp
        event_time = eastern.localize(
            datetime.combine(first_friday_aware.date(), datetime.strptime("08:30", "%H:%M").time())
        )
        event_ts = pd.Timestamp(event_time)
        
        # Check if within date range
        if start_date <= event_ts.to_pydatetime() <= today:
            events.append(MacroEvent(
                name="NFP",
                timestamp=event_ts,
                timezone="US/Eastern",
                source="BLS"
            ))
        
        # Move to next month
        if current_month.month == 12:
            current_month = current_month.replace(year=current_month.year + 1, month=1)
        else:
            current_month = current_month.replace(month=current_month.month + 1)
    
    # FOMC dates are irregular and cannot be algorithmically generated without valid source data.
    # Users should ensure data/macro_events.csv is up to date for FOMC meetings.

    
    # Sort by timestamp and remove duplicates
    events.sort(key=lambda x: x.timestamp)
    seen = set()
    unique_events = []
    for event in events:
        key = (event.name, event.timestamp.date())
        if key not in seen:
            seen.add(key)
            unique_events.append(event)
    
    return unique_events


def get_recent_events_for_analysis(days_back: int = 30) -> List[MacroEvent]:
    """
    Get recent events for analysis, prioritizing auto-generated events
    within the data availability window.
    
    Args:
        days_back: Days back to look for events (default 30 for yfinance limit)
    
    Returns:
        List of recent MacroEvent objects
    """
    # Try to load from CSV first
    try:
        csv_events = load_events_from_csv()
        # Filter to recent events only
        today = pd.Timestamp.now(tz="US/Eastern")
        start_date = today - pd.Timedelta(days=days_back)
        recent_csv = filter_events_by_date_range(csv_events, start_date, today)
    except Exception:
        recent_csv = []
    
    # Generate recent events automatically
    auto_events = generate_recent_events(days_back=days_back)
    
    # Combine and deduplicate (prefer auto-generated for recent dates)
    all_events = {}
    for event in recent_csv + auto_events:
        key = (event.name, event.timestamp.date())
        # Keep auto-generated if duplicate (more accurate)
        if key not in all_events or event.source in ["BLS", "Fed"]:
            all_events[key] = event
    
    return sorted(all_events.values(), key=lambda x: x.timestamp)

