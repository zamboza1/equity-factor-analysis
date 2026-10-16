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
        default_path = Path(__file__).parent.parent / "data" / "macro_events.csv"
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


def get_events_for_analysis(start_date, end_date) -> List[MacroEvent]:
    """Return sourced CSV events in the start-inclusive, end-exclusive range.

    The bundled calendar is a partial historical sample, not a live calendar.
    Missing calendar coverage must not be replaced with weekday guesses.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start.tz is None: start = start.tz_localize("US/Eastern")
    if end.tz is None: end = end.tz_localize("US/Eastern")
    if start >= end:
        raise DataError("Event range start must precede end")
    return [event for event in load_events_from_csv() if start <= event.timestamp < end]


def get_recent_events_for_analysis(days_back: int = 30) -> List[MacroEvent]:
    """Read recent sourced entries; no entries means unavailable calendar data."""
    now = pd.Timestamp.now(tz="US/Eastern")
    return get_events_for_analysis(now - pd.Timedelta(days=days_back), now)


def generate_recent_events(days_back: int = 30) -> List[MacroEvent]:
    """Compatibility wrapper; event timestamps are loaded, never invented."""
    return get_recent_events_for_analysis(days_back)
