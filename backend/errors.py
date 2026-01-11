"""Custom exceptions for the radar module."""


class RadarError(Exception):
    """Base exception for radar module errors."""
    pass


class DataError(RadarError):
    """Raised when data is missing, invalid, or insufficient."""
    pass


class CacheError(RadarError):
    """Raised when caching operations fail."""
    pass


class CalibrationError(RadarError):
    """Raised when model calibration fails (e.g., SABR fit)."""
    pass


