"""
Caching layer for downloaded data.

This module provides a simple disk-based cache that stores raw downloads
by query hash to avoid repeated network calls.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional, Any
import pandas as pd
from backend.errors import CacheError


class DataCache:
    """
    A disk-based cache for storing downloaded data.

    Caches data by computing a hash of the query parameters and storing
    the result on disk. Supports both pandas DataFrames/Series and
    arbitrary Python objects (via pickle).

    Representation Invariants:
        - cache_dir exists and is a directory
        - cache files are named by their hash
    """

    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize the cache.

        Preconditions:
            - cache_dir is a valid path (will be created if it doesn't exist)

        Postconditions:
            - cache_dir exists as a directory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_hash(self, query_params: dict) -> str:
        """
        Compute a hash for query parameters.

        Args:
            query_params: Dictionary of query parameters

        Returns:
            Hex string hash
        """
        # Sort keys for consistent hashing
        sorted_params = json.dumps(query_params, sort_keys=True)
        return hashlib.md5(sorted_params.encode()).hexdigest()

    def get(self, query_params: dict) -> Optional[Any]:
        """
        Retrieve cached data if it exists.

        Preconditions:
            - query_params is a dictionary

        Postconditions:
            - Returns cached data if found, None otherwise
            - Does not modify cache

        Args:
            query_params: Query parameters used to generate cache key

        Returns:
            Cached data (DataFrame, Series, or other) or None if not found
        """
        cache_key = self._compute_hash(query_params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            # If cache file is corrupted, return None
            raise CacheError(f"Failed to read cache file: {e}") from e

    def set(self, query_params: dict, data: Any) -> None:
        """
        Store data in cache.

        Preconditions:
            - query_params is a dictionary
            - data is serializable (pandas objects or pickle-able)

        Postconditions:
            - Data is stored in cache file
            - Cache file is named by query hash

        Args:
            query_params: Query parameters used to generate cache key
            data: Data to cache (DataFrame, Series, or other)
        """
        cache_key = self._compute_hash(query_params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            raise CacheError(f"Failed to write cache file: {e}") from e

    def clear(self) -> None:
        """
        Clear all cached files.

        Postconditions:
            - All .pkl files in cache_dir are removed
        """
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

    def exists(self, query_params: dict) -> bool:
        """
        Check if cached data exists for query.

        Args:
            query_params: Query parameters

        Returns:
            True if cache exists, False otherwise
        """
        cache_key = self._compute_hash(query_params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        return cache_file.exists()


