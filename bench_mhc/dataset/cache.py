"""Module to implement dataset caching functionality."""

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl

from bench_mhc.constants import LazyOrDataFrame
from bench_mhc.utils.logging import system

log = system.get(__name__)


class DatasetCache:
    """Cache system for datasets.

    This class provides functionality to cache transformed datasets to avoid redundant
    transformations. The cache is keyed by a hash of the input parameters and data.

    Attributes:
        cache_directory: Directory where cached datasets are stored.
    """

    def __init__(self, cache_directory: Path) -> None:
        """Initialize the cache.

        Args:
            cache_directory: Directory where cached datasets will be stored.
        """
        self.cache_directory = cache_directory
        self.cache_directory.mkdir(parents=True, exist_ok=True)

    def compute_hash(self, df: LazyOrDataFrame, metadata: dict[str, Any]) -> str:
        """Compute a hash for the dataset based on its parameters and data.

        Args:
            df: The dataset to hash.
            metadata: Metadata about the model, the mode, etc.

        Returns:
            A MD5 hash string.
        """
        columns = sorted(
            df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
        )

        # Sort columns and rows to ensure consistent hash generation
        buffer = BytesIO()
        df.lazy().select(columns).sort(by=columns).collect(engine="streaming").write_csv(buffer)

        hasher = hashlib.md5()
        hasher.update(buffer.getvalue())
        hasher.update(json.dumps(metadata, sort_keys=True).encode())

        return hasher.hexdigest()

    def cache_from_hash(self, hash_: str, df: LazyOrDataFrame) -> None:
        """Cache a dataset using a pre-computed hash value.

        Args:
            hash_: The pre-computed hash string to use as the cache key.
            df: The dataset to cache, either as a Polars DataFrame or LazyFrame.
        """
        cache_path = self.cache_directory / f"{hash_}.parquet"

        if isinstance(df, pl.LazyFrame):
            df.sink_parquet(cache_path)
        else:
            df.write_parquet(cache_path)

        log.info(f"Dataset cached at {cache_path}")

    def load_from_hash(self, hash_: str) -> pl.DataFrame:
        """Load a dataset from a pre-computed hash value.

        Args:
            hash_: The pre-computed hash string to use as the cache key.

        Returns:
            The cached dataset if found.
        """
        cache_path = self.cache_directory / f"{hash_}.parquet"

        try:
            return pl.read_parquet(cache_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Dataset with hash {hash_} not found in cache.") from error
