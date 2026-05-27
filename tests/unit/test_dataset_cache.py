"""Unit tests related to bench_mhc/dataset/cache.py."""

from pathlib import Path

import polars as pl
import pytest

from bench_mhc.constants import LazyOrDataFrame
from bench_mhc.dataset.cache import DatasetCache
from bench_mhc.utils.mode import Mode
from bench_mhc.variables.variables import Outputs
from bench_mhc.variables.variables import Variables


@pytest.fixture
def sample_dataframe(request: pytest.FixtureRequest) -> pl.DataFrame | pl.LazyFrame:
    """Get a sample dataframe for testing with peptide and allele features."""
    df = pl.DataFrame(
        {
            "peptide": ["SIINFEKL", "NLVPMVATV", "GILGFVFTL"],
            "allele": ["A0201", "A0101", "B0702"],
            "hit": [1, 0, 1],
        }
    )

    param = getattr(request, "param", "eager")
    if param == "lazy":
        return df.lazy()

    return df


@pytest.fixture
def sample_variables() -> tuple[Variables, Outputs]:
    """Get sample input and output variables."""
    inputs = Variables.from_dict(
        {
            "peptide": {
                "class_name": "NNAlignVariable",
                "unk_token": "X",
            },
            "allele": {
                "class_name": "AASeqVariable",
                "unk_token": "X",
            },
        }
    )
    outputs = Outputs.from_dict(
        {
            "hit": {
                "class_name": "BinaryOutput",
            }
        }
    )

    return inputs, outputs


def test_init(cache_directory: Path) -> None:
    """Test DatasetCache initialization."""
    cache = DatasetCache(cache_directory / "cache")
    assert cache.cache_directory == cache_directory / "cache"
    assert cache.cache_directory.exists()


def test_compute_hash_consistency(
    cache_directory: Path,
    sample_dataframe: pl.DataFrame,
    sample_variables: tuple[Variables, Outputs],
) -> None:
    """Test that compute_hash produces consistent results for same inputs."""
    cache = DatasetCache(cache_directory)
    inputs, outputs = sample_variables

    # Compute hash twice with same inputs
    metadata = {
        "inputs": inputs.to_dict(),
        "outputs": outputs.to_dict(),
        "mode": Mode.TRAIN.value,
    }
    hash1 = cache.compute_hash(sample_dataframe, metadata)
    hash2 = cache.compute_hash(sample_dataframe, metadata)

    assert hash1 == hash2
    assert len(hash1) == 32  # MD5 hash length


def test_compute_hash_different_modes(
    cache_directory: Path,
    sample_dataframe: pl.DataFrame,
    sample_variables: tuple[Variables, Outputs],
) -> None:
    """Test that compute_hash produces different results for different modes."""
    cache = DatasetCache(cache_directory)
    inputs, outputs = sample_variables

    metadata = {
        "inputs": inputs.to_dict(),
        "outputs": outputs.to_dict(),
        "mode": Mode.TRAIN.value,
    }
    train_hash = cache.compute_hash(sample_dataframe, metadata)
    metadata["mode"] = Mode.VAL.value
    val_hash = cache.compute_hash(sample_dataframe, metadata)

    assert train_hash != val_hash


@pytest.mark.parametrize("mode", [Mode.TRAIN, Mode.VAL, Mode.TEST])
@pytest.mark.parametrize("sample_dataframe", ["lazy", "eager"], indirect=True)
def test_cache_and_load(
    mode: Mode,
    sample_dataframe: LazyOrDataFrame,
    cache_directory: Path,
    sample_variables: tuple[Variables, Outputs],
) -> None:
    """Test caching and loading a dataset."""
    cache = DatasetCache(cache_directory)
    inputs, outputs = sample_variables

    # Compute hash and cache the dataframe
    metadata = {
        "inputs": inputs.to_dict(),
        "outputs": outputs.to_dict(),
        "mode": mode.value,
    }
    hash_ = cache.compute_hash(sample_dataframe, metadata)
    cache.cache_from_hash(hash_, sample_dataframe)

    # Load the cached dataframe
    loaded_df = cache.load_from_hash(hash_)

    assert loaded_df is not None
    assert loaded_df.equals(
        sample_dataframe.collect()
        if isinstance(sample_dataframe, pl.LazyFrame)
        else sample_dataframe
    )

    Path(cache_directory / f"{hash_}.parquet").unlink()
    with pytest.raises(FileNotFoundError, match=f"Dataset with hash {hash_} not found in cache."):
        cache.load_from_hash(hash_)


def test_load_nonexistent_hash(cache_directory: Path) -> None:
    """Test loading a non-existent hash returns None."""
    cache = DatasetCache(cache_directory)
    with pytest.raises(
        FileNotFoundError, match="Dataset with hash nonexistent_hash not found in cache."
    ):
        cache.load_from_hash("nonexistent_hash")


def test_cache_different_dataframes(
    cache_directory: Path,
    sample_dataframe: pl.DataFrame,
    sample_variables: tuple[Variables, Outputs],
) -> None:
    """Test caching different dataframes produces different hashes."""
    cache = DatasetCache(cache_directory)
    inputs, outputs = sample_variables

    # Create a modified dataframe with a different peptide
    modified_df = sample_dataframe.with_columns(
        peptide=pl.Series(["BBBBBB", "AAAAAA", "GILGFVFTL"])
    )

    # Compute hashes for both dataframes
    metadata = {
        "inputs": inputs.to_dict(),
        "outputs": outputs.to_dict(),
        "mode": Mode.TRAIN.value,
    }

    original_hash = cache.compute_hash(sample_dataframe, metadata)
    modified_hash = cache.compute_hash(modified_df, metadata)

    assert original_hash != modified_hash
