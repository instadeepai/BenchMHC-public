"""Module used to define useful constants for the project."""

import os
from pathlib import Path
from typing import TypeVar

import polars as pl

LazyOrDataFrame = TypeVar("LazyOrDataFrame", pl.LazyFrame, pl.DataFrame)

CURRENT_DIRECTORY = Path(__file__).resolve().parent
ROOT_DIRECTORY = CURRENT_DIRECTORY.parent


def _directory_from_environment(variable: str, default: Path) -> Path:
    """Read a directory from an environment variable, as an absolute path.

    A relative value is resolved against the working directory once, at import, so the
    directory does not move with the working directory of each process, and so that
    paths built under it stay absolute: `BaseLightningModule` treats a relative model
    path as relative to `MODEL_DIRECTORY`, so a relative `MODEL_DIRECTORY` would nest
    training artifacts under itself twice.

    Args:
        variable: Name of the environment variable to read.
        default: Directory to use when the variable is unset or empty.

    Returns:
        The absolute directory, with `~` expanded.
    """
    return Path(os.environ.get(variable) or default).expanduser().resolve()


# The two directories the package writes to. They default inside the repository, which is
# where both documented installations run from, and are overridable because an installed
# package puts ROOT_DIRECTORY in site-packages, where a process that is not root cannot
# write.
CACHE_DIRECTORY = _directory_from_environment(
    "BENCH_MHC_CACHE_DIRECTORY", ROOT_DIRECTORY / ".cache"
)

MODEL_DIRECTORY = _directory_from_environment(
    "BENCH_MHC_MODEL_DIRECTORY", ROOT_DIRECTORY / "models"
)

NATURAL_AAS = "ACDEFGHIKLMNPQRSTVWY"

SEPARATOR = "__"

# Path to the allele to pseudo sequence mapping file.
ALLELE_MAPPING_PATH = ROOT_DIRECTORY / "data" / "mappings" / "allele2netmhc_pseudo_seq.json"

# Number of steps in the percentile rank calculation.
NUM_STEPS_PCTRNK = 10_000

FLANK_SIZE = 3

PAD_TOKEN = "-"
