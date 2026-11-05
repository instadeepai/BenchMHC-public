"""Module used to define useful constants for the project."""

from pathlib import Path
from typing import TypeVar

import polars as pl

LazyOrDataFrame = TypeVar("LazyOrDataFrame", pl.LazyFrame, pl.DataFrame)

CURRENT_DIRECTORY = Path(__file__).resolve().parent
ROOT_DIRECTORY = CURRENT_DIRECTORY.parent

CACHE_DIRECTORY = ROOT_DIRECTORY / ".cache"

MODEL_DIRECTORY = ROOT_DIRECTORY / "models"

NATURAL_AAS = "ACDEFGHIKLMNPQRSTVWY"

SEPARATOR = "__"

# Path to the allele to pseudo sequence mapping file. It lives inside the package, so that
# it is part of the distribution and resolves wherever bench_mhc is installed from, rather
# than only when the working directory is the repository root.
ALLELE_MAPPING_PATH = CURRENT_DIRECTORY / "data" / "mappings" / "allele2netmhc_pseudo_seq.json"

# Number of steps in the percentile rank calculation.
NUM_STEPS_PCTRNK = 10_000

FLANK_SIZE = 3

PAD_TOKEN = "-"
