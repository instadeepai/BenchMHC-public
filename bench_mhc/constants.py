"""Module used to define useful constants for the project."""

import os
from pathlib import Path
from typing import TypeVar

import polars as pl

LazyOrDataFrame = TypeVar("LazyOrDataFrame", pl.LazyFrame, pl.DataFrame)

CURRENT_DIRECTORY = Path(__file__).resolve().parent
ROOT_DIRECTORY = CURRENT_DIRECTORY.parent

# The two directories the package writes to. They default inside the repository, which is
# where both documented installations run from, and are overridable because an installed
# package puts ROOT_DIRECTORY in site-packages, where a process that is not root cannot
# write.
CACHE_DIRECTORY = Path(os.environ.get("BENCH_MHC_CACHE_DIRECTORY") or ROOT_DIRECTORY / ".cache")

MODEL_DIRECTORY = Path(os.environ.get("BENCH_MHC_MODEL_DIRECTORY") or ROOT_DIRECTORY / "models")

NATURAL_AAS = "ACDEFGHIKLMNPQRSTVWY"

SEPARATOR = "__"

# Path to the allele to pseudo sequence mapping file.
ALLELE_MAPPING_PATH = ROOT_DIRECTORY / "data" / "mappings" / "allele2netmhc_pseudo_seq.json"

# Number of steps in the percentile rank calculation.
NUM_STEPS_PCTRNK = 10_000

FLANK_SIZE = 3

PAD_TOKEN = "-"
