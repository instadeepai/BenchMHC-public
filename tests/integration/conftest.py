"""Configuration file for shared fixtures used in integration tests."""

from pathlib import Path

import polars as pl
import pytest

from bench_mhc.utils.io import load_yml
from bench_mhc.utils.io import save_yml


@pytest.fixture
def configuration_file_path_hit_only(tmp_path: Path, configuration_file_path: str) -> str:
    """Get model configuration file path for the tests with only hit output."""
    file_path = tmp_path / "configuration_hit_only.yaml"

    configuration = load_yml(configuration_file_path)
    del configuration["variables"]["outputs"]["binding_affinity"]
    save_yml(configuration, file_path)

    return str(file_path)


@pytest.fixture
def configuration_file_path_with_logits_loss(tmp_path: Path, configuration_file_path: str) -> str:
    """Get model configuration file path for tests with BCEWithLogitsLoss."""
    file_path = tmp_path / "configuration_with_logits_loss.yaml"

    configuration = load_yml(configuration_file_path)
    # Change hit output to use BCEWithLogitsLoss
    configuration["variables"]["outputs"]["hit"]["loss"] = {
        "class_name": "BCEWithLogitsLoss",
        "reduction": "sum",
    }
    save_yml(configuration, file_path)

    return str(file_path)


@pytest.fixture
def configuration_file_path_hit_only_with_logits_loss(
    tmp_path: Path, configuration_file_path: str
) -> str:
    """Get model configuration file path for tests with only hit output using BCEWithLogitsLoss."""
    file_path = tmp_path / "configuration_hit_only_with_logits_loss.yaml"

    configuration = load_yml(configuration_file_path)
    del configuration["variables"]["outputs"]["binding_affinity"]
    # Change hit output to use BCEWithLogitsLoss
    configuration["variables"]["outputs"]["hit"]["loss"] = {
        "class_name": "BCEWithLogitsLoss",
        "reduction": "sum",
    }
    save_yml(configuration, file_path)

    return str(file_path)


@pytest.fixture
def input_dataset_file_path(dataframe: pl.DataFrame, tmp_path: Path) -> str:
    """Get the file path of the input dataset with 'peptide' and 'random' columns."""
    file_path = tmp_path / "input_peptides_data.csv"

    dataframe.select(["nn_align", "random", "allele"]).rename({"nn_align": "peptide"}).write_csv(
        file_path
    )

    return str(file_path)
