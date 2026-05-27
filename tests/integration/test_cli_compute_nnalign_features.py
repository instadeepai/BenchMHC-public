"""Integration tests for the command lines in bench_mhc/cli/compute_nnalign_features_command.py."""

from pathlib import Path

import polars as pl
import polars.testing as pl_testing
import pytest
import rich_click as click
from click.testing import CliRunner

from bench_mhc.__main__ import main
from bench_mhc.cli.compute_nnalign_features_command import compute_nnalign_features
from bench_mhc.variables import NNAlignVariable


@pytest.mark.parametrize(
    ("output_file_name", "output_file_exists", "force"),
    [
        (None, False, True),
        (None, False, False),
        ("output.csv", False, False),
        ("output.csv", True, False),
        ("output.csv", True, True),
    ],
)
def test_compute_nnalign_features(
    tmp_path: Path,
    input_dataset_file_path: str,
    expected_nn_align_processed_mhc1_df: pl.DataFrame,
    output_file_name: str | None,
    output_file_exists: bool,
    force: bool,
) -> None:
    """Check that the compute-nnalign-features command works as expected.

    We check the following cases:
    - if an output file path is provided,
    - if the provided output file path already exists,
    - if --force is provided.
    """
    parameters = [
        "--dataset_path",
        input_dataset_file_path,
    ]

    if force:
        parameters.append("--force")

    output_file_path = None
    if output_file_name is not None:
        output_file_path = tmp_path / output_file_name
        parameters.extend(["--output_file_path", str(output_file_path)])

        if output_file_exists:
            output_file_path.touch()

        to_read = str(output_file_path)

    else:
        to_read = input_dataset_file_path

    runner = CliRunner()

    if not force and (output_file_name is None or output_file_exists):
        msg = f"The file {to_read} already exists. Use --force if you want to overwrite it."
        with pytest.raises(FileExistsError, match=msg):
            runner.invoke(compute_nnalign_features, parameters, catch_exceptions=False)

    else:
        result = runner.invoke(compute_nnalign_features, parameters)
        assert result.exit_code == 0, f"The command 'compute_nnalign_features' failed - {result}"

        if output_file_path is not None:
            assert output_file_path.exists(), "The output file has not been created"

        column_name2new_column_name = {
            f"nn_align_{suffix}": f"peptide_{suffix}" for suffix in NNAlignVariable.suffixes
        }
        expected_df = expected_nn_align_processed_mhc1_df.rename(column_name2new_column_name)

        actual_df = pl.read_csv(to_read, null_values="")
        pl_testing.assert_frame_equal(
            actual_df.select(sorted(actual_df.columns)),
            expected_df.select(sorted(expected_df.columns)),
        )


@pytest.mark.parametrize("output_file_name", [None, "output_no_peptide_column.csv"])
def test_compute_nnalign_features_wrong_column(
    tmp_path: Path, input_dataset_file_path: str, output_file_name: str | None
) -> None:
    """Check compute-nnalign-features command fails with a non-existing peptide column."""
    parameters = [
        "--dataset_path",
        input_dataset_file_path,
        "--peptide_column_name",
        "unknown_peptide_column",
    ]

    if output_file_name is not None:
        output_filepath = tmp_path / output_file_name
        parameters.extend(["--output_file_path", str(output_filepath)])

    msg = (
        f"The dataset {input_dataset_file_path} does not have the 'unknown_peptide_column' column "
        "required to compute the NetMHCpan-4.1 features."
    )
    runner = CliRunner()
    with pytest.raises(ValueError, match=msg):
        runner.invoke(compute_nnalign_features, parameters, catch_exceptions=False)


def test_compute_nnalign_features_missing_options(tmp_path: Path) -> None:
    """Test the 'compute-nnalign-features' command does not work when options are missing.

    We also tests relevant errors are raised when the file paths do not exist.

    Args:
        tmp_path: Path to the temporary test directory.
    """
    runner = CliRunner()
    parameters = ["compute-nnalign-features"]
    dataset_path = tmp_path / "dataset.csv"
    result = runner.invoke(main, parameters)
    assert result.exit_code != 0
    assert "Missing option '--dataset_path' / '-d'" in click.unstyle(result.output)

    parameters.extend(["--dataset_path", str(dataset_path)])
    result = runner.invoke(main, parameters)
    assert result.exit_code != 0
    assert "Invalid value for '--dataset_path' / '-d'" in click.unstyle(result.output)
