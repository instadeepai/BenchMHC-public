"""Integration tests for the command lines in bench_mhc/cli/format_allele_feature_command.py."""

from pathlib import Path

import polars as pl
import polars.testing as pl_testing
import pytest
import rich_click as click
from click.testing import CliRunner

from bench_mhc.__main__ import main
from bench_mhc.cli.format_allele_feature_command import format_allele_feature


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
def test_format_allele_feature(
    tmp_path: Path,
    input_dataset_file_path: str,
    expected_nn_align_processed_mhc1_df: pl.DataFrame,
    output_file_name: str | None,
    output_file_exists: bool,
    force: bool,
) -> None:
    """Check that the format-allele-feature command works as expected.

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
            runner.invoke(format_allele_feature, parameters, catch_exceptions=False)

    else:
        result = runner.invoke(format_allele_feature, parameters)
        assert result.exit_code == 0, f"The command 'format-allele-feature' failed - {result}"

        if output_file_path is not None:
            assert output_file_path.exists(), "The output file has not been created"

        expected_df = pl.DataFrame(
            {
                "peptide": expected_nn_align_processed_mhc1_df["peptide"],
                "random": expected_nn_align_processed_mhc1_df["random"],
                "allele": [
                    "HLA__A0201",
                    "HLA__DRB10201",
                    "HLA__DPA10301=DPB10305",
                    "HLA__DRA10101=DRB10201",
                    "unparsable",
                    "Mamu__A20511",
                ],
            }
        )

        actual_df = pl.read_csv(to_read, null_values="")
        pl_testing.assert_frame_equal(
            actual_df.select(sorted(actual_df.columns)),
            expected_df.select(sorted(expected_df.columns)),
        )


@pytest.mark.parametrize("output_file_name", [None, "output_no_allele_column.csv"])
def test_format_allele_feature_wrong_column(
    tmp_path: Path, input_dataset_file_path: str, output_file_name: str | None
) -> None:
    """Check format-allele-feature command fails with a non-existing allele column."""
    parameters = [
        "--dataset_path",
        input_dataset_file_path,
        "--allele_column_name",
        "unknown_allele_column",
    ]

    if output_file_name is not None:
        output_filepath = tmp_path / output_file_name
        parameters.extend(["--output_file_path", str(output_filepath)])

    msg = (
        f"The dataset {input_dataset_file_path} does not have the 'unknown_allele_column' column "
        "to be formatted."
    )
    runner = CliRunner()
    with pytest.raises(ValueError, match=msg):
        runner.invoke(format_allele_feature, parameters, catch_exceptions=False)


def test_format_allele_feature_missing_options(tmp_path: Path) -> None:
    """Test the 'format_allele_feature' command does not work when options are missing.

    We also tests relevant errors are raised when the file paths do not exist.

    Args:
        tmp_path: Path to the temporary test directory.
    """
    runner = CliRunner()
    parameters = ["format-allele-feature"]
    dataset_path = tmp_path / "dataset.csv"
    result = runner.invoke(main, parameters)
    assert result.exit_code != 0
    assert "Missing option '--dataset_path' / '-d'" in click.unstyle(result.output)

    parameters.extend(["--dataset_path", str(dataset_path)])
    result = runner.invoke(main, parameters)
    assert result.exit_code != 0
    assert "Invalid value for '--dataset_path' / '-d'" in click.unstyle(result.output)
