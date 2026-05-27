"""Integration tests related to bench_mhc/cli/refine_peptides_command.py."""

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from click.testing import CliRunner

from bench_mhc.cli.refine_peptides_command import refine_peptides


@pytest.fixture
def peptides_csv(tmp_path: Path) -> Path:
    """Create a CSV file with peptides."""
    df = pl.DataFrame(
        {
            "peptide": [
                "TESTPEP",  # Exact match, multiple proteins
                "IMPERFECT",  # Imperfect match
                "DECOYPEP",
            ],
            "protein_id": [
                "P12345|TEST1;P67890|TEST2",
                "sp|Q9H2K5|TEST10",  # Imperfect
                "sp|Q9H2K9|TEST12",
            ],
            "origin_species": [
                "Homo sapiens;Homo sapiens",
                "Lizard",
                "Lizard",
            ],
            "left_flank": ["A;C", "E", "F"],
            "right_flank": ["X;Y", "Y", "U"],
            "exact_protein_match": [1, 0, 1],
            "hit": [1, 1, 0],
        }
    )
    peptides_path = tmp_path / "peptides_with_features.csv"
    df.write_csv(peptides_path)

    return peptides_path


def test_refine_peptides_remove_multiple_matches(peptides_csv: Path, tmp_path: Path) -> None:
    """Test the refine-peptides command with --remove_multiple_matches option."""
    output_path = tmp_path / "output.csv"

    runner = CliRunner()
    result = runner.invoke(
        refine_peptides,
        [
            "--peptides_path",
            str(peptides_csv),
            "--output_file_path",
            str(output_path),
            "--flank_size",
            "1",
            "--remove_multiple_matches",
        ],
    )
    assert result.exit_code == 0

    df_output = pl.read_csv(output_path)
    assert {"DECOYPEP", "TESTPEP", "IMPERFECT"}.issubset(df_output["peptide"].unique())
    # Should have only one protein_id per row
    testpep_row = df_output.filter(pl.col("peptide") == "TESTPEP")
    contains_multiple_proteins = testpep_row.select(
        pl.all_horizontal(pl.col("protein_id", "left_flank", "right_flank").str.contains(";"))
    ).item()
    assert not contains_multiple_proteins


def test_refine_peptides_remove_imperfect_matching(peptides_csv: Path, tmp_path: Path) -> None:
    """Test the refine-peptides command with --remove_imperfect_matching option."""
    output_path = tmp_path / "output.csv"

    runner = CliRunner()
    result = runner.invoke(
        refine_peptides,
        [
            "--peptides_path",
            str(peptides_csv),
            "--output_file_path",
            str(output_path),
            "--flank_size",
            "1",
            "--remove_imperfect_matching",
        ],
    )
    assert result.exit_code == 0

    df_output = pl.read_csv(output_path)
    assert "IMPERFECT" not in df_output["peptide"].unique()
    assert {"DECOYPEP", "TESTPEP"}.issubset(df_output["peptide"].unique())


def test_refine_peptides_nullify_imperfect_flanks(peptides_csv: Path, tmp_path: Path) -> None:
    """Test the refine-peptides command with --nullify_imperfect_flanks option."""
    output_path = tmp_path / "output.csv"

    runner = CliRunner()
    result = runner.invoke(
        refine_peptides,
        [
            "--peptides_path",
            str(peptides_csv),
            "--output_file_path",
            str(output_path),
            "--flank_size",
            "1",
            "--nullify_imperfect_flanks",
        ],
    )
    assert result.exit_code == 0

    df_output = pl.read_csv(output_path)
    imperfect_row = df_output.filter(pl.col("peptide") == "IMPERFECT")
    assert imperfect_row["left_flank"].item() == "-"
    assert imperfect_row["right_flank"].item() == "-"


def test_refine_peptides_errors(peptides_csv: Path, tmp_path: Path) -> None:
    """Test error cases: mutually exclusive flags and missing columns."""
    runner = CliRunner()

    # Test mutually exclusive flags
    result = runner.invoke(
        refine_peptides,
        [
            "--peptides_path",
            str(peptides_csv),
            "--flank_size",
            "1",
            "--remove_imperfect_matching",
            "--nullify_imperfect_flanks",
        ],
    )
    assert result.exit_code != 0
    assert (
        "Flags 'remove_imperfect_matching' and 'nullify_imperfect_flanks' "
        "are mutually exclusive, you should pick only one." in str(result.exception)
    )

    # Test missing required columns
    invalid_csv = tmp_path / "invalid.csv"
    pl.DataFrame({"peptide": ["TEST"]}).write_csv(invalid_csv)

    result = runner.invoke(
        refine_peptides,
        ["--peptides_path", str(invalid_csv), "--flank_size", "1"],
    )
    assert result.exit_code != 0
    assert (
        "Input file is missing required columns: "
        "'exact_protein_match' | 'left_flank' | 'origin_species' | "
        "'protein_id' | 'right_flank'. Please run 'assign-protein-features' "
        "first to generate these columns." in str(result.exception)
    )

    # Test null protein_id filtering
    null_csv = tmp_path / "null.csv"
    pl.DataFrame(
        {
            "peptide": ["VALID", "NULL"],
            "protein_id": ["P12345|TEST1", None],
            "origin_species": ["Homo sapiens", "Homo sapiens"],
            "left_flank": ["A", "B"],
            "right_flank": ["X", "Y"],
            "exact_protein_match": [1, 1],
        }
    ).write_csv(null_csv)
    with patch("bench_mhc.cli.refine_peptides.log") as log_mock:
        result = runner.invoke(
            refine_peptides,
            [
                "--peptides_path",
                str(null_csv),
                "--flank_size",
                "1",
                "--output_file_path",
                str(tmp_path / "out.csv"),
            ],
        )
    assert result.exit_code == 0
    assert "NULL" not in pl.read_csv(tmp_path / "out.csv")["peptide"].unique()
    log_mock.warning.assert_called_once_with(
        "Found 1 peptide(s) with no associated protein ID. " "Those peptides will be ignored."
    )
