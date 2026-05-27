"""Integration tests related to bench_mhc/cli/assign_protein_features_command.py."""

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import polars as pl
import pytest
from click.testing import CliRunner
from polars import testing as pl_testing

from bench_mhc.cli.assign_protein_features_command import assign_protein_features
from bench_mhc.utils.blastp import get_blastp_output


@pytest.fixture
def mock_get_blastp_output_fn() -> Callable[[list[str], int, str, str, Any], pl.DataFrame]:
    """Mock the get_blastp_output function to always return a match."""
    return lambda peptides, num_threads, blast_db_name, blast_db_path, **kwargs: get_blastp_output(
        peptides=peptides,  # List of peptides
        blast_db_name=blast_db_name,
        blast_db_path=blast_db_path,
        num_threads=num_threads,
        word_size="2",
        evalue="1000000",
        matrix="BLOSUM62",
        threshold="5",
        **kwargs,
    )


@pytest.fixture
def peptides_csv(tmp_path: Path) -> Path:
    """Create a CSV file with peptides.

    Returns:
        Path to the created CSV file containing peptides with a column 'peptide'.

    Note:
        The test data includes:
            - Peptides of different lengths (2-16 amino acids).
            - Split between peptides found in proteome and those requiring SwissProt:
                - Proteome matches: TESTPEP, PEPTEST, TT, EE, CYSABC
                - SwissProt pseudo matches: KZAKFAZJ, FAYAAAAAC
                - No match: BBBBBBBB and KKKKKKKKKKKKKKKK
    """
    df = pl.DataFrame(
        {
            "peptide": [
                # Peptides found in human proteome
                "TESTPEP",  # 7-mer
                "PEPTEST",  # 7-mer
                "TT",  # 2-mer
                "EE",  # 2-mer
                "CYSABC",  # 6-mer with C
                # Peptides requiring SwissProt
                "KZAKFAZJ",  # 8-mer
                "FAYAAAAAC",  # 9-mer with C
                # Peptides with no match and no pseudo match
                "BBBBBBBB",  # 8-mer, will be processed with other 8-mers that have pseudo match
                "K" * 16,  # 16-mer, will be processed alone (no pseudo match for 16-mers)
            ],
            "original_fold": ["fold1"] * 9,
            "dummy_column": ["dummy1"] * 9,
        }
    )
    peptides_path = tmp_path / "peptides.csv"
    df.write_csv(peptides_path)

    return peptides_path


@pytest.fixture
def proteome_fasta(tmp_path: Path) -> Path:
    """Create a FASTA file with proteome sequences.

    Returns:
        Path to the created FASTA file containing protein sequences with headers:
            - sp|P12345|TEST1: Contains TESTPEP sequence.
            - sp|P67890|TEST2: Contains PEPTEST sequence.
            - sp|P24680|TEST3: Contains TT sequence.
            - sp|P13579|TEST4: Contains EE sequence.
            - sp|P98765|TEST5: Contains CYSABC sequence.
            - sp|P54321|TEST6: Contains CYSABC sequence.
    """
    fasta_content = """>P12345|TEST1
TESTPEPABHDEFGHIJKLMNOPQRSTUVWXYZGHJJGFFTHJKKHJHJ
>P67890|TEST2
PEPTESTABHDEFGHIJKLMNOPQRSTUVWXYZ
>P24680|TEST3
ATTABCDEFGHIJKLMNOPQRSTUVWXYZ
>P13579|TEST4
ABCDEFGHIJKLMNOPQRSTUVWXYZEE
>P98765|TEST5
CYSCYSABCDEFGHIJKLMNOPQRSTUVWXYZKZAKFJZJ
>P54321|TEST6
TESTCYSABCDEFGHIJKLMNOPQRSTUVWXYZ"""

    proteome_path = tmp_path / "proteome.fasta"
    proteome_path.write_text(fasta_content)

    return proteome_path


@pytest.fixture
def swissprot_fasta(tmp_path: Path) -> Path:
    """Create a SwissProt FASTA file for BLASTP search.

    Returns:
        Path to the created SwissProt FASTA file containing protein sequences with headers:
            - sp|Q9H2K6|TEST9: Contains FAYAAAAAC pseudo match sequence.
            - sp|Q9H2K5|TEST10: Contains KZAKFAZJ pseudo match sequence.
            - sp|Q9H2K8|TEST11: Contains KZAKFAZJ pseudo match sequence.
    """
    swissprot_path = tmp_path / "swissprot.fasta"
    with open(swissprot_path, "w") as f:
        f.write(">sp|Q9H2K6|TEST9 Test protein 9 OS=Lizard OX=Q9H2K6\n")
        f.write("AYSATESABADEFAYAAAAAAAAZAZLONGCTZZZJWXYLOZZZ\n")
        f.write(">sp|Q9H2K5|TEST10 Test protein 10 OS=Lizard OX=Q9H2K6\n")
        f.write("AYSATESABADEFAYSATESOPKZAKFWXYLOZZZLONGATESABADEFGHIJKLLONGATEAKFAZJZLONTESTZ\n")
        f.write(">sp|Q9H2K8|TEST11 Test protein 11 OS=Lizard OX=Q9H2K8\n")
        f.write("LONGATESABADEFGHIJKLLONGATEAKFAZJZLONGATZZZ\n")

    return swissprot_path


@pytest.mark.parametrize("with_columns", [None, "original_fold", "all"])
@pytest.mark.parametrize("flank_size", [0, 6])
def test_assign_protein_features_command(
    flank_size: int,
    with_columns: str | None,
    peptides_csv: Path,
    proteome_fasta: Path,
    swissprot_fasta: Path,
    tmp_path: Path,
    mock_get_blastp_output_fn: pl.DataFrame,
) -> None:
    """Test the assign-protein-features command with sample data.

    Verify that the command successfully processes peptides from the CSV file, matches
    peptides to proteins in the proteome, extract flanks and outputs the results to the specified
    CSV file.

    Args:
        flank_size: Size of the flanks to consider.
        with_columns: Columns to copy from peptides to output.
        peptides_csv: Path to CSV file containing peptides.
        proteome_fasta: Path to FASTA file with proteome sequences.
        swissprot_fasta: Path to SwissProt FASTA file for BLASTP search.
        tmp_path: Path to temporary directory for test files.
        mock_get_blastp_output_fn: Mocked get_blastp_output function.
    """
    output_path = tmp_path / "output.csv"

    runner = CliRunner()
    params = [
        "--peptides_path",
        str(peptides_csv),
        "--flank_size",
        str(flank_size),
        "--human_proteome_path",
        str(proteome_fasta),
        "--swissprot_db_path",
        str(swissprot_fasta),
        "--output_file_path",
        str(output_path),
    ]
    if with_columns:
        params.append("--with_columns")
        params.append(with_columns)

    with (
        patch(
            "bench_mhc.cli.assign_protein_features.get_blastp_output",
            side_effect=mock_get_blastp_output_fn,
        ),
        patch(
            "bench_mhc.cli.assign_protein_features.tempfile.TemporaryDirectory",
            return_value=tmp_path,
        ),
    ):
        result = runner.invoke(
            assign_protein_features,
            params,
            catch_exceptions=False,
        )
    assert result.exit_code == 0

    df_output = pl.read_csv(output_path)

    # Check number of rows
    df_peptides = pl.read_csv(peptides_csv)
    expected_num_peptides = len(df_peptides)
    assert len(df_output) == expected_num_peptides

    # Check columns
    expected_columns = {
        "peptide",
        "protein_id",
        "origin_species",
        "left_flank",
        "right_flank",
        "exact_protein_match",
    }
    if with_columns:
        if with_columns == "all":
            assert set(df_output.columns) == expected_columns | {"dummy_column", "original_fold"}
            assert df_output["dummy_column"].to_list() == ["dummy1"] * expected_num_peptides
        else:
            assert set(df_output.columns) == expected_columns | {"original_fold"}

        assert df_output["original_fold"].to_list() == ["fold1"] * expected_num_peptides

    else:
        assert set(df_output.columns) == expected_columns

    # Check the exact matches
    df_exact_match_expected = pl.DataFrame(
        {
            "peptide": ["TESTPEP", "PEPTEST", "TT", "EE", "CYSABC"],
            "protein_id": [
                "P12345|TEST1",
                "P67890|TEST2",
                "P24680|TEST3",
                "P13579|TEST4",
                "P54321|TEST6;P98765|TEST5",
            ],
            "origin_species": ["Homo sapiens"] * 4 + ["Homo sapiens;Homo sapiens"],
            "exact_protein_match": [1] * 5,
        }
    )

    if flank_size == 0:
        df_exact_match_expected = df_exact_match_expected.with_columns(
            pl.lit(None).alias("left_flank"),
            pl.lit(None).alias("right_flank"),
        )
    else:
        df_exact_match_expected = df_exact_match_expected.with_columns(
            left_flank=pl.Series(
                [
                    "------",
                    "------",
                    "-----A",
                    "UVWXYZ",
                    "--TEST;---CYS",
                ],
                dtype=pl.String,
            ),
            right_flank=pl.Series(
                [
                    "ABHDEF",
                    "ABHDEF",
                    "ABCDEF",
                    "------",
                    "DEFGHI;DEFGHI",
                ],
                dtype=pl.String,
            ),
        )

    df_exact_match = df_output.filter(pl.col("exact_protein_match") == 1)

    pl_testing.assert_frame_equal(
        df_exact_match.select(expected_columns),
        df_exact_match_expected.select(expected_columns),
        check_row_order=False,
        check_column_order=False,
        check_dtypes=False,
    )

    # Check flank size
    df_flank_size = (
        df_output.select(["left_flank", "right_flank"])
        .with_columns(
            pl.col("left_flank").str.split(";"),
            pl.col("right_flank").str.split(";"),
        )
        .explode(["left_flank", "right_flank"])
    )
    left_flank_is_valid = (pl.col("left_flank").str.len_bytes() == flank_size).all()
    right_flank_is_valid = (pl.col("right_flank").str.len_bytes() == flank_size).all()

    assert df_flank_size.select(left_flank_is_valid).item()
    assert df_flank_size.select(right_flank_is_valid).item()

    # Check that the species distribution is correct
    df_species_distribution_expected = pl.DataFrame(
        {
            "origin_species": [
                "Homo sapiens",
                "Homo sapiens;Homo sapiens",
                "Lizard",
                None,
            ],
            "len": [4, 1, 2, 2],
        }
    )
    df_species_distribution = df_output.group_by("origin_species").agg(pl.len())

    pl_testing.assert_frame_equal(
        df_species_distribution,
        df_species_distribution_expected,
        check_dtypes=False,
        check_row_order=False,
    )

    assert (tmp_path / output_path.with_name(f"{output_path.name}_matching_stats.json")).exists()
