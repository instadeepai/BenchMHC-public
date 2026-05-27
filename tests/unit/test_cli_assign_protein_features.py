"""Unit tests for the assign_protein_features module."""

from pathlib import Path

import pytest

from bench_mhc.cli.assign_protein_features import assign_protein_features
from bench_mhc.utils.format import format_iterable


def test_match_peptides_to_proteins_with_columns_validation(tmp_path: Path) -> None:
    """Test the validation of with_columns parameter."""
    peptides_path = tmp_path / "peptides.csv"
    peptides_path.write_text("peptide,score,source\nTESTPEP,0.8,database1\nPEPTEST,0.6,database2\n")

    # Test with non-existent column
    with pytest.raises(
        ValueError,
        match="The following columns specified in --with_columns are not present in the input CSV: "
        f"{format_iterable({'non_existent_column'})}",
    ):
        assign_protein_features(
            peptides_path=peptides_path,
            flank_size=3,
            human_proteome_path=Path("tests/data/human_proteome.fasta"),
            swissprot_db_path=Path("tests/data/swissprot.fasta"),
            output_file_path=Path("tests/data/source_proteins.csv"),
            with_columns={"non_existent_column"},
        )


def test_assign_source_protein_errors(tmp_path: Path) -> None:
    """Test the assignment of source proteins with missing columns."""
    peptides_path = tmp_path / "peptides.csv"
    peptides_path.write_text("other\nTESTPEP\nPEPTEST\n")

    with pytest.raises(ValueError, match="Input CSV must contain a 'peptide' column"):
        assign_protein_features(
            peptides_path=peptides_path,
            flank_size=3,
            human_proteome_path=Path("tests/data/human_proteome.fasta"),
            swissprot_db_path=Path("tests/data/swissprot.fasta"),
            output_file_path=Path("tests/data/source_proteins.csv"),
            with_columns=None,
        )
