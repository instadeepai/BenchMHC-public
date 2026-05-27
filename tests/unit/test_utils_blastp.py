"""Module used to define all the tests linked to bench_mhc/utils/blastp.py module."""

from pathlib import Path

import pytest

from bench_mhc.utils.blastp import build_blast_db
from bench_mhc.utils.blastp import get_blastp_output


@pytest.fixture
def proteome_path(tmp_path: Path) -> Path:
    """Return the path to the proteome file."""
    proteome_file_path = tmp_path / "proteome.fasta"

    with proteome_file_path.open("w") as f:
        f.write(">protein3\n")
        f.write("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG\n")
        f.write(">protein2\n")
        f.write("SIINFEKLLERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG\n")

    return proteome_file_path


def test_build_blast_db(proteome_path: Path, tmp_path: Path) -> None:
    """Test the build_blast_db function."""
    blast_db_name = "dummy_blast_db"
    blast_db_output_path = tmp_path / blast_db_name

    build_blast_db(
        proteome_path=proteome_path,
        blast_db_output_path=str(blast_db_output_path),
        blast_db_name=str(blast_db_name),
    )

    # Check that all expected BLAST database files are created
    expected_files = [
        blast_db_output_path.with_suffix(".ptf"),
        blast_db_output_path.with_suffix(".phr"),
        blast_db_output_path.with_suffix(".pin"),
        blast_db_output_path.with_suffix(".pjs"),
        blast_db_output_path.with_suffix(".pot"),
        blast_db_output_path.with_suffix(".pdb"),
        blast_db_output_path.with_suffix(".pto"),
        blast_db_output_path.with_suffix(".psq"),
    ]

    for file_path in expected_files:
        assert file_path.exists()


@pytest.mark.parametrize("word_size", ["2", "3"])
def test_get_blastp_output(word_size: str, proteome_path: Path, tmp_path: Path) -> None:
    """Test the get_blastp_output function."""
    blast_db_name = "dummy_blast_db"
    blast_db_output_path = tmp_path / blast_db_name

    build_blast_db(
        proteome_path=proteome_path,
        blast_db_output_path=str(blast_db_output_path),
        blast_db_name=str(blast_db_name),
    )

    peptides = ["GYNIVA", "GYNI"]

    blastp_output = get_blastp_output(
        peptides=peptides,
        blast_db_name=blast_db_name,
        blast_db_path=str(blast_db_output_path.parent),
        num_threads=1,
        word_size=word_size,
    )

    # Check that the output is not empty
    # No additional check is possible because the output is very hard to predict
    # and depends on the word size. Higher word size will have less hits even if the
    # peptides are perfectly matched.
    assert len(blastp_output) != 0


def tests_get_blastp_output_no_match(proteome_path: Path, tmp_path: Path) -> None:
    """Tests blastp search when no match found."""
    blast_db_name = "dummy_blast_db"
    blast_db_output_path = tmp_path / blast_db_name

    build_blast_db(
        proteome_path=proteome_path,
        blast_db_output_path=str(blast_db_output_path),
        blast_db_name=str(blast_db_name),
    )

    peptides = ["AAAAAAAAAAA"]

    df_blastp_output = get_blastp_output(
        peptides=peptides,
        blast_db_name=blast_db_name,
        blast_db_path=str(blast_db_output_path.parent),
        num_threads=1,
        # Ensure there cannot be any match with high word size
        word_size="7",
    )

    assert df_blastp_output.is_empty()

    assert set(df_blastp_output.columns) == {
        "qseqid",
        "sseqid",
        "stitle",
        "sstart",
        "send",
        "evalue",
        "bitscore",
    }
