"""Unit tests related to bench_mhc/utils/kmers.py."""

from pathlib import Path

import polars as pl
import polars.testing as pl_testing
import pytest

from bench_mhc.utils.kmers import generate_kmers_from_proteome
from bench_mhc.utils.kmers import generate_kmers_sampling_pool
from bench_mhc.utils.kmers import load_human_proteome
from bench_mhc.utils.kmers import load_swissprot_proteome


@pytest.mark.parametrize("include_cysteine_info", [True, False])
@pytest.mark.parametrize("flank_size", [0, 1])
def test_generate_kmers_from_proteome(
    include_cysteine_info: bool,
    flank_size: int,
) -> None:
    """Test the generation of k-mers from proteome sequences."""
    df_proteome = pl.DataFrame(
        {
            "protein_id": ["P12345", "P23456"],
            "sequence": ["TESTPEPP", "PEPTESTC"],
        }
    )

    df_kmers = generate_kmers_from_proteome(
        df_proteome,
        kmer_length=7,
        include_cysteine_info=include_cysteine_info,
        flank_size=flank_size,
    ).collect()

    expected_df = pl.DataFrame(
        {
            "peptide": ["TESTPEP", "ESTPEPP", "PEPTEST", "EPTESTC"],
            "protein_id": ["P12345", "P12345", "P23456", "P23456"],
        },
        schema={"peptide": pl.String, "protein_id": pl.String},
    )
    if flank_size != 0:
        expected_df = expected_df.with_columns(
            pl.Series(["-", "T", "-", "P"], dtype=pl.String).alias("left_flank"),
            pl.Series(["P", "-", "C", "-"], dtype=pl.String).alias("right_flank"),
        )

    if include_cysteine_info:
        expected_df = expected_df.with_columns(
            pl.Series([False, False, False, True], dtype=pl.Boolean).alias("with_c")
        )

    pl_testing.assert_frame_equal(
        df_kmers, expected_df, check_row_order=False, check_column_order=False
    )


def test_generate_kmers_sampling_pool() -> None:
    """Test the generation of the sampling pool for k-mers."""
    lf_kmers = pl.LazyFrame(
        {
            "peptide": ["TESCPEP", "PEPTEST", "CYSCTEST"],
            "with_c": [True, False, True],
            "protein_id": ["P1", "P1", "P2"],
        }
    )

    sampling_pool = generate_kmers_sampling_pool(lf_kmers)

    expected_sampling_pool = {
        True: {"P1": {"TESCPEP"}, "P2": {"CYSCTEST"}},
        False: {"P1": {"PEPTEST"}},
    }

    assert sampling_pool == expected_sampling_pool


def test_load_swissprot_proteome(tmpdir: Path) -> None:
    """Test the loading of the SwissProt proteome."""
    swissprot_db_path = tmpdir / "swissprot_db.fasta"

    fasta_content = """>prot1
TESTPEPABHDEFGHIJKLMNOPQRSTUVWXYZGHJJGFFTHJKKHJHJ
>prot2
PEPTESTABHDEFGHIJKLMNOPQRSTUVWXYZ
>prot3
ATTABCDEFGHIJKLMNOPQRSTUVWXYZ
>prot4
ABCDEFGHIJKLMNOPQRSTUVWXYZEE
>prot5
CYSCYSABCDEFGHIJKLMNOPQRSTUVWXYZKZAKFJZJ
>to_remove
TESTCYSABCDEFGHIJKLMNOPQRSTUVWXYZ*
"""

    swissprot_db_path.write_text(fasta_content, encoding="utf-8")

    df_swissprot_proteome = load_swissprot_proteome(swissprot_db_path)

    expected_df = pl.DataFrame(
        {
            "protein_id": ["prot1", "prot2", "prot3", "prot4", "prot5"],
            "sequence": [
                "TESTPEPABHDEFGHIJKLMNOPQRSTUVWXYZGHJJGFFTHJKKHJHJ",
                "PEPTESTABHDEFGHIJKLMNOPQRSTUVWXYZ",
                "ATTABCDEFGHIJKLMNOPQRSTUVWXYZ",
                "ABCDEFGHIJKLMNOPQRSTUVWXYZEE",
                "CYSCYSABCDEFGHIJKLMNOPQRSTUVWXYZKZAKFJZJ",
            ],
        }
    )
    pl_testing.assert_frame_equal(df_swissprot_proteome, expected_df)


def test_load_human_proteome(tmpdir: Path) -> None:
    """Test the loading of the human proteome."""
    human_proteome_path = tmpdir / "human_proteome.fasta"
    fasta_content = """>prot1
TESTPEPABHDEFGHIJKLMNOPQRSTUVWXYZGHJJGFFTHJKKHJHJ
>prot2
PEPTESTABHDEFGHIJKLMNOPQRSTUVWXYZ
>prot3
ATTABCDEFGHIJKLMNOPQRSTUVWXYZ
>prot4
ABCDEFGHIJKLMNOPQRSTUVWXYZEE
>to_remove
CYSCYSABCDEFGHIJKLMNOPQRSTUVWXYZKZAKFJZJ*
    """
    human_proteome_path.write_text(fasta_content, encoding="utf-8")

    df_human_proteome = load_human_proteome(human_proteome_path)
    expected_df = pl.DataFrame(
        {
            "protein_id": ["prot1", "prot2", "prot3", "prot4"],
            "sequence": [
                "TESTPEPABHDEFGHIJKLMNOPQRSTUVWXYZGHJJGFFTHJKKHJHJ",
                "PEPTESTABHDEFGHIJKLMNOPQRSTUVWXYZ",
                "ATTABCDEFGHIJKLMNOPQRSTUVWXYZ",
                "ABCDEFGHIJKLMNOPQRSTUVWXYZEE",
            ],
        }
    )
    pl_testing.assert_frame_equal(df_human_proteome, expected_df)
