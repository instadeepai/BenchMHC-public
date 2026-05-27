"""Unit tests related to bench_mhc/cli/calibrate.py."""

from pathlib import Path

import polars as pl
import polars.testing as pl_testing

from bench_mhc.cli.calibrate import build_reference_set_from_peptides_and_alleles_mapping


def test_build_reference_set(tmp_path: Path) -> None:
    """Test build_reference_set_from_peptides_and_alleles_mapping."""
    peptides = ["AAAAA", "CCCCC"]
    df = pl.DataFrame({"peptide": peptides})
    file_path = tmp_path / "reference_set.csv"
    df.write_csv(file_path)

    lf_alleles = pl.LazyFrame({"allele": ["HLA__A0107", "HLA__A0108"]})

    df_reference = build_reference_set_from_peptides_and_alleles_mapping(
        reference_path=file_path,
        lf_alleles=lf_alleles,
    )

    expected_df = pl.DataFrame(
        {
            "peptide": ["AAAAA", "AAAAA", "CCCCC", "CCCCC"],
            "allele": ["HLA__A0107", "HLA__A0108", "HLA__A0107", "HLA__A0108"],
        }
    )

    pl_testing.assert_frame_equal(
        df_reference,
        expected_df,
        check_row_order=False,
        check_column_order=False,
    )
