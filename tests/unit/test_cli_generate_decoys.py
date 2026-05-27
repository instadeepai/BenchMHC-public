"""Unit tests for the generate_decoys module."""

from pathlib import Path

import polars as pl
import pytest

from bench_mhc.cli.generate_decoys import generate_decoys
from bench_mhc.cli.generate_decoys import generate_decoys_df
from bench_mhc.cli.generate_decoys import get_decoys_sample_for_hit
from bench_mhc.cli.generate_decoys import sample_one_protein
from bench_mhc.utils.format import format_iterable


def test_get_hit_decoys_sample() -> None:
    """Test the generation of decoy samples for a hit peptide.

    Verify that the function correctly generates decoys for both cysteine-containing
    and non-cysteine peptides maintaining same allele and origin species.
    """
    with_c2protein_id2peptides = {
        False: {
            "P1": {"TESTPEPG", "PEPTESTG", "OTHERGGG"},
            "P2": {"TESTPEPG", "HHHHHHHH", "OTHERGGG"},
        },
        True: {"P2": {"CYSCTEST", "TESTCYSG", "OTHERCGG"}},
    }

    # Test with a non-cysteine peptide
    df_decoys = get_decoys_sample_for_hit(
        peptide="TESTPEPG",
        protein_ids=["P1", "P2"],
        with_c2protein_id2peptides=with_c2protein_id2peptides,
        peptide_to_reject={"TESTPEPG", "PEPTESTG"},
        num_decoys=2,
        origin_species=["Homo sapiens"],
    )
    assert len(df_decoys) == 2  # 2 decoys
    assert all("C" not in p for p in df_decoys["peptide"])  # No cysteine in decoys
    assert df_decoys["origin_species"].unique().item() == "Homo sapiens"

    # Test with a cysteine peptide
    df_decoys = get_decoys_sample_for_hit(
        peptide="CYSCTEST",
        protein_ids=["P2"],
        with_c2protein_id2peptides=with_c2protein_id2peptides,
        peptide_to_reject={"CYSCTEST"},
        num_decoys=2,
        origin_species=["Homo sapiens"],
    )

    assert len(df_decoys) == 2  # 2 decoys
    assert all("C" in p for p in df_decoys["peptide"])  # All have cysteine
    assert df_decoys["origin_species"].unique().item() == "Homo sapiens"


def test_generate_hit_decoys_df() -> None:
    """Test the generation of hit-decoy DataFrames."""
    df_hits = pl.DataFrame(
        {
            "peptide": ["TESTPEP", "PEPTEST"],
            "allele": ["HLA-A*02:01"] * 2,
            "origin_species": ["Homo sapiens"] * 2,
            "protein_id": ["P1", "P2"],
            "left_flank": ["A", "A"],
            "right_flank": ["B", "B"],
            "exact_protein_match": [1, 1],
        }
    )

    # Create allele2peptide_to_reject with sets
    allele2peptide_to_reject = {
        row["allele"]: set(row["peptide"])
        for row in df_hits.group_by("allele").agg(pl.col("peptide").unique()).iter_rows(named=True)
    }

    with_c2protein_id2peptides = {
        False: {
            "P1": {"TESTPEP", "OTHER1"},
            "P2": {"PEPTEST", "OTHER2"},
        }
    }

    dfs, num_hits_processed = generate_decoys_df(
        df_hits=df_hits,
        with_c2protein_id2peptides=with_c2protein_id2peptides,
        allele2peptide_to_reject=allele2peptide_to_reject,
        num_decoys=2,
        num_hits_processed=0,
        with_columns=None,
    )

    assert len(dfs) == 2  # One DataFrame per hit
    assert num_hits_processed == 2  # Both hits processed


def test_generate_hit_decoys_df_with_columns() -> None:
    """Test the generation of hit-decoy DataFrames with additional columns."""
    df_hits = pl.DataFrame(
        {
            "peptide": ["TESTPEP", "PEPTEST"],
            "allele": ["HLA-A*02:01"] * 2,
            "hit": [1, 1],
            "score": [0.8, 0.6],
            "source": ["database1", "database2"],
            "protein_id": ["P1", "P2"],
            "origin_species": ["Homo sapiens"] * 2,
            "left_flank": ["A", "A"],
            "right_flank": ["B", "B"],
            "exact_protein_match": [1, 1],
        }
    )

    # Create allele2peptide_to_reject with sets
    allele2peptide_to_reject = {
        row["allele"]: set(row["peptide"])
        for row in df_hits.group_by("allele").agg(pl.col("peptide").unique()).iter_rows(named=True)
    }

    with_c2protein_id2peptides = {
        False: {
            "P1": {"TESTPEP", "OTHER1"},
            "P2": {"PEPTEST", "OTHER2"},
        }
    }

    # Test with additional columns
    dfs, num_hits_processed = generate_decoys_df(
        df_hits=df_hits,
        with_c2protein_id2peptides=with_c2protein_id2peptides,
        allele2peptide_to_reject=allele2peptide_to_reject,
        num_decoys=2,
        num_hits_processed=0,
        with_columns={"score", "source"},
    )

    assert len(dfs) == 2  # One DataFrame per hit
    assert num_hits_processed == 2  # Both hits processed

    # Check that additional columns are copied to decoys
    for df in dfs:
        assert "score" in df.columns
        assert "source" in df.columns
        # All decoys should have the same score and source as the original hit
        assert df["score"].n_unique() == 1
        assert df["source"].n_unique() == 1


def test_generate_decoys_with_columns_validation(tmp_path: Path) -> None:
    """Test the validation of with_columns parameter."""
    hits_path = tmp_path / "hits.csv"
    hits_path.write_text(
        "peptide,allele,protein_id,origin_species,score,left_flank,right_flank,exact_protein_match\n"
        "TESTPEP,HLA-A*02:01,P1,Homo sapiens,0.8,A,B,1\n"
        "PEPTEST,HLA-A*02:01,P2,Homo sapiens,0.6,A,B,1\n"
    )

    # Test with non-existent column
    match_msg = (
        "The following columns specified in --with_columns are not present in the input CSV: "
        f"{format_iterable({'non_existent_column'})}"
    )
    with pytest.raises(ValueError, match=match_msg):
        generate_decoys(
            hits_path=hits_path,
            # Dummy paths that must be set and must exist
            human_proteome_path=hits_path,
            output_file_path=hits_path,
            num_decoys=2,
            swissprot_db_path=hits_path,
            flank_size=0,
            remove_multiple_matches=False,
            remove_imperfect_matching=False,
            nullify_imperfect_flanks=False,
            with_columns={"non_existent_column"},
        )


def test_generate_decoys_errors(tmp_path: Path) -> None:
    """Test the generation of decoys with missing columns."""
    hits_path = tmp_path / "hits.csv"
    hits_path.write_text(
        "other,allele,protein_id,origin_species,left_flank,right_flank,exact_protein_match\n"
        "TESTPEP,HLA-A*02:01,P1,Homo sapiens,A,B,1\n"
        "PEPTEST,HLA-A*02:01,P2,Homo sapiens,A,B,1\n"
    )

    with pytest.raises(ValueError, match="Input CSV must contain a 'peptide' column"):
        generate_decoys(
            hits_path=hits_path,
            # Dummy paths that must be set and must exist
            human_proteome_path=hits_path,
            output_file_path=hits_path,
            num_decoys=2,
            swissprot_db_path=hits_path,
            flank_size=0,
            remove_multiple_matches=False,
            remove_imperfect_matching=False,
            nullify_imperfect_flanks=False,
            with_columns=None,
        )

    hits_path.write_text(
        "peptide,other,protein_id,origin_species,left_flank,right_flank,exact_protein_match\n"
        "TESTPEP,HLA-A*02:01,P1,Homo sapiens,A,B,1\n"
        "PEPTEST,HLA-A*02:01,P2,Homo sapiens,A,B,1\n"
    )

    with pytest.raises(ValueError, match="Input CSV must contain a 'allele' column"):
        generate_decoys(
            hits_path=hits_path,
            # Dummy paths that must be set and must exist
            human_proteome_path=hits_path,
            output_file_path=hits_path,
            num_decoys=2,
            swissprot_db_path=hits_path,
            flank_size=0,
            remove_multiple_matches=False,
            remove_imperfect_matching=False,
            nullify_imperfect_flanks=False,
            with_columns=None,
        )

    hits_path.write_text("peptide,allele\nTESTPEP,HLA-A*02:01\nPEPTEST,HLA-A*02:01\n")
    expected_columns = {
        "origin_species",
        "protein_id",
        "left_flank",
        "right_flank",
        "exact_protein_match",
    }
    match_msg = (
        "Input file is missing required columns: "
        f"{format_iterable(expected_columns)}. "
        "Please run 'assign-protein-features' first to generate these columns."
    )
    with pytest.raises(ValueError, match=match_msg):
        generate_decoys(
            hits_path=hits_path,
            # Dummy paths that must be set and must exist
            human_proteome_path=hits_path,
            output_file_path=hits_path,
            num_decoys=2,
            swissprot_db_path=hits_path,
            flank_size=0,
            remove_multiple_matches=False,
            remove_imperfect_matching=False,
            nullify_imperfect_flanks=False,
            with_columns=None,
        )


def test_sample_one_protein() -> None:
    """Test sample_one_protein function."""
    df = pl.DataFrame(
        {
            "protein_id": ["P1;P2", "P3", "P4;P5;P6", None],
            "origin_species": ["H;M", "R", "C;D;E", None],
            "left_flank": ["AAA;CCC", "DDD", "FFF;HHH;III", None],
            "right_flank": ["GGG;TTT", "UUU", "JJJ;KKK;LLL", None],
        }
    )

    # Define the set of all possible valid rows for the (random) output
    possible_outputs_row0 = {("P1", "H", "AAA", "GGG"), ("P2", "M", "CCC", "TTT")}
    possible_outputs_row1 = {("P3", "R", "DDD", "UUU")}
    possible_outputs_row2 = {
        ("P4", "C", "FFF", "JJJ"),
        ("P5", "D", "HHH", "KKK"),
        ("P6", "E", "III", "LLL"),
    }
    possible_outputs_row3 = {(None, None, None, None)}

    all_possible_outputs: list[set[tuple]] = [
        possible_outputs_row0,
        possible_outputs_row1,
        possible_outputs_row2,
        possible_outputs_row3,
    ]

    df_output = sample_one_protein(df)

    assert df_output.shape == (4, 4)
    assert df_output.filter(pl.col("protein_id").str.contains(";")).is_empty()

    # Check for sampling consistency in each row
    for i, row in enumerate(df_output.iter_rows()):
        assert row in all_possible_outputs[i]
