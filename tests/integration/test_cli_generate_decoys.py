"""Integration tests related to bench_mhc/cli/generate_decoys_command.py."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from click.testing import CliRunner
from polars import testing as pl_testing

from bench_mhc.cli.generate_decoys_command import generate_decoys
from bench_mhc.utils.blastp import get_blastp_output
from bench_mhc.utils.kmers import load_human_proteome
from bench_mhc.utils.kmers import load_swissprot_proteome


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
def hits_csv(tmp_path: Path) -> Path:
    """Create a CSV file with peptide hits.

    Returns:
        Path to the created CSV file containing peptide hits with columns:
            - peptide: Peptide sequences.
            - allele: MHC allele names.
            - hit: Binary indicator (1 for hits).

    Note:
        The test data includes:
            - Multiple MHC1 alleles (HLA-A*02:01, HLA-B*07:02, HLA-C*07:01).
            - Peptides with and without cysteine residues.
            - Peptides of different lengths (7-9 amino acids).
            - Split between peptides found in proteome and those requiring SwissProt:
                - Proteome matches: TESTPEP, PEPTEST, TT, EE, CYSTEST, TESTCYS
                - SwissProt pseudo matches: KZAKFAZJ, AAAAAAAAC
            - peptides with hit=0 to filter out.
    """
    df = pl.DataFrame(
        {
            "peptide": [
                # Peptides found in proteome
                "TESTPEP",  # 7-mer in proteome
                "PEPTEST",  # 7-mer in proteome
                "TT",  # 2-mer in proteome
                "EE",  # 2-mer in proteome
                "CYSTEST",  # 7-mer with C in proteome
                "TESTCYS",  # 7-mer with C in proteome
                # Peptides requiring SwissProt
                "KZAKFAZJ",  # 8-mer
                "AAAAAAAAC",  # 9-mer with C, pseudo match in SwissProt
                # Peptide to filter out
                "TOREMOVE",  # 8-mer with hit=0
                "WHATEVER",  # NULL protein_id to remove
            ],
            "allele": [
                # Alleles for human proteome matches
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-B*07:02",
                "HLA-B*07:02",
                "HLA-C*07:01",
                "HLA-C*07:01",
                # Alleles for pseudo matches
                "HLA-A*02:01",
                "HLA-B*07:02",
                # Allele for filtering
                "HLA-A*02:02",
                "HLA-A*02:02",
            ],
            "hit": [1] * 8 + [0] + [1],
            "original_fold": ["fold1"] * 10,
            "protein_id": [
                "P12345|TEST1",  # TESTPEP
                "P67890|TEST2",  # PEPTEST
                "P24680|TEST3",  # TT
                "P13579|TEST4",  # EE
                "P98765|TEST5",  # CYSTEST
                "P54321|TEST6",  # TESTCYS
                "sp|Q9H2K5|TEST10;sp|Q9H2K8|TEST11",  # KZAKFAZJ
                "sp|Q9H2K6|TEST9",  # AAAAAAAAC
                "sp|Q9H2K9|TEST12",  # TOREMOVE
                None,  # NULL protein_id to remove
            ],
            "origin_species": [
                "Homo sapiens",  # TESTPEP
                "Homo sapiens",  # PEPTEST
                "Homo sapiens",  # TT
                "Homo sapiens",  # EE
                "Homo sapiens",  # CYSTEST
                "Homo sapiens",  # TESTCYS
                "Lizard;Lizard",  # KZAKFAZJ
                "Lizard",  # AAAAAAAAC
                "Lizard",  # TOREMOVE
                "Whatever",  # TOREMOVE
            ],
            # Dummy fake wrong flanks
            "left_flank": ["A", "A", "A", "A", "A", "A", "A;C", "A", "A", "A"],
            "right_flank": ["B", "B", "B", "B", "B", "B", "B;D", "B", "B", "B"],
            "exact_protein_match": [1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
            "dummy_column": ["dummy1"] * 10,
        }
    )
    hits_path = tmp_path / "hits.csv"
    df.write_csv(hits_path)

    return hits_path


@pytest.fixture
def proteome_fasta(tmp_path: Path) -> Path:
    """Create a FASTA file with proteome sequences.

    Returns:
        Path to the created FASTA file containing protein sequences with headers:
            - sp|P12345|TEST1: Contains TESTPEP sequence.
            - sp|P67890|TEST2: Contains PEPTEST sequence.
            - sp|P24680|TEST3: Contains TT sequence.
            - sp|P13579|TEST4: Contains EE sequence.
            - sp|P98765|TEST5: Contains CYSTEST sequence.
            - sp|P54321|TEST6: Contains TESTCYS sequence.
    """
    fasta_content = """>P12345|TEST1
TESTPEPABHDEFGHIJKLMNOPQRSTUVWXYZGHJJGFFTHJKKHJHJ
>P67890|TEST2
PEPTESTABHDEFGHIJKLMNOPQRSTUVWXYZ
>P24680|TEST3
TTABCDEFGHIJKLMNOPQRSTUVWXYZ
>P13579|TEST4
EEABCDEFGHIJKLMNOPQRSTUVWXYZ
>P98765|TEST5
CYSTESTABCDEFGHIJKLMNOPQRSTUVWXYZKZAKFJZJ
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
            - sp|Q9H2K6|TEST9: Contains AAAAAAAAC pseudo match sequence.
            - sp|Q9H2K5|TEST10: Contains KZAKFAZJ pseudo match sequence.
            - sp|Q9H2K8|TEST11: Contains KZAKFAZJ pseudo match sequence.
    """
    swissprot_path = tmp_path / "swissprot.fasta"
    with open(swissprot_path, "w") as f:
        f.write(">sp|Q9H2K6|TEST9 Test protein 9 OS=Lizard OX=Q9H2K6\n")
        f.write("AYSATESABADEFAYAAAAAAAAZAZLONGTZZZJWXYLOZZZ\n")
        f.write(">sp|Q9H2K5|TEST10 Test protein 10 OS=Lizard OX=Q9H2K6\n")
        f.write("AYSATESABADEFAYSATESOPKZAZJWXYLOZZZLONGATESABADEFGHIJKLLONGATEKFAZJZLONTESTZ\n")
        f.write(">sp|Q9H2K8|TEST11 Test protein 11 OS=Lizard OX=Q9H2K8\n")
        f.write("LONGATESABADEFGHIJKLLONGATEKFAZJZLONGATZZZ\n")

    return swissprot_path


@pytest.mark.parametrize("with_columns", [None, "original_fold", "all"])
@pytest.mark.parametrize("flank_size", [0, 1, 2])
def test_generate_decoys_command(
    flank_size: int,
    with_columns: str | None,
    hits_csv: Path,
    proteome_fasta: Path,
    swissprot_fasta: Path,
    tmp_path: Path,
) -> None:
    """Test the generate-decoys command with sample data.

    Verify that the command successfully processes hits from the CSV file, matches hits
    to proteins in the proteome, generates the specified number of decoys per hit, and
    outputs the results to the specified CSV file.

    Args:
        flank_size: Size of the flanks to consider.
        with_columns: Columns to copy from hits to decoys.
        hits_csv: Path to CSV file containing peptide hits.
        proteome_fasta: Path to FASTA file with proteome sequences.
        swissprot_fasta: Path to SwissProt FASTA file for BLASTP search.
        tmp_path: Path to temporary directory for test files.
    """
    output_path = tmp_path / "output.csv"
    num_decoys = 10

    runner = CliRunner()
    params = [
        "--hits_path",
        str(hits_csv),
        "--human_proteome_path",
        str(proteome_fasta),
        "--output_file_path",
        str(output_path),
        "--num_decoys",
        str(num_decoys),
        "--swissprot_db_path",
        str(swissprot_fasta),
        "--flank_size",
        str(flank_size),
    ]
    if with_columns:
        params.append("--with_columns")
        params.append(with_columns)

    result = runner.invoke(
        generate_decoys,
        params,
    )
    assert result.exit_code == 0

    with_columns2columns = {
        "all": ["original_fold", "dummy_column"],
        "original_fold": ["original_fold"],
        None: [],
    }

    df_output = pl.read_csv(output_path)
    assert "TOREMOVE" not in df_output["peptide"].unique()

    df_hits = pl.read_csv(hits_csv).filter(
        (pl.col("peptide") != "TOREMOVE") & pl.col("protein_id").is_not_null()
    )

    # Check number of rows (original hits + decoys)
    num_expected_hits = len(df_hits)
    expected_num_rows = num_expected_hits * (1 + num_decoys)
    assert len(df_output) == expected_num_rows

    expected_columns = {
        "peptide",
        "protein_id",
        "hit",
        "allele",
        "origin_species",
        "exact_protein_match",
    }
    expected_columns.update(with_columns2columns[with_columns])
    if flank_size != 0:
        expected_columns.update(["left_flank", "right_flank"])

    # Check columns
    if with_columns:
        assert df_output["original_fold"].to_list() == ["fold1"] * expected_num_rows
        if with_columns == "all":
            assert df_output["dummy_column"].to_list() == ["dummy1"] * expected_num_rows

    assert set(df_output.columns) == expected_columns

    # Check that decoys have the same length distribution as hits
    # Each hit should have exactly num_decoys decoys of the same length
    hits_length_counts = (
        df_output.filter(pl.col("hit") == 1)
        .with_columns(pl.col("peptide").str.len_bytes().alias("length"))
        .group_by("length")
        .agg(pl.len().alias("count"))
        .sort("length")
    )
    decoys_length_counts = (
        df_output.filter(pl.col("hit") == 0)
        .with_columns(pl.col("peptide").str.len_bytes().alias("length"))
        .group_by("length")
        .agg(pl.len().alias("count"))
        .sort("length")
    )
    expected_decoys_length_counts = hits_length_counts.with_columns(
        (pl.col("count") * num_decoys).alias("count")
    )
    pl_testing.assert_frame_equal(decoys_length_counts, expected_decoys_length_counts)

    # Check that the species distribution is correct
    expected_species_distribution_df = pl.DataFrame(
        {
            "origin_species": [
                "Homo sapiens",
                "Homo sapiens",
                "Lizard;Lizard",
                "Lizard",
                "Lizard",
            ],
            "hit": [1, 0, 1, 0, 1],
            "count": [6, 60, 1, 20, 1],
        }
    )
    output_species_distribution_df = df_output.group_by("origin_species", "hit").agg(pl.count())

    pl_testing.assert_frame_equal(
        output_species_distribution_df,
        expected_species_distribution_df,
        check_dtypes=False,
        check_row_order=False,
    )

    df_proteome = (
        load_human_proteome(proteome_fasta)
        .vstack(load_swissprot_proteome(swissprot_fasta))
        .with_columns(sequence=flank_size * "-" + pl.col("sequence") + flank_size * "-")
    )

    if flank_size != 0:
        # Verify that peptides with flanks are correctly embedded in their protein sequences
        assert (
            df_output.filter(pl.col("hit") == 0)
            .join(df_proteome, on="protein_id", how="left")
            .with_columns(
                (pl.col("left_flank") + pl.col("peptide") + pl.col("right_flank")).alias(
                    "peptide_with_flanks"
                )
            )
            .with_columns(
                pl.col("sequence")
                .str.contains(pl.col("peptide_with_flanks"))
                .alias("contains_peptide_with_flanks")
            )
            .get_column("contains_peptide_with_flanks")
            .all()
        )


@pytest.mark.parametrize("remove_multiple_matches", [False, True])
def test_generate_decoys_multiple_proteins(
    remove_multiple_matches: bool,
    hits_csv: Path,
    proteome_fasta: Path,
    swissprot_fasta: Path,
    tmp_path: Path,
) -> None:
    """Test the generate-decoys command to manage samples matching multiple proteins.

    Args:
        remove_multiple_matches: Whether match with multiple source proteins is allowed. If
            not, one protein is randomly sampled with its associated species and flanks.
        hits_csv: Path to CSV file containing peptide hits.
        proteome_fasta: Path to FASTA file with proteome sequences.
        swissprot_fasta: Path to SwissProt FASTA file for BLASTP search.
        tmp_path: Path to temporary directory for test files.
    """
    output_path = tmp_path / "output.csv"
    num_decoys = 10

    runner = CliRunner()
    params = [
        "--hits_path",
        str(hits_csv),
        "--human_proteome_path",
        str(proteome_fasta),
        "--output_file_path",
        str(output_path),
        "--num_decoys",
        str(num_decoys),
        "--swissprot_db_path",
        str(swissprot_fasta),
        "--flank_size",
        "1",
    ]

    if remove_multiple_matches:
        params.append("--remove_multiple_matches")

    result = runner.invoke(
        generate_decoys,
        params,
    )
    assert result.exit_code == 0

    df_output = pl.read_csv(output_path)

    if remove_multiple_matches:
        assert df_output.filter(pl.col("protein_id").str.contains(";")).is_empty()
        assert df_output.filter(pl.col("origin_species").str.contains(";")).is_empty()
        assert df_output.filter(pl.col("left_flank").str.contains(";")).is_empty()
        assert df_output.filter(pl.col("right_flank").str.contains(";")).is_empty()
    else:
        assert not df_output.filter(pl.col("protein_id").str.contains(";")).is_empty()
        assert not df_output.filter(pl.col("origin_species").str.contains(";")).is_empty()
        assert not df_output.filter(pl.col("left_flank").str.contains(";")).is_empty()
        assert not df_output.filter(pl.col("right_flank").str.contains(";")).is_empty()


@pytest.mark.parametrize("remove_imperfect_matching", [False, True])
@pytest.mark.parametrize("nullify_imperfect_flanks", [False, True])
def test_generate_decoys_imperfect_matching(
    remove_imperfect_matching: bool,
    nullify_imperfect_flanks: bool,
    hits_csv: Path,
    proteome_fasta: Path,
    swissprot_fasta: Path,
    tmp_path: Path,
) -> None:
    """Test the generate-decoys command with imperfect matching.

    Args:
        remove_imperfect_matching: Whether we remove peptides with imperfect match.
        nullify_imperfect_flanks: Whether we replace imperfect flanks with padding token.
        hits_csv: Path to CSV file containing peptide hits.
        proteome_fasta: Path to FASTA file with proteome sequences.
        swissprot_fasta: Path to SwissProt FASTA file for BLASTP search.
        tmp_path: Path to temporary directory for test files.
    """
    output_path = tmp_path / "output.csv"
    num_decoys = 10

    runner = CliRunner()
    params = [
        "--hits_path",
        str(hits_csv),
        "--human_proteome_path",
        str(proteome_fasta),
        "--output_file_path",
        str(output_path),
        "--num_decoys",
        str(num_decoys),
        "--swissprot_db_path",
        str(swissprot_fasta),
        "--flank_size",
        "1",
    ]

    if remove_imperfect_matching:
        params.append("--remove_imperfect_matching")
    if nullify_imperfect_flanks:
        params.append("--nullify_imperfect_flanks")

    if remove_imperfect_matching and nullify_imperfect_flanks:
        expected_msg = (
            "Flags 'remove_imperfect_matching' and 'nullify_imperfect_flanks' are mutually "
            "exclusive, you should pick only one."
        )
        with pytest.raises(ValueError, match=expected_msg):
            runner.invoke(generate_decoys, params, catch_exceptions=False)

        return

    result = runner.invoke(
        generate_decoys,
        params,
    )
    assert result.exit_code == 0

    df_output = pl.read_csv(output_path)

    if remove_imperfect_matching:
        assert "KZAKFAZJ" not in df_output["peptide"].unique()
        assert "AAAAAAAAC" not in df_output["peptide"].unique()
    else:
        assert "KZAKFAZJ" in df_output["peptide"].unique()
        assert "AAAAAAAAC" in df_output["peptide"].unique()

    expected_df_nullify_flanks = pl.DataFrame(
        {
            "peptide": [
                "TESTPEP",
                "PEPTEST",
                "TT",
                "EE",
                "CYSTEST",
                "TESTCYS",
                "KZAKFAZJ",
                "AAAAAAAAC",
            ],
            "left_flank": ["A", "A", "A", "A", "A", "A", "-;-", "-"],
            "right_flank": ["B", "B", "B", "B", "B", "B", "-;-", "-"],
        }
    )

    if nullify_imperfect_flanks and not remove_imperfect_matching:
        pl_testing.assert_frame_equal(
            df_output.filter(pl.col("hit") == 1).select(["peptide", "left_flank", "right_flank"]),
            expected_df_nullify_flanks,
        )
