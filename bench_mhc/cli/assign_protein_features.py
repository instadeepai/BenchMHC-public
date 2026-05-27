"""Module to define the command line to assign protein-related features."""

import tempfile
from pathlib import Path
from typing import cast

import polars as pl

from bench_mhc.constants import PAD_TOKEN
from bench_mhc.utils.blastp import build_blast_db
from bench_mhc.utils.blastp import get_blastp_output
from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.kmers import generate_kmers_from_proteome
from bench_mhc.utils.kmers import load_human_proteome
from bench_mhc.utils.kmers import load_swissprot_proteome
from bench_mhc.utils.logging import system
from bench_mhc.utils.stats import SourceProteinMatchingStats

log = system.get(__name__)

DEFAULT_SPECIES = "Homo sapiens"


def assign_protein_features(
    peptides_path: Path,
    flank_size: int,
    human_proteome_path: Path,
    swissprot_db_path: Path,
    output_file_path: Path,
    with_columns: set[str] | None,
) -> None:
    """Assign source protein(s) and related features to each peptide using the reference proteome.

    Refer to bench_mhc.cli.assign_protein_features_command.py::assign_protein_features for the
    complete documentation.

    Args:
        peptides_path: Path to CSV file containing peptides with a 'peptide' column.
        flank_size: Size of the flanks to be assigned. It can include padding tokens if needed.
        human_proteome_path: Path to proteome FASTA file.
        swissprot_db_path: Path to SwissProt FASTA file.
        output_file_path: Path to output CSV file with peptides and source proteins.
        with_columns: Optional set of column names to copy from peptides to output.

    Raises:
        ValueError:
            - If input CSV does not contain a 'peptide' column.
            - If any column in `with_columns` is not present in the input CSV.
    """
    log.info("Loading peptides from CSV...")

    lf_peptides = pl.scan_csv(peptides_path)
    available_columns = lf_peptides.collect_schema().names()

    if "peptide" not in available_columns:
        raise ValueError("Input CSV must contain a 'peptide' column.")

    # If 'all' is the unique value, all columns are returned
    should_select_all_columns = (
        with_columns and len(with_columns) == 1 and list(with_columns)[0] == "all"
    )
    if should_select_all_columns:
        with_columns = set(available_columns) - {"peptide"}

    # Check if with_columns are present in the CSV
    if with_columns:
        missing_columns = with_columns - set(available_columns)
        if missing_columns:
            raise ValueError(
                "The following columns specified in --with_columns "
                f"are not present in the input CSV: {format_iterable(missing_columns)}"
            )
    # Select required columns plus any additional columns specified
    columns_to_select = ["peptide"]
    if with_columns:
        columns_to_select.extend(list(with_columns))

    df_peptides = lf_peptides.select(columns_to_select).collect(engine="streaming")

    log.info(f"Loaded {len(df_peptides)} peptides.")

    log.info(f"Loading human proteome sequences from {str(human_proteome_path)}...")
    df_human_proteome = load_human_proteome(human_proteome_path)
    log.info(f"Loaded {len(df_human_proteome)} proteome sequences.")
    log.info("")

    lfs = []

    stats = SourceProteinMatchingStats(len(df_peptides))

    df_proteome_swissprot = None
    protein_ids_to_keep = set(df_human_proteome["protein_id"].unique().to_list())

    with tempfile.TemporaryDirectory() as blast_db_dir_:
        blast_db_dir = Path(blast_db_dir_)

        for peptide_length_tuple, df_peptides_length in df_peptides.group_by(
            pl.col("peptide").str.len_bytes()
        ):
            # Required for mypy type checking
            peptide_length = cast(int, peptide_length_tuple[0])
            log.info(f"Processing peptide with length {peptide_length}...")

            lf_kmers = generate_kmers_from_proteome(
                df_proteome=df_human_proteome,
                kmer_length=peptide_length,
                flank_size=flank_size,
            )

            lf_peptides_unmatched, lf_peptides_matched, num_peptides_matched = (
                assign_exact_protein_features_per_length(
                    lf_peptides=df_peptides_length.lazy(),
                    lf_kmers=lf_kmers,
                    with_columns=with_columns,
                )
            )

            log.info(f"Matched exactly {num_peptides_matched} peptides to human proteome.")

            lfs.append(lf_peptides_matched)
            stats.num_items_processed += num_peptides_matched

            num_peptides_unmatched = lf_peptides_unmatched.select(pl.len()).collect().item()
            num_peptides_unmatched_after_blastp = 0

            # Handle peptides that do not have an exact match in the human proteome
            if num_peptides_unmatched > 0:
                log.info(
                    "Number of peptides that don't match exactly to the human proteome: "
                    f"{num_peptides_unmatched}. "
                    "We will run blastp against the human proteome and SwissProt."
                )

                _maybe_build_blast_databases(swissprot_db_path, human_proteome_path, blast_db_dir)

                if df_proteome_swissprot is None:
                    log.info(f"Loading SwissProt sequences from {str(swissprot_db_path)}...")
                    df_proteome_swissprot = load_swissprot_proteome(swissprot_db_path)
                    log.info(f"Loaded {len(df_proteome_swissprot)} SwissProt sequences.")
                    protein_ids_to_keep.update(
                        df_proteome_swissprot["protein_id"].unique().to_list()
                    )

                df_combined_proteome = df_human_proteome.vstack(df_proteome_swissprot).unique(
                    subset=["protein_id"], keep="first"
                )
                log.info(
                    f"Created a combined database of {len(df_combined_proteome)} protein sequences."
                )

                lf_blastp_out = _run_blastp_human_swissprot(
                    lf_peptides=lf_peptides_unmatched,
                    blast_db_dir=blast_db_dir,
                    protein_ids_to_keep=protein_ids_to_keep,
                )

                lf_peptides_after_blastp, num_peptides_imperfectly_matched = (
                    assign_imperfect_protein_features_per_length(
                        lf_peptides=lf_peptides_unmatched,
                        lf_blastp_out=lf_blastp_out,
                        lf_proteome=df_combined_proteome.lazy(),
                        flank_size=flank_size,
                        with_columns=with_columns,
                    )
                )

                num_peptides_unmatched_after_blastp = (
                    lf_peptides_after_blastp.filter(pl.col("protein_id").is_null())
                    .select(pl.len())
                    .collect()
                    .item()
                )

                log.info(f"Matched {num_peptides_imperfectly_matched} peptides with blastp.")

                lfs.append(lf_peptides_after_blastp)
                stats.num_items_processed += num_peptides_imperfectly_matched

            stats.update_kmer_stats(
                kmer_length=peptide_length,
                peptides_in_kmer=len(df_peptides_length),
                matched_peptides=num_peptides_matched,
                unmatched_peptides=num_peptides_unmatched,
                unmatched_after_blastp=num_peptides_unmatched_after_blastp,
            )

            stats.update_global_stats(
                unmatched_human_proteome_peptides=num_peptides_unmatched,
                unmatched_swissprot_human_proteome_peptides=num_peptides_unmatched_after_blastp,
                num_peptides_processed=stats.num_items_processed,
            )

            stats.display_statistics(kmer_length=peptide_length)
            log.info("")

    output_file_path = output_file_path or peptides_path

    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    stats.display_statistics(kmer_length=None)
    stats.save_to_file(output_file_path.with_name(f"{output_file_path.name}_matching_stats.json"))

    pl.concat(lfs).collect(engine="streaming").write_csv(output_file_path)

    log.info(f"Peptides with source protein features saved to {output_file_path}.")


def assign_exact_protein_features_per_length(
    lf_peptides: pl.LazyFrame,
    lf_kmers: pl.LazyFrame,
    with_columns: set[str] | None,
) -> tuple[pl.LazyFrame, pl.LazyFrame, int]:
    """Assign exact protein-related features to peptides, using the human proteome.

    Protein-related features include the protein ID, the species and the flanks. A peptide can be
    matched to multiple source proteins.

    Args:
        lf_peptides: Peptide lazyframe for a specific peptide length and with a column 'peptide'.
        lf_kmers: Lazyframe with k-mers extracted from the proteome. Columns include:
            - `peptide`: k-mer sequence
            - `protein_id`: Source protein ID
            - `with_c`: (Optional) Boolean indicating if the peptide contains cysteine
            - `left_flank` and `right_flank`: (Optional) The peptide's flanks
                for a given source protein
        with_columns: Optional set of column names to copy from peptides dataframe to output.

    Returns:
        lf_peptides_unmatched: Lazyframe with unmatched peptides.
        lf_peptides_matched: Lazyframe with the matched peptides. Columns include:
            - peptide: Peptide sequence
            - protein_id: Protein ID(s) containing the peptide
            - origin_species: Origin species of the protein
            - `left_flank` and `right_flank`: The peptide's flanks for a given source protein
            - `exact_protein_match`: Whether the peptide has an exact match (True here)
            - Additional columns specified with `--with_columns` (copied from input CSV)
        num_peptides_matched: Number of peptides matched to a source protein.
    """
    peptides = lf_peptides.select("peptide").unique().collect().get_column("peptide").to_list()

    if not {"left_flank", "right_flank"}.issubset(lf_kmers.collect_schema().names()):
        lf_kmers = lf_kmers.with_columns(
            pl.lit(None).alias("left_flank"), pl.lit(None).alias("right_flank")
        )

    lf_kmers_w_flanks = (
        lf_kmers.filter(pl.col("peptide").is_in(peptides))
        # Add the default species
        .with_columns(pl.lit(DEFAULT_SPECIES).alias("origin_species"))
        # Sort to ensure the order in the list of proteins is reproducible
        .unique()
        .sort(["peptide", "protein_id"])
        .group_by("peptide")
        .agg(
            pl.col("protein_id").str.join(";"),
            pl.col("origin_species").str.join(";"),
            pl.col("left_flank").str.join(";", ignore_nulls=False),
            pl.col("right_flank").str.join(";", ignore_nulls=False),
        )
    )

    columns_to_keep = {"peptide", "protein_id", "origin_species", "left_flank", "right_flank"}
    if with_columns:
        columns_to_keep.update(with_columns)

    lf_peptides_w_flanks = lf_peptides.join(
        lf_kmers_w_flanks, on="peptide", how="left", maintain_order="left"
    )

    lf_peptides_unmatched = (
        lf_peptides_w_flanks.filter(pl.col("protein_id").is_null())
        .select(columns_to_keep)
        .with_columns(pl.lit(0).alias("exact_protein_match"))
    )

    lf_peptides_matched = (
        lf_peptides_w_flanks.filter(pl.col("protein_id").is_not_null())
        .select(columns_to_keep)
        .with_columns(pl.lit(1).alias("exact_protein_match"))
    )

    num_peptides_matched = lf_peptides_matched.select(pl.len()).collect().item()

    return lf_peptides_unmatched, lf_peptides_matched, num_peptides_matched


def assign_imperfect_protein_features_per_length(
    lf_peptides: pl.LazyFrame,
    lf_blastp_out: pl.LazyFrame,
    lf_proteome: pl.LazyFrame,
    flank_size: int,
    with_columns: set[str] | None,
) -> tuple[pl.LazyFrame, int]:
    """Assign imperfect protein-related features to peptides, running blastp.

    We run blastp against the human proteome and SwissProt.

    Protein related features include the protein ID, the species and the flanks. A peptide can be
    matched to multiple source proteins.

    Args:
        lf_peptides: Peptide lazyframe with a column 'peptide'.
        lf_blastp_out: Lazyframe with the blastp output and columns:
            - `peptide`: Peptide sequence
            - `protein_id`: Protein ID(s) containing the peptide
            - `origin_species`: Origin species of the protein
            - `match_start` and `match_end`: Starting and ending position of the alignment
        lf_proteome: Proteome lazyframe with the columns `protein_id` and `sequence`.
        flank_size: Flank size to extract.
        with_columns: Optional set of column names to copy from peptides dataframe to output.

    Returns:
        lf_peptides_w_protein_features: Output lazyframe. Unmatched peptides after blastp keep
            protein-related features with null values. Columns include:
                - peptide: Peptide sequence
                - protein_id: Protein ID(s) containing the peptide
                - origin_species: Origin species of the protein
                - `left_flank` and `right_flank`: Peptide's flanks for a given source protein
                - `exact_protein_match`: Whether the peptide has an exact match (False here)
                - Additional columns specified with `--with_columns` (copied from input CSV)
        num_peptides_matched: Number of peptides imperfectly matched to a source protein.
    """
    null_flank = PAD_TOKEN * flank_size

    protein_ids = (
        lf_blastp_out.select("protein_id").unique().collect().get_column("protein_id").to_list()
    )

    lf_blastp_w_flanks = (
        lf_blastp_out.join(
            lf_proteome.filter(pl.col("protein_id").is_in(protein_ids)),
            on="protein_id",
            how="left",
            maintain_order="left",
        )
        # Pad directly the source protein to manage peptides on the edges
        .with_columns(
            pl.concat_str([pl.lit(null_flank), pl.col("sequence"), pl.lit(null_flank)]).alias(
                "protein_sequence"
            ),
            (pl.col("match_start") + flank_size).alias("match_start"),
            (pl.col("match_end") + flank_size).alias("match_end"),
        )
        # Extract flanks
        .with_columns(
            pl.col("protein_sequence")
            .str.slice((pl.col("match_start") - flank_size), flank_size)
            .alias("left_flank"),
            pl.col("protein_sequence")
            .str.slice(pl.col("match_end"), flank_size)
            .alias("right_flank"),
        )
        # Sort to ensure the order in the list of proteins is reproducible
        .unique()
        .sort(["peptide", "protein_id"])
        .group_by("peptide")
        .agg(
            pl.col("protein_id").str.join(";"),
            pl.col("origin_species").str.join(";"),
            pl.col("left_flank").str.join(";", ignore_nulls=False),
            pl.col("right_flank").str.join(";", ignore_nulls=False),
        )
        .select(["peptide", "protein_id", "origin_species", "left_flank", "right_flank"])
    )

    columns_to_keep = {"peptide", "protein_id", "origin_species", "left_flank", "right_flank"}
    if with_columns:
        columns_to_keep.update(with_columns)

    lf_peptides_w_protein_features = (
        lf_peptides.drop(["protein_id", "origin_species", "left_flank", "right_flank"])
        .join(lf_blastp_w_flanks, on="peptide", how="left", maintain_order="left")
        .select(columns_to_keep)
        .with_columns(pl.lit(0).alias("exact_protein_match"))
    )

    num_peptides_matched = (
        lf_peptides_w_protein_features.filter(pl.col("protein_id").is_not_null())
        .select(pl.len())
        .collect()
        .item()
    )

    return lf_peptides_w_protein_features, num_peptides_matched


def _maybe_build_blast_databases(
    swissprot_db_path: Path,
    human_proteome_path: Path,
    blast_db_dir: Path,
) -> None:
    """Create BLAST databases for SwissProt and human proteome if they don't exist.

    Args:
        swissprot_db_path: Path to the SwissProt FASTA file.
        human_proteome_path: Path to the human proteome FASTA file.
        blast_db_dir: Directory path where BLAST databases will be created.
    """
    swissprot_blast_db_path = blast_db_dir / "swissprot"
    hg38_blast_db_path = blast_db_dir / "hg38"

    if not swissprot_blast_db_path.exists():
        build_blast_db(
            blast_db_name="swissprot",
            blast_db_output_path=str(swissprot_blast_db_path),
            proteome_path=swissprot_db_path,
        )

    if not hg38_blast_db_path.exists():
        build_blast_db(
            blast_db_name="hg38",
            blast_db_output_path=str(hg38_blast_db_path),
            proteome_path=human_proteome_path,
        )


def _run_blastp_human_swissprot(
    lf_peptides: pl.LazyFrame, blast_db_dir: Path, protein_ids_to_keep: set[str]
) -> pl.LazyFrame:
    """Run blastp and format output against human proteome and swissprot.

    Args:
        lf_peptides: Peptide lazyframe with the column `peptide`.
        blast_db_dir: Path to where the blast database is built.
        protein_ids_to_keep: Set of protein IDs to keep.

    Returns:
        Lazyframe with blastp results. Columns includes:
            - `peptide`: Peptide sequence
            - `protein_id`: Protein ID(s) containing the peptide
            - `origin_species`: Origin species of the protein
            - `match_start` and `match_end`: Starting and ending position of the alignment
    """
    peptides = lf_peptides.select("peptide").unique().collect().get_column("peptide").to_list()

    # Run BLASTp search against SwissProt
    df_blastp_swissprot_out = get_blastp_output(
        peptides=peptides,
        blast_db_name="swissprot",
        blast_db_path=str(blast_db_dir),
        num_threads=1,
    ).select(
        pl.col("qseqid").alias("peptide"),
        pl.col("sseqid").alias("protein_id"),
        pl.col("stitle")
        .str.split_exact("OS=", 1)
        .struct.field("field_1")
        .str.split_exact(" OX=", 1)
        .struct.field("field_0")
        .alias("origin_species"),
        pl.col("bitscore"),
        (pl.col("sstart") - 1).alias("match_start"),
        pl.col("send").alias("match_end"),
    )

    # Run BLASTp search against human proteome
    df_blastp_hg38_out = get_blastp_output(
        peptides=peptides,
        blast_db_name="hg38",
        blast_db_path=str(blast_db_dir),
        num_threads=1,
    ).select(
        pl.col("qseqid").alias("peptide"),
        pl.col("sseqid").alias("protein_id"),
        pl.lit(DEFAULT_SPECIES).alias("origin_species"),
        pl.col("bitscore"),
        (pl.col("sstart") - 1).alias("match_start"),
        pl.col("send").alias("match_end"),
    )

    return (
        df_blastp_swissprot_out.vstack(df_blastp_hg38_out)
        # This removes match to proteins that contains a stop codon (*).
        .filter(pl.col("protein_id").is_in(protein_ids_to_keep))
        .lazy()
        # Keep the top 1 match per peptide
        # Highest bitscore is the best match
        # If several ties, keep all of them since 'min' method
        # will assign the same smallest rank to all the ties
        .filter(pl.col("bitscore").rank("min", descending=True).over("peptide") == 1)
        .select(["peptide", "protein_id", "origin_species", "match_start", "match_end"])
    )
