"""Module to define the command line to refine the peptides."""

from pathlib import Path

import polars as pl

from bench_mhc.cli.generate_decoys import maybe_remove_imperfect_matching
from bench_mhc.cli.generate_decoys import sample_one_protein
from bench_mhc.cli.generate_decoys import validate_and_select_columns
from bench_mhc.utils.logging import system

log = system.get(__name__)


def refine_peptides(
    peptides_path: Path,
    output_file_path: Path | None,
    flank_size: int,
    remove_multiple_matches: bool,
    remove_imperfect_matching: bool,
    nullify_imperfect_flanks: bool,
    with_columns: set[str] | None,
) -> None:
    """Filter and process peptides dataset.

    This function filters and processes a dataset of peptides that has already
    been processed with the `assign-protein-features` command. It applies filtering
    options (remove imperfect matches, nullify imperfect flanks, sample one protein
    per peptide) and selects columns based on the provided options.

    Refer to bench_mhc.cli.refine_peptides_command.py::refine_peptides for complete
    documentation of the command-line interface.

    Args:
        peptides_path: Path to CSV file containing peptides with 'peptide',
            'protein_id', 'exact_protein_match', 'origin_species', 'left_flank' and
            'right_flank' columns.
        output_file_path: Path to output CSV file with filtered peptides. If not
            provided, the input file will be overwritten.
        flank_size: Size of the flanks to consider. If 0, flank columns are not
            included in the output even if requested via with_columns.
        remove_multiple_matches: Whether match with multiple source proteins is allowed. If
            not, one protein is randomly sampled with its associated species and flanks.
        remove_imperfect_matching: Whether we remove peptides with imperfect match.
        nullify_imperfect_flanks: Whether we replace imperfect flanks with padding token.
        with_columns: Optional set of column names to copy from dataset. If 'all' is the
            unique value, all columns are returned.

    Raises:
        ValueError:
            - If 'remove_imperfect_matching' and 'nullify_imperfect_flanks' are both provided.
            - If required columns are missing from the input CSV.
    """
    if remove_imperfect_matching and nullify_imperfect_flanks:
        raise ValueError(
            "Flags 'remove_imperfect_matching' and 'nullify_imperfect_flanks' are mutually "
            "exclusive, you should pick only one."
        )

    log.info("Loading dataset from CSV.")

    lf_peptides = pl.scan_csv(peptides_path)
    available_columns = set(lf_peptides.collect_schema().names())

    with_columns, columns_to_select = validate_and_select_columns(
        available_columns, with_columns, flank_size, require_allele=False
    )

    df_peptides = lf_peptides.select(columns_to_select).collect(engine="streaming")

    if (unmatched_count := df_peptides["protein_id"].is_null().sum()) > 0:
        log.warning(
            f"Found {unmatched_count} peptide(s) with no associated protein ID. "
            "Those peptides will be ignored."
        )
    df_peptides = df_peptides.filter(~pl.col("protein_id").is_null())

    df_peptides = maybe_remove_imperfect_matching(
        df_hits=df_peptides,
        remove_imperfect_matching=remove_imperfect_matching,
        nullify_imperfect_flanks=nullify_imperfect_flanks,
        flank_size=flank_size,
    )

    log.info(f"Loaded {len(df_peptides)} peptides.")

    if remove_multiple_matches:
        df_peptides = sample_one_protein(df_peptides)
        log.info("Applied remove_multiple_matches: sampled one protein per peptide.")

    output_file_path = output_file_path or peptides_path
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    df_peptides.write_csv(output_file_path)

    log.info(f"Refined dataset saved to {output_file_path}.")
