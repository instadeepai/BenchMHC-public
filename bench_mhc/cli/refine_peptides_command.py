"""Module to define the entry point of the command line to refine the peptides."""

from pathlib import Path

import rich_click as click

from bench_mhc.utils.click import arguments


@click.command()
@arguments.peptides_path
@arguments.output_file_path
@arguments.flank_size
@arguments.remove_multiple_matches
@arguments.remove_imperfect_matching
@arguments.nullify_imperfect_flanks
@arguments.with_columns
def refine_peptides(
    peptides_path: Path,
    output_file_path: Path | None,
    flank_size: int,
    remove_multiple_matches: bool,
    remove_imperfect_matching: bool,
    nullify_imperfect_flanks: bool,
    with_columns: set[str] | None,
) -> None:
    """Filter and process dataset.

    This command applies filtering and processing options to a dataset that has already
    been processed with the `assign-protein-features` command. Unlike `generate-decoys`,
    this command does not generate decoys, but only filters and processes the
    existing peptides.

    The process is the following:

    \b
    1. **Load peptides** from a CSV file (must contain `peptide`, `protein_id`,
    `exact_protein_match`, `origin_species`, `left_flank`, `right_flank` columns)
    2. **Apply filtering options**:
       - Remove peptides with imperfect matches (if `--remove_imperfect_matching`)
       - Nullify imperfect flanks (if `--nullify_imperfect_flanks`)
       - Sample one protein per peptide (if `--remove_multiple_matches`)
    3. **Select columns** based on `--with_columns` option
    4. **Output filtered dataset** to a CSV file

    The output CSV file contains the following columns:

    \b
    - `peptide`: The peptide sequence
    - `protein_id`: The protein ID(s) containing the peptide
    - `origin_species`: The species of origin for the peptide
    - `exact_protein_match`: Whether the peptide has an exact match to a source protein
    - `left_flank` and `right_flank`: The flanks (if `--flank_size > 0`)
    - Additional columns specified with `--with_columns` (copied from input)

    !!! Important

        The `assign-protein-features` command must be run first to map the peptides
        to their source proteins, origin species and flanks. This command assumes
        all the peptides have been mapped to their source proteins, origin species and flanks.

    !!! Note

        The filtering options:

        \b
        - `--remove_imperfect_matching`: Removes peptides that don't have an exact match
          to a source protein (i.e., `exact_protein_match == 0`).
        - `--nullify_imperfect_flanks`: Replaces imperfect flanks with padding tokens
          (mutually exclusive with `--remove_imperfect_matching`).
        - `--remove_multiple_matches`: Randomly samples one protein per peptide when
          multiple source proteins are available.
        - `--with_columns`: Selects additional columns to include in the output.
          Use `--with_columns all` to include all available columns.
    """  # noqa: D301
    from bench_mhc.cli.refine_peptides import refine_peptides

    refine_peptides(
        peptides_path=peptides_path,
        output_file_path=output_file_path,
        flank_size=flank_size,
        remove_multiple_matches=remove_multiple_matches,
        remove_imperfect_matching=remove_imperfect_matching,
        nullify_imperfect_flanks=nullify_imperfect_flanks,
        with_columns=with_columns,
    )
