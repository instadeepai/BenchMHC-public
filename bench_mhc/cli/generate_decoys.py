"""Module to define the command line to generate decoys."""

import gc
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import polars as pl

from bench_mhc.constants import PAD_TOKEN
from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.kmers import generate_kmers_from_proteome
from bench_mhc.utils.kmers import generate_kmers_sampling_pool
from bench_mhc.utils.kmers import load_human_proteome
from bench_mhc.utils.kmers import load_swissprot_proteome
from bench_mhc.utils.logging import system

log = system.get(__name__)

DEFAULT_SPECIES = "Homo sapiens"


def generate_decoys(
    hits_path: Path,
    human_proteome_path: Path,
    output_file_path: Path,
    num_decoys: int,
    swissprot_db_path: Path,
    flank_size: int,
    remove_multiple_matches: bool,
    remove_imperfect_matching: bool,
    nullify_imperfect_flanks: bool,
    with_columns: set[str] | None,
) -> None:
    """Generate decoys from hits using proteome sequences.

    Refer to bench_mhc.cli.generate_decoys_command.py::generate_decoys for
    the complete documentation.

    Args:
        hits_path: Path to CSV file containing hits with 'peptide', 'protein_id', 'origin_species'
            and 'allele' columns.
        human_proteome_path: Path to proteome FASTA file.
        output_file_path: Path to output CSV file with hits and decoys.
        num_decoys: Number of decoys to generate per hit.
        swissprot_db_path: Path to SwissProt FASTA file.
        flank_size: Size of the flanks to consider.
        remove_multiple_matches: Whether match with multiple source proteins is allowed. If
            not, one protein is randomly sampled with its associated species and flanks.
        remove_imperfect_matching: Whether we remove peptides with imperfect match.
        nullify_imperfect_flanks: Whether we replace imperfect flanks with padding token.
        with_columns: Optional set of column names to copy from hits to decoys. If 'all' is the
            unique value, all columns are returned.

    Raises:
        ValueError: If 'remove_imperfect_matching' and 'nullify_imperfect_flanks' are both provided.
    """
    log.info("Loading hits from CSV...")

    lf_hits = pl.scan_csv(hits_path)
    available_columns = set(lf_hits.collect_schema().names())

    with_columns, columns_to_select = validate_and_select_columns(
        available_columns, with_columns, flank_size
    )

    # If 'hit' column is in original dataset, filter to only hit=1
    if "hit" in available_columns:
        lf_hits = lf_hits.filter(pl.col("hit") == 1)
        log.info("Filtering to only rows with hit=1.")

    if remove_imperfect_matching and nullify_imperfect_flanks:
        raise ValueError(
            "Flags 'remove_imperfect_matching' and 'nullify_imperfect_flanks' are mutually "
            "exclusive, you should pick only one."
        )

    df_hits = lf_hits.select(columns_to_select).collect(engine="streaming")

    if (unmatched_count := df_hits["protein_id"].is_null().sum()) > 0:
        log.warning(
            f"Found {unmatched_count} peptide(s) with no associated protein ID. "
            "Those peptides will be ignored."
        )
    df_hits = df_hits.filter(~pl.col("protein_id").is_null())

    df_hits = maybe_remove_imperfect_matching(
        df_hits=df_hits,
        remove_imperfect_matching=remove_imperfect_matching,
        nullify_imperfect_flanks=nullify_imperfect_flanks,
        flank_size=flank_size,
    )

    allele2peptide_to_reject = {
        row["allele"]: set(row["peptide"])
        for row in df_hits.group_by("allele").agg(pl.col("peptide").unique()).iter_rows(named=True)
    }

    log.info(f"Loaded {len(df_hits)} hits.")

    log.info(f"Loading human proteome sequences from {str(human_proteome_path)}...")
    df_human_proteome = load_human_proteome(human_proteome_path)
    log.info(f"Loaded {len(df_human_proteome)} proteome sequences.")

    log.info(f"Loading SwissProt sequences from {str(swissprot_db_path)}...")

    unique_protein_ids = df_hits["protein_id"].str.split(";").explode().unique()

    df_proteome_swissprot = load_swissprot_proteome(swissprot_db_path).filter(
        pl.col("protein_id").is_in(unique_protein_ids)
    )
    log.info(f"Loaded {len(df_proteome_swissprot)} SwissProt sequences.")

    df_combined_proteome = (
        df_human_proteome.select("protein_id", "sequence")
        .filter(pl.col("protein_id").is_in(unique_protein_ids))
        .vstack(df_proteome_swissprot)
        .unique(subset=["protein_id"], keep="first")
    )
    log.info(f"Created a combined database of {len(df_combined_proteome)} protein sequences.")

    num_hits_processed = 0

    with TemporaryDirectory() as tmp_dir_:
        tmp_dir = Path(tmp_dir_)
        columns_to_select.append("hit")
        for peptide_length_tuple, df_hits_length in df_hits.group_by(
            pl.col("peptide").str.len_bytes()
        ):
            # Required for mypy checking
            peptide_length = cast(int, peptide_length_tuple[0])

            log.info(f"Processing {len(df_hits_length)} hits for {peptide_length}-mers...")

            unique_protein_ids_length = (
                df_hits_length["protein_id"].str.split(";").explode().unique()
            )

            lf_kmers = generate_kmers_from_proteome(
                df_combined_proteome.filter(pl.col("protein_id").is_in(unique_protein_ids_length)),
                peptide_length,
                include_cysteine_info=True,
                flank_size=flank_size,
            )

            with_c2protein_id2peptides = generate_kmers_sampling_pool(lf_kmers)

            log.info(
                f"Generated sampling pool for {peptide_length}-mers from human + SwissProt "
                "proteomes."
            )
            dfs_decoys, num_hits_processed = generate_decoys_df(
                df_hits=df_hits_length,
                allele2peptide_to_reject=allele2peptide_to_reject,
                with_c2protein_id2peptides=with_c2protein_id2peptides,
                num_decoys=num_decoys,
                num_hits_processed=num_hits_processed,
                with_columns=with_columns,
            )

            df_decoys = pl.concat(dfs_decoys).with_columns(pl.lit(1).alias("exact_protein_match"))

            # Add the flanks to the decoys dataframes
            if flank_size != 0:
                df_decoys = (
                    # Since `lf_kmers` has been build with flanks,
                    # they will be added at this step for the decoys
                    df_decoys.lazy()
                    .join(
                        lf_kmers.unique(subset=["peptide", "protein_id"]),
                        on=["peptide", "protein_id"],
                        how="left",
                    )
                    .select(columns_to_select)
                    .collect(engine="streaming")
                )

            df_decoys.write_csv(tmp_dir / f"decoys_kmers={peptide_length}.csv")

            del df_decoys
            del dfs_decoys
            del lf_kmers
            del with_c2protein_id2peptides

            gc.collect()

        if remove_multiple_matches:
            df_hits = sample_one_protein(df_hits)

        output_file_path = output_file_path or hits_path
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        pl.concat(
            [
                pl.scan_csv(tmp_file).select(columns_to_select)
                for tmp_file in tmp_dir.glob("decoys_kmers=*.csv")
            ]
            + [df_hits.with_columns(pl.lit(1).alias("hit")).select(columns_to_select).lazy()],
            how="vertical_relaxed",
        ).sink_csv(output_file_path)

    log.info(f"Generated decoys saved to {output_file_path}.")


def get_decoys_sample_for_hit(
    peptide: str,
    protein_ids: list[str],
    with_c2protein_id2peptides: dict[bool, dict[str, set[str]]],
    num_decoys: int,
    origin_species: list[str],
    peptide_to_reject: set[str],
) -> pl.DataFrame:
    """Get a sample of decoys for a hit peptide.

    This function generates decoy peptides for a given hit peptide by sampling from
    peptides found in the same proteins. The decoys are selected based on whether
    they contain cysteine (C) residues, matching the cysteine status of the original
    hit peptide.

    Args:
        peptide: The hit peptide sequence for which to generate decoys.
        protein_ids: List of protein IDs that contain the hit peptide.
        with_c2protein_id2peptides: Dictionary mapping cysteine presence (True/False)
            to a nested dictionary of protein IDs and their associated peptides.
        num_decoys: The number of decoy peptides to sample.
        origin_species: The origin species of the hit peptide.
        peptide_to_reject: Set of peptides to reject.

    Returns:
        A DataFrame containing the original hit peptide and its decoys, with columns:
            - peptide: The peptide sequence
            - protein_id: The protein ID(s) containing the peptide
            - origin_species: The origin species of the peptide
    """
    with_c = "C" in peptide

    try:
        candidate_peptides = [
            f"{peptide_}@{protein_id}@{origin_species_}"
            for origin_species_, protein_id in zip(origin_species, protein_ids, strict=False)
            for peptide_ in (with_c2protein_id2peptides[with_c][protein_id] - peptide_to_reject)
        ]
    # For some peptides containing C that are not exactly matched to a protein
    # it can happen that the sequence of the source protein does not contain any C.
    # In this case, we use non-C containing peptides to sample from for the decoys.
    except KeyError:
        candidate_peptides = [
            f"{peptide_}@{protein_id}@{origin_species_}"
            for origin_species_, protein_id in zip(origin_species, protein_ids, strict=False)
            for peptide_ in (with_c2protein_id2peptides[not with_c][protein_id] - peptide_to_reject)
        ]
    decoys = np.random.choice(candidate_peptides, num_decoys, replace=True)

    return pl.DataFrame(
        {
            "peptide": [decoy.split("@")[0] for decoy in decoys],
            "protein_id": [decoy.split("@")[1] for decoy in decoys],
            "origin_species": [decoy.split("@")[2] for decoy in decoys],
        }
    )


def generate_decoys_df(
    df_hits: pl.DataFrame,
    allele2peptide_to_reject: dict[str, set[str]],
    with_c2protein_id2peptides: dict[bool, dict[str, set[str]]],
    num_decoys: int,
    num_hits_processed: int,
    with_columns: set[str] | None,
) -> tuple[list[pl.DataFrame], int]:
    """Generate hit-decoy DataFrames from hits that match in the proteome.

    This function processes hit peptides that have matches in the proteome and generates
    decoy peptides for each hit.

    Args:
        df_hits: DataFrame containing hit peptides with required columns:
            - peptide: The peptide sequence
            - allele: The MHC allele
            - protein_id: The protein ID(s) containing the peptide
            - origin_species: The origin species of the peptide
        allele2peptide_to_reject: Dictionary mapping alleles to peptides to reject.
        with_c2protein_id2peptides: Dictionary mapping cysteine presence (True/False)
            to a nested dictionary of protein IDs and their associated peptides.
        num_decoys: The number of decoy peptides to generate for each hit.
        num_hits_processed: The current count of processed hits, used for progress tracking.
        with_columns: Optional set of column names to copy from hits to decoys.

    Returns:
        - dfs: A list of DataFrames containing the original hit peptide and its decoys,
            with columns:
            - peptide: The peptide sequence
            - protein_id: The protein ID(s) containing the peptide
            - origin_species: The origin species of the peptide
            - hit: Binary indicator (0 for decoys)
            - allele: The MHC allele
            - Additional columns specified in with_columns (copied from hits to decoys)
        - num_hits_processed: The current count of processed hits, used for progress tracking.
    """
    hit_column = pl.Series(name="hit", values=[0] * (num_decoys))

    dfs_decoys = []

    for row in df_hits.iter_rows(named=True):
        df_decoys = get_decoys_sample_for_hit(
            with_c2protein_id2peptides=with_c2protein_id2peptides,
            num_decoys=num_decoys,
            peptide_to_reject=allele2peptide_to_reject[row["allele"]],
            peptide=row["peptide"],
            protein_ids=row["protein_id"].split(";"),
            origin_species=row["origin_species"].split(";"),
        ).with_columns(
            pl.lit(row["allele"]).alias("allele"),
            hit_column,
        )

        # Add additional columns from the hit to the decoys
        if with_columns:
            for col_name in with_columns:
                if col_name in row:
                    df_decoys = df_decoys.with_columns(pl.lit(row[col_name]).alias(col_name))

        dfs_decoys.append(df_decoys)

        num_hits_processed += 1

        if num_hits_processed % 100 == 0:  # pragma: no cover
            log.info(f"Processed {num_hits_processed} hits.")

    return dfs_decoys, num_hits_processed


def sample_one_protein(df: pl.DataFrame) -> pl.DataFrame:
    """Randomly pick one source protein among many if necessary.

    The function also samples 'origin_species' and flanks accordingly.

    Args:
        df: Dataframe to process, with the columns 'protein_id', 'origin_species' 'left_flank' and
            'right_flank'.

    Returns:
        Dataframe with one source protein per peptide (and its respective 'origin_species' and
        flanks if necessary).
    """
    return (
        df.with_columns(
            pl.col("protein_id").str.split(";"),
            pl.col("origin_species").str.split(";"),
            pl.col("left_flank").str.split(";"),
            pl.col("right_flank").str.split(";"),
        )
        # Generate a random index per sample
        .with_columns(
            pl.int_ranges(start=0, end=pl.col("protein_id").list.len())
            .list.sample(n=1)
            .list.first()
            .alias("index")
        )
        .with_columns(
            pl.col("protein_id").list.get("index"),
            pl.col("origin_species").list.get("index"),
            pl.col("left_flank").list.get("index"),
            pl.col("right_flank").list.get("index"),
        )
        .drop("index")
    )


def maybe_remove_imperfect_matching(
    df_hits: pl.DataFrame,
    remove_imperfect_matching: bool,
    nullify_imperfect_flanks: bool,
    flank_size: int,
) -> pl.DataFrame:
    """Filter peptide with imperfect matches or replace imperfect flanks.

    Args:
        df_hits: Input dataframe with hits.
        remove_imperfect_matching: Whether we remove peptides with imperfect match.
        nullify_imperfect_flanks: Whether we replace imperfect flanks with padding token.
        flank_size: Size of the flanks to consider.

    Returns:
        Input dataframe with hits and imperfect matches processed if necessary.
    """
    # We remove peptides that match imperfectly with a source protein (using blastp)
    if remove_imperfect_matching:
        df_hits = df_hits.filter(pl.col("exact_protein_match") == 1)

    # We replace imperfect flanks with padding tokens
    if flank_size != 0 and nullify_imperfect_flanks:
        df_hits = df_hits.with_columns(
            pl.when(pl.col("exact_protein_match") == 0)
            .then(pl.col("left_flank").str.replace_all(r"[A-Z]", PAD_TOKEN))
            .otherwise(pl.col("left_flank"))
            .alias("left_flank"),
            pl.when(pl.col("exact_protein_match") == 0)
            .then(pl.col("right_flank").str.replace_all(r"[A-Z]", PAD_TOKEN))
            .otherwise(pl.col("right_flank"))
            .alias("right_flank"),
        )

    return df_hits


def validate_and_select_columns(
    available_columns: set[str],
    with_columns: set[str] | None,
    flank_size: int,
    require_allele: bool = True,
) -> tuple[set[str] | None, list[str]]:
    """Check if required columns are present and validate with_columns parameter.

    Args:
        available_columns: Set of available columns in the input dataframe.
        with_columns: Set of columns to check if they are present in the input dataframe.
        flank_size: Size of the flanks to consider.
        require_allele: Whether the 'allele' column is required (e.g., for generate-decoys).

    Returns:
        Tuple containing:
            - with_columns: Set of columns to copy from the original dataset.
            - columns_to_select: List of columns to select from the input dataframe.

    Raises:
        ValueError:
            - If required columns are not present in the input dataframe.
            - If any column in `with_columns` is not present in the input dataframe.
            - If "peptide" column is not present in the input dataframe.
            - If "allele" column is not present when `require_allele=True`.
    """
    assign_protein_features_columns = {
        "protein_id",
        "origin_species",
        "left_flank",
        "right_flank",
        "exact_protein_match",
    }
    if not assign_protein_features_columns.issubset(available_columns):
        missing_required_columns = assign_protein_features_columns - available_columns
        raise ValueError(
            f"Input file is missing required columns: "
            f"{format_iterable(missing_required_columns)}. "
            "Please run 'assign-protein-features' first to generate these columns."
        )

    if "peptide" not in available_columns:
        raise ValueError("Input CSV must contain a 'peptide' column.")

    if require_allele and "allele" not in available_columns:
        raise ValueError("Input CSV must contain a 'allele' column.")

    should_select_all_columns = (
        with_columns and len(with_columns) == 1 and list(with_columns)[0] == "all"
    )

    if should_select_all_columns:
        excluded_columns = {"peptide", "hit"} | assign_protein_features_columns
        if require_allele:
            excluded_columns.add("allele")
        with_columns = available_columns - excluded_columns

    # Check if with_columns are present in the CSV
    if with_columns:
        missing_columns = with_columns - set(available_columns)
        if missing_columns:
            raise ValueError(
                "The following columns specified in --with_columns "
                f"are not present in the input CSV: {format_iterable(missing_columns)}"
            )

    # Select required columns plus any additional columns specified
    columns_to_select = ["peptide", "protein_id", "origin_species", "exact_protein_match"]
    if require_allele:
        columns_to_select.insert(0, "allele")

    if with_columns:
        columns_to_select.extend(list(with_columns))

    if flank_size != 0:
        columns_to_select.extend(["left_flank", "right_flank"])

    return with_columns, columns_to_select
