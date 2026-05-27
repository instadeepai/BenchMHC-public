"""Module to define the command line to calibrate a model."""

import tempfile
import typing
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit

from bench_mhc.cli.predict import predict
from bench_mhc.constants import ALLELE_MAPPING_PATH
from bench_mhc.constants import NUM_STEPS_PCTRNK
from bench_mhc.utils.io import load_json
from bench_mhc.utils.logging.system import get
from bench_mhc.utils.model import should_apply_sigmoid_for_percentile_rank
from bench_mhc.utils.percentile_rank import PercentileRankCalculator

log = get(__name__)


def calibrate(
    model_path: set[Path],
    output_directory: Path,
    reference_path: Path,
    batch_size: int,
    num_workers: int,
    gpus: list[int] | None,
    alleles: set[str] | None,
    peptides_only: bool,
) -> None:
    """Calibrate a model or ensemble by generating percentile rank files for each allele group.

    This function performs the following steps:

    1. Loads the allele-to-pseudo-sequence mapping, filter based on the optionally provided
        alleles and groups alleles by unique pseudo sequences. For each group, the first allele
        is used as a representative for calibration.

    2. Builds a reference set by taking the cartesian product of all unique peptides
        in the provided reference set and the representative alleles.

    3. Runs predictions for all peptide-allele pairs using the specified model(s).

    4. For each output type ("hit" and "binding_affinity"), and for each allele group,
        computes percentile ranks from the predictions.

    5. Saves the resulting percentile rank files as `.npz` files,
        named `{allele}__{output_name}.npz`, in the specified output directory.

    Args:
        model_path: Set of paths to model files or ensemble model txt files.
        output_directory: Directory where percentile rank files will be saved.
        reference_path: Path to the reference set file (CSV with a 'peptide' column).
        batch_size: Number of peptides to process in a single batch.
        num_workers: Number of worker threads to use for parallel processing.
        gpus: List of GPU IDs to use for parallel processing (or None for CPU).
        alleles: Set of alleles to calibrate. If not provided,
            all alleles in the mapping `allele2pseudo_sequence.json` will be used.
        peptides_only: If True, builds reference set from unique peptides in the
            reference file and allele pseudo sequences from the mapping file (Cartesian product).
            If False, loads the prebuilt reference set directly from the reference file.
    """
    if peptides_only:
        allele_mapping = load_json(ALLELE_MAPPING_PATH)
        if alleles is not None:
            allele_mapping = {allele: allele_mapping[allele] for allele in alleles}

        df_allele_mapping = (
            pl.from_records(
                list(allele_mapping.items()),
                schema=["allele", "pseudo_seq"],
                orient="row",
            )
            # Create groups of alleles with the same pseudo sequence
            .group_by("pseudo_seq")
            # Sort the alleles within each group for reproducibility and testing
            .agg(pl.col("allele").sort().alias("allele_group"))
            # Get the first allele of each allele group
            #    1. to use for the reference set
            #    2. to map back the predictions to the original allele groups
            .with_columns(pl.col("allele_group").list.first().alias("allele"))
            # Drop the pseudo sequence column to clear some memory
            .drop("pseudo_seq")
        )

        log.info(f"Running calibration for {len(df_allele_mapping)} alleles.")

        lf_allele_mapping = df_allele_mapping.lazy()

        reference_set_df = build_reference_set_from_peptides_and_alleles_mapping(
            reference_path=reference_path,
            lf_alleles=lf_allele_mapping.select("allele"),
        )

    else:
        reference_set_df = pl.scan_csv(reference_path).collect(engine="streaming")

    log.info(f"Reference set has {len(reference_set_df):_} rows.")

    log.info("Starting calibration ...")
    pctrnk_calculator = PercentileRankCalculator(NUM_STEPS_PCTRNK)
    with tempfile.NamedTemporaryFile(suffix=".csv") as reference_set_file_obj:
        tmp_ref_set_path = Path(reference_set_file_obj.name)
        reference_set_df.write_csv(tmp_ref_set_path)

        predictions_column_prefix = list(model_path)[0].stem

        predict(
            model_paths=model_path,
            dataset_path=tmp_ref_set_path,
            predictions_column_prefix=predictions_column_prefix,
            output_file_path=tmp_ref_set_path,
            batch_size=batch_size,
            num_workers=num_workers,
            gpus=gpus,
            percentile_rank_directory=None,
        )

        available_columns = set(pl.scan_csv(tmp_ref_set_path).collect_schema().names())

        # Check if we need to apply sigmoid (for models using BCEWithLogitsLoss)
        # Load the first model to check its configuration
        first_model_path = list(model_path)[0]
        apply_sigmoid = should_apply_sigmoid_for_percentile_rank(first_model_path)

        if apply_sigmoid:
            log.info(
                "Model uses BCEWithLogitsLoss. Applying sigmoid to convert logits to "
                "probabilities before computing percentile ranks."
            )

        # Calibrate for both hit and binding affinity since NetMHCpan-4.1
        # also provides binding affinity ranks
        for output_name in {"hit", "binding_affinity"}:
            prediction_column_name = f"{predictions_column_prefix}__{output_name}"
            if prediction_column_name not in available_columns:
                log.info(f"Column {prediction_column_name} not found in {tmp_ref_set_path}.")

                continue

            if peptides_only:
                df_predictions = (
                    pl.scan_csv(tmp_ref_set_path)
                    .select(prediction_column_name, "allele")
                    .join(lf_allele_mapping, on="allele", how="left")
                    .collect(engine="streaming")
                )
            else:
                df_predictions = (
                    pl.scan_csv(tmp_ref_set_path)
                    .select(prediction_column_name, "allele")
                    .collect(engine="streaming")
                )

            # Check if 'allele_group' column exists
            # (from dict mapping with pseudo sequences)
            if "allele_group" in df_predictions.columns:
                for allele_group_tuple, df_allele_predictions in df_predictions.group_by(
                    "allele_group"
                ):
                    # Required by mypy since type of the grouper is inferred as object
                    allele_group = typing.cast(
                        list[str],
                        # polars.DataFrame.group_by returns a tuple with a single element
                        allele_group_tuple[0],
                    )

                    predictions = df_allele_predictions[prediction_column_name].to_numpy()

                    # Apply sigmoid if model outputs logits
                    if apply_sigmoid:
                        predictions = expit(predictions)

                    percentile_ranks = pctrnk_calculator.compute(predictions)

                    for allele in allele_group:
                        output_pctrnk_file = output_directory / f"{allele}__{output_name}.npz"
                        np.savez(file=output_pctrnk_file, **percentile_ranks)

                        log.info(f"Saved percentile ranks to {output_pctrnk_file}.")
            else:
                # No allele_group column: calculate percentile ranks separately for each allele
                for allele in df_predictions["allele"].unique().to_list():
                    df_allele_predictions = df_predictions.filter(pl.col("allele") == allele)

                    predictions = df_allele_predictions[prediction_column_name].to_numpy()

                    # Apply sigmoid if model outputs logits
                    if apply_sigmoid:
                        predictions = expit(predictions)

                    percentile_ranks = pctrnk_calculator.compute(predictions)

                    output_pctrnk_file = output_directory / f"{allele}__{output_name}.npz"
                    np.savez(file=output_pctrnk_file, **percentile_ranks)

                    log.info(f"Saved percentile ranks to {output_pctrnk_file}.")
    log.info("Calibration complete")


def build_reference_set_from_peptides_and_alleles_mapping(
    reference_path: Path,
    lf_alleles: pl.LazyFrame,
) -> pl.DataFrame:
    """Build a reference set from peptides and allele pseudo sequences.

    Creates a reference dataset by taking the cartesian product of a list of peptides
    and a set of allele pseudo sequences derived from an allele mapping dictionary.

    Args:
        reference_path: Path to the reference set file.
        lf_alleles: LazyFrame containing first allele of each allele group
            having the same pseudo sequence.

    Returns:
        pl.DataFrame: A DataFrame containing all possible peptide-pseudo sequence pairs,
            with columns 'peptide' and 'allele'.
    """
    # Use lazy frames for cross join to handle large datasets
    lf_peptide = pl.scan_csv(reference_path).select(pl.col("peptide").unique())

    # Cross join with streaming to handle large result sets (>2^32 rows)
    return lf_peptide.join(lf_alleles, how="cross").collect(engine="streaming")
