"""Module to define the command line to predict."""

import typing
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from lightning import Trainer
from scipy.special import expit

from bench_mhc.cli.compute_nnalign_features import compute_nnalign_features
from bench_mhc.constants import CACHE_DIRECTORY
from bench_mhc.constants import SEPARATOR
from bench_mhc.dataset.main import DataModule
from bench_mhc.utils.device import get_devices_and_accelerator
from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.logging import system
from bench_mhc.utils.model import load_model_from_path
from bench_mhc.utils.model import should_apply_sigmoid_for_percentile_rank
from bench_mhc.utils.percentile_rank import PercentileRankCalculator
from bench_mhc.utils.predictions import process_outputs

log = system.get(__name__)


def predict(
    model_paths: set[Path],
    dataset_path: Path,
    predictions_column_prefix: str,
    output_file_path: Path | None,
    num_workers: int,
    batch_size: int,
    gpus: list[int] | None,
    percentile_rank_directory: Path | None,
) -> None:
    """Perform predictions on the provided dataset.

    Refer to bench_mhc.cli.predict_command.py::predict for the complete documentation.

    Args:
        model_paths: The names of the model to use for the ensemble.
        dataset_path: The path to the dataset to run inference.
        predictions_column_prefix: Prefix of the column to append predictions to the dataset.
        output_file_path: The optional path to the output predictions file.
            If not provided, predictions are saved in dataset_path.
        num_workers: The number of workers to use.
        batch_size: The batch size to use.
        gpus: The GPUs to use.
        percentile_rank_directory: Optional directory containing percentile rank files.
            If provided and valid, predictions will be converted to percentile ranks.
    """
    num_samples = pl.scan_csv(dataset_path).select(pl.len()).collect(engine="streaming").item()
    output_name2predictions: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(
            num_samples,
        )
    )
    output_name2count: dict[str, int] = defaultdict(int)

    devices, accelerator = get_devices_and_accelerator(gpus)
    num_models = len(model_paths)
    for i, model_path in enumerate(model_paths, start=1):
        log.info(f"Predicting with model '{model_path}'...")

        model = load_model_from_path(model_path)

        for name in model.outputs.names:
            output_name2count[name] += 1

        data_module = DataModule(
            inputs=model.inputs,
            outputs=model.outputs,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers != 0 else None,
            predict_file_path=dataset_path,
            cache_directory=CACHE_DIRECTORY,
        )

        trainer = Trainer(
            default_root_dir=model_path,
            deterministic=True,
            devices=devices,
            accelerator=accelerator,
        )

        predictions = process_outputs(
            # mypy says list[Any] is incompatible with list[dict[str, torch.Tensor]]
            trainer.predict(datamodule=data_module, model=model, return_predictions=True)  # type: ignore
        )

        log.info(f"Prediction with model '{model_path}' done.")
        for output_name, predictions_ in predictions.items():
            if output_name.endswith("selected_core_indices"):
                output_name2predictions[output_name] = np.vstack(
                    [output_name2predictions[output_name], predictions_]
                )
            else:
                output_name2predictions[output_name] += predictions_

        log.info(f"{i}/{num_models} models finished running predictions.")

    output_name2predictions_postprocessed = postprocess_predictions(
        output_name2predictions=output_name2predictions,
        output_name2count=output_name2count,
    )

    log.info("Prediction for all models done.")
    output_file_path = output_file_path or dataset_path
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    df_predictions = (
        pl.DataFrame(output_name2predictions_postprocessed)
        # Convert list of selected core to ;-separated string
        .with_columns(cs.ends_with("selected_core_indices").arr.join(";"))
        # Rename columns `predictions_column_prefix__output_name`
        .select(pl.all().name.prefix(f"{predictions_column_prefix}{SEPARATOR}"))
    )

    df = pl.read_csv(dataset_path)
    common_columns = set(df.columns).intersection(df_predictions.columns)

    if common_columns:
        log.info(
            "Input dataset already contains predictions for the current model(s). "
            "The current predictions will be overridden by the new predictions."
        )

    df = (
        df.select(pl.exclude(common_columns))
        # Concat predictions to input dataset
        .hstack(df_predictions)
    )

    df = _maybe_add_percentile_rank_values(
        df=df,
        percentile_rank_directory=percentile_rank_directory,
        output_name2predictions=output_name2predictions,
        predictions_column_prefix=predictions_column_prefix,
        model_paths=model_paths,
    )

    df.write_csv(output_file_path)
    log.info(f"Predictions written to {output_file_path}.")

    _maybe_add_9mers_core_columns(df_predictions.columns, output_file_path)


def postprocess_predictions(
    output_name2predictions: dict[str, np.ndarray], output_name2count: dict[str, int]
) -> dict[str, np.ndarray]:
    """Post-process predictions.

    It computes the majority core index for the selected_core_indices predictions.
    It also computes the average for the other predictions.

    Args:
        output_name2predictions: The predictions to post-process.
        output_name2count: The number of submodels per output.

    Returns:
        The post-processed predictions. It contains:
            - The majority core index for the selected_core_indices predictions
              (binding_affinity and/or hit).
            - The list of selected_core_indices predictions, one element of the list
              corresponds to one submodel prediction (binding_affinity and/or hit).
            - The average value for the other predictions (probability of hit and/or
              binding_affinity).
    """
    output_name2predictions_postprocessed = {}
    for output_name, predictions in output_name2predictions.items():
        if output_name.endswith("selected_core_indices"):
            # Remove the first column of 0 and convert to integer
            predictions = predictions[1:, :].T.astype(int)

            output_name2predictions_postprocessed[
                output_name.replace("selected_core_indices", "majority_core_index")
            ] = torch.mode(torch.from_numpy(predictions), dim=1).values.numpy()
            # Convert predictions to string
            output_name2predictions_postprocessed[output_name] = predictions.astype(str)
        else:
            # Compute the average
            output_name2predictions_postprocessed[output_name] = (
                output_name2predictions[output_name] / output_name2count[output_name]
            )

    return output_name2predictions_postprocessed


def _maybe_add_9mers_core_columns(
    df_predictions_columns: list[str], output_file_path: Path
) -> None:
    """Adds 9-mers core columns to the dataset if the dataset contains relevant core columns.

    Args:
        df_predictions_columns: The columns of the predictions.
        output_file_path: The path to the output file.
    """
    majority_core_column_names = {
        column_name
        for column_name in df_predictions_columns
        if column_name.endswith("majority_core_index")
    }

    if not majority_core_column_names:  # pragma: no cover
        log.info(
            "No majority core column names found. Skipping 9-mers core columns addition."
        )  # pragma: no cover

        return  # pragma: no cover

    log.info("Adding selected core aa sequence to the dataset...")
    compute_nnalign_features(
        output_file_path, output_file_path, peptide_column_name="peptide", force=True
    )

    df = (
        pl.scan_csv(output_file_path)
        .with_columns(
            # No way to write this more elegantly
            pl.when(pl.col(majority_core_column_name) == 0)
            .then(pl.col("peptide_core_0"))
            .when(pl.col(majority_core_column_name) == 1)
            .then(pl.col("peptide_core_1"))
            .when(pl.col(majority_core_column_name) == 2)
            .then(pl.col("peptide_core_2"))
            .when(pl.col(majority_core_column_name) == 3)
            .then(pl.col("peptide_core_3"))
            .when(pl.col(majority_core_column_name) == 4)
            .then(pl.col("peptide_core_4"))
            .when(pl.col(majority_core_column_name) == 5)
            .then(pl.col("peptide_core_5"))
            .when(pl.col(majority_core_column_name) == 6)
            .then(pl.col("peptide_core_6"))
            .when(pl.col(majority_core_column_name) == 7)
            .then(pl.col("peptide_core_7"))
            .when(pl.col(majority_core_column_name) == 8)
            .then(pl.col("peptide_core_8"))
            .when(pl.col(majority_core_column_name) == 9)
            .then(pl.col("peptide_core_9"))
            # This is now the amino acid sequence so we rename it without the _index suffix
            .alias(majority_core_column_name.removesuffix("_index"))
            for majority_core_column_name in majority_core_column_names
        )
        # Keep only the original columns + the majority AA sequence core columns
        # by removing all new columns produced by the compute_nnalign_features function
        .select(~cs.starts_with("peptide_"))
        .collect(engine="streaming")
    )

    df.select(sorted(df.columns)).write_csv(output_file_path)
    log.info(f"Selected 9mers core added to {output_file_path}.")


def _maybe_add_percentile_rank_values(
    df: pl.DataFrame,
    percentile_rank_directory: Path | None,
    output_name2predictions: dict[str, np.ndarray],
    predictions_column_prefix: str,
    model_paths: set[Path],
) -> pl.DataFrame:
    """Add percentile rank columns to the dataset if the percentile rank directory is provided.

    Args:
        df: The input dataset.
        percentile_rank_directory: The directory containing the percentile rank files.
        output_name2predictions: The predictions to add the percentile rank column to.
        predictions_column_prefix: The prefix of the predictions column.
        model_paths: Set of model paths used for prediction (to check if sigmoid is needed).

    Returns:
        The dataset with the percentile rank column added if percentile_rank_directory is provided.
        Otherwise, the input dataset is returned unchanged.
    """
    if percentile_rank_directory is None:
        return df

    # Check if we need to apply sigmoid (for models using BCEWithLogitsLoss)
    first_model_path = list(model_paths)[0]
    apply_sigmoid = should_apply_sigmoid_for_percentile_rank(first_model_path)

    if apply_sigmoid:
        log.info(
            "Model uses BCEWithLogitsLoss. Applying sigmoid to convert logits to "
            "probabilities before converting to percentile ranks."
        )

    column_names = df.columns

    unique_alleles = set(df.select(pl.col("allele").unique())["allele"])
    for output_name in output_name2predictions:
        missing_alleles_in_pctrnk_dir = {
            allele
            for allele in unique_alleles
            if not (percentile_rank_directory / f"{allele}__{output_name}.npz").exists()
        }
        if missing_alleles_in_pctrnk_dir:
            log.warning(
                f"Missing percentile rank files for alleles: "
                f"{format_iterable(missing_alleles_in_pctrnk_dir)} "
                f"for output `{output_name}`. "
                "Falling back to raw predictions."
            )

            continue

        predictions_column_name = f"{predictions_column_prefix}{SEPARATOR}{output_name}"
        pctrnk_column_name = f"{predictions_column_prefix}{SEPARATOR}pctrnk{SEPARATOR}{output_name}"

        dfs_output = []
        for allele_tuple, df_predictions_allele in df.group_by("allele"):
            # Required by mypy since type of the grouper is inferred as object
            allele = typing.cast(
                str,
                # polars.DataFrame.group_by returns a tuple with a single element
                allele_tuple[0],
            )

            dfs_output.append(
                df_predictions_allele.pipe(
                    _add_percentile_rank_columns,
                    predictions_column_name,
                    pctrnk_column_name,
                    percentile_rank_directory,
                    allele,
                    output_name,
                    apply_sigmoid,
                )
            )

        df_output = pl.concat(dfs_output, how="vertical")
        # This adds the percentile rank column to the original dataset.
        # We ensure with `join` that the order of the rows produced by the `group_by` is not
        # a concern for the final result in case it is different from the order of the rows
        # in the original dataset.
        df = df.join(
            df_output.select(*column_names, pctrnk_column_name),
            on=column_names,
            # In case `hit` or `binding_affinity` columns have `null` values,
            # we still want to keep the rows.
            nulls_equal=True,
        )

    return df


def _add_percentile_rank_columns(
    df_predictions_allele: pl.DataFrame,
    predictions_column_name: str,
    pctrnk_column_name: str,
    percentile_rank_directory: Path,
    allele: str,
    output_name: str,
    apply_sigmoid: bool = False,
) -> pl.DataFrame:
    """Add percentile rank columns to the input dataset.

    Args:
        df_predictions_allele: The predictions dataset.
        predictions_column_name: The name of the predictions column.
        pctrnk_column_name: The name of the percentile rank column.
        percentile_rank_directory: The directory containing the percentile rank files.
        allele: The allele to add the percentile rank column to.
        output_name: The name of the output to add the percentile rank column to.
        apply_sigmoid: Whether to apply sigmoid to convert logits to probabilities.

    Returns:
        The input dataset with the percentile rank column added.
    """
    pctrnk_fp = percentile_rank_directory / f"{allele}__{output_name}.npz"
    pctrnk_calculator = PercentileRankCalculator(pctrnk_path=pctrnk_fp)

    # Convert predictions to percentile ranks
    predictions = df_predictions_allele[predictions_column_name].to_numpy()

    # Apply sigmoid if model outputs logits
    if apply_sigmoid:
        predictions = expit(predictions)

    pctrnk_values = pctrnk_calculator.convert_probs_to_percentile_ranks(predictions)

    # Add percentile rank column
    df_predictions_allele = df_predictions_allele.with_columns(
        pl.Series(name=pctrnk_column_name, values=pctrnk_values)
    )

    return df_predictions_allele
