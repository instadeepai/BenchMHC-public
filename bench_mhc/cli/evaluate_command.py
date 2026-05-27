"""Module to define the entry point of the command line to evaluate."""

from pathlib import Path

import rich_click as click

from bench_mhc.utils.click import arguments


@click.command()
@arguments.dataset_path
@arguments.prediction_identifier
@arguments.target_identifier
@arguments.output_directory
@arguments.group_identifier
@arguments.with_frank_score
def evaluate(
    dataset_path: Path,
    prediction_identifier: str,
    target_identifier: str,
    output_directory: Path,
    group_identifier: str,
    with_frank_score: bool,
) -> None:
    """Evaluate on dataset_path based on predictions.

    It computes the following metrics for each allele and globally:

    \b
    - Top-K
    - PR-AUC (Precision-Recall Area Under Curve)
    - AP (Average Precision)
    - ROC-AUC (Receiver Operating Characteristic Area Under Curve)
    - F1 score

    The input dataset must contain three columns:

    \b
    - An allele column (specified by `--allele_column_name`) for grouping. Default is
      `allele`.
    - A predictions column (specified by `--prediction_identifier`).
    - A target column (specified by `--target_identifier`).

    Results are saved as CSV files under the output_directory with the following naming
    convention: `{dataset_path.stem}_{prediction_identifier}_{target_identifier}.csv`
    """  # noqa: D301
    from bench_mhc.cli.evaluate import evaluate

    evaluate(
        dataset_path=dataset_path,
        prediction_identifier=prediction_identifier,
        target_identifier=target_identifier,
        output_dir=output_directory,
        group_identifier=group_identifier,
        with_frank_score=with_frank_score,
    )
