"""Module to define the entry point of the command line to predict."""

from pathlib import Path

import rich_click as click

from bench_mhc.utils.click import arguments


@click.command(context_settings={"show_default": True})
@arguments.model_path
@arguments.dataset_path
@arguments.predictions_column_prefix
@arguments.output_file_path
@arguments.batch_size
@arguments.num_workers
@arguments.gpus
@arguments.percentile_rank_directory
def predict(
    model_path: set[Path],
    dataset_path: Path,
    predictions_column_prefix: str,
    output_file_path: Path | None,
    batch_size: int,
    num_workers: int,
    gpus: list[int] | None,
    percentile_rank_directory: Path | None,
) -> None:
    """Run predictions on a dataset.

    If multiple models are provided, the predictions of each of them will be averaged per output.
    Final predictions are written to the output file under `output_file_path` (or `dataset_path`
    if the latter is not provided) containing the input dataset with the predictions columns
    appended. There are as many predictions columns as different outputs.

    If a percentile rank directory is provided, additional columns to the output dataset
    containing the percentile ranks of the predictions are added.
    """
    from bench_mhc.cli.predict import predict

    predict(
        model_paths=model_path,
        dataset_path=dataset_path,
        predictions_column_prefix=predictions_column_prefix,
        output_file_path=output_file_path,
        batch_size=batch_size,
        num_workers=num_workers,
        gpus=gpus,
        percentile_rank_directory=percentile_rank_directory,
    )
