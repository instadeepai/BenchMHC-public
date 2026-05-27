"""Module to define the entry point of the command line to calibrate a model."""

from pathlib import Path

import rich_click as click

from bench_mhc.utils.click import arguments


@click.command(context_settings={"show_default": True})
@arguments.model_path
@arguments.output_directory
@arguments.reference_path
@arguments.batch_size
@arguments.num_workers
@arguments.gpus
@arguments.alleles
@arguments.peptides_only
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
    """Calibrate a model to generate interpretable prediction scores.

    Key steps:

    \b
    - Takes a reference dataset of peptides and alleles
    - Makes predictions for all possible combinations
    - Creates lookup tables to convert raw predictions into percentile ranks
    - Saves results as .npz files for each allele and head type

    There is one output file per allele and head type, named `{allele}__{output_name}.npz`,
    containing the percentile ranks.

    For the fully detailed logic, refer to `bench_mhc.cli.calibrate.py::calibrate`.

    !!! Note

        If `--alleles` is not provided, the calibration involves running predictions for all
        possible combinations of peptides and alleles which represents ~50_000_000 samples.
    """  # noqa: D301
    from bench_mhc.cli.calibrate import calibrate

    calibrate(
        model_path=model_path,
        output_directory=output_directory,
        reference_path=reference_path,
        batch_size=batch_size,
        num_workers=num_workers,
        gpus=gpus,
        alleles=alleles,
        peptides_only=peptides_only,
    )
