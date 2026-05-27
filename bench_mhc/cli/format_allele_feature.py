"""Module to define the command line to format the allele feature."""

from pathlib import Path

import polars as pl

from bench_mhc.utils.click import logic
from bench_mhc.utils.logging import system
from bench_mhc.utils.mhc import format_allele_name

log = system.get(__name__)


def format_allele_feature(
    dataset_path: Path,
    output_file_path: Path | None,
    allele_column_name: str,
    force: bool,
) -> None:
    """Format the allele column in the dataset.

    Refer to bench_mhc.cli.format_allele_feature_command.py::format_allele_feature
    for the complete documentation.
    """
    logic.check_abort_if_file_exists(output_file_path, force)

    lf = pl.scan_csv(dataset_path)

    if allele_column_name not in lf.collect_schema().names():
        raise ValueError(
            f"The dataset {dataset_path} does not have the '{allele_column_name}' column "
            "to be formatted."
        )

    if output_file_path is None:
        logic.check_abort_if_file_exists(dataset_path, force)
        output_file_path = dataset_path

    unique_alleles = lf.select(pl.col(allele_column_name).unique()).collect(engine="streaming")[
        allele_column_name
    ]
    allele2formatted_allele = {allele: format_allele_name(allele) for allele in unique_alleles}

    lf = lf.with_columns(pl.col(allele_column_name).replace_strict(allele2formatted_allele))

    lf.collect(engine="streaming").write_csv(output_file_path)
    log.info(
        f"Column '{allele_column_name}' for the allele feature formatted in {output_file_path}."
    )
