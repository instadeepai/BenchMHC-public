"""Module to define the entry point of the command line to format the allele feature."""

from pathlib import Path

import rich_click as click

from bench_mhc.utils.click import arguments
from bench_mhc.utils.logging import system

log = system.get(__name__)


@click.command(context_settings={"show_default": True})
@arguments.dataset_path
@arguments.output_file_path
@arguments.allele_column_name
@arguments.force
def format_allele_feature(
    dataset_path: Path,
    output_file_path: Path,
    allele_column_name: str,
    force: bool,
) -> None:
    """Format the allele feature column in the dataset.

    The following format is used: '{species}__{compact_parsed_allele_string}',
    e.g. HLA__A0101, Mamu__A20511 or HLA__DPB10305.
    """  # noqa: D301
    from bench_mhc.cli.format_allele_feature import format_allele_feature

    format_allele_feature(
        dataset_path=dataset_path,
        output_file_path=output_file_path,
        allele_column_name=allele_column_name,
        force=force,
    )
