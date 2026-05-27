"""Module to define the entry point of the command line to generate performance plots."""

from pathlib import Path

import rich_click as click

from bench_mhc.utils.click import arguments


@click.command()
@arguments.configuration_file_path
@arguments.output_directory
@arguments.cd8_epitopes
@arguments.ms_ligands
def generate_performance_plot(
    configuration_file_path: Path,
    output_directory: Path,
    cd8_epitopes: bool,
    ms_ligands: bool,
) -> None:
    """Generate performance plots from metrics files.

    The configuration file should be a YAML file with model names as keys and paths to their
    respective metrics files as values. The metrics files should be CSV files with at least
    the following columns:

    For MS ligands metrics:

    \b
    - `allele`: The HLA allele name
    - `Top-K`: The Top-K score (ranging from 0 to 1)

    For CD8 epitopes' metrics:

    \b
    - `epitope`: The epitope name
    - `Frank Score`: The FRANK score (ranging from 0 to 1)

    You can also check the `configuration/template_performance_plot.yml` file for
    a complete example.

    Each metrics file should also include:

    \b
    - Individual metrics for each allele / epitope
    - A "Global" row with overall metrics for MS ligands and
        a "Median" row with median values for CD8 epitopes
    - A "Mean" row with mean values for both MS ligands and CD8 epitopes

    You should provide either the `--cd8_epitopes` or `--ms_ligands` flag, but not both.
    """  # noqa: D301
    from bench_mhc.cli.generate_performance_plot import generate_performance_plot

    generate_performance_plot(
        configuration_file_path=configuration_file_path,
        output_directory=output_directory,
        cd8_epitopes=cd8_epitopes,
        ms_ligands=ms_ligands,
    )
