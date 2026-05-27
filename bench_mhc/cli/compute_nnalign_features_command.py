"""Module to define the entry point of the command line to compute NNAlign features."""

from pathlib import Path

import rich_click as click

from bench_mhc.utils.click import arguments
from bench_mhc.utils.logging import system

log = system.get(__name__)


@click.command(context_settings={"show_default": True})
@arguments.dataset_path
@arguments.output_file_path
@click.option(
    "--peptide_column_name",
    "-n",
    type=str,
    required=False,
    default="peptide",
    help="Name of the column to use to extract the NetMHCpan-4.1 features.",
)
@arguments.force
def compute_nnalign_features(
    dataset_path: Path,
    output_file_path: Path,
    peptide_column_name: str,
    force: bool,
) -> None:
    """Add the NNAlign features required by the NetMHCpan-4.1 model to the input dataset.

    This command is only supported for MHC1 (NetMHCpan-4.x) for now.
    Support for MHC2 (NetMHCIIpan-4.x) will be added later.

    \b
    This features' extraction is based on NNAlign, which is the framework used by NetMHCpan-4.1.
    Details regarding these features can be found in the 'Implementation/Network training and
    architecture' section from the NetMHCpan-3.0 paper (old version of NetMHCpan-4.1), cf.
    https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-016-0288-x.

    Out of the `peptide` feature, the following features are computed:

    - `num_possible_cores`: the number of possible 9-mer cores than can be extracted from the
    original k-mer (= peptide)

    - `core_{i}/core_{i}_flank_left_len/core_{i}_flank_right_len`: all possible 9-mer cores, flank
    lefts' lengths and flank rights' lengths, for i in [0; 9], 10 being the maximal number of
    possible 9-mer cores

    - `insertion_len`: length of the insertion

    - `deletion_len`: length of the deletion

    - `is_8mer_or_less`: whether the original k-mer peptide is an 8-mer or less

    - `is_9mer`: whether the original k-mer peptide is a 9-mer

    - `is_10mer`: whether the original k-mer peptide is a 10-mer

    - `is_11mer_or_more`: whether the original k-mer peptide is an 11-mer or more
    """  # noqa: D301
    from bench_mhc.cli.compute_nnalign_features import compute_nnalign_features

    compute_nnalign_features(
        dataset_path=dataset_path,
        output_file_path=output_file_path,
        peptide_column_name=peptide_column_name,
        force=force,
    )
