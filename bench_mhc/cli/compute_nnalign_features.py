"""Module to define the command line to compute NNAlign features."""

from pathlib import Path

import polars as pl

from bench_mhc.utils.click import logic
from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.logging import system
from bench_mhc.variables.nn_align import NNAlignVariable

log = system.get(__name__)


def compute_nnalign_features(
    dataset_path: Path,
    output_file_path: Path | None,
    peptide_column_name: str,
    force: bool,
) -> None:
    """Add the NNAlign features to the dataset.

    Refer to bench_mhc.cli.compute_nnalign_features_command.py::compute_nnalign_features
    for the complete documentation.
    """
    logic.check_abort_if_file_exists(output_file_path, force)

    lf = pl.scan_csv(dataset_path)

    if peptide_column_name not in lf.collect_schema().names():
        raise ValueError(
            f"The dataset {dataset_path} does not have the '{peptide_column_name}' column "
            "required to compute the NetMHCpan-4.1 features."
        )

    if output_file_path is None:
        logic.check_abort_if_file_exists(dataset_path, force)
        output_file_path = dataset_path

    new_columns = [f"{peptide_column_name}_{suffix}" for suffix in NNAlignVariable.suffixes]
    log.info(
        f"The NetMHCpan-4.1 (NNAlign) features {format_iterable(new_columns)} "
        "will be added to the dataset."
    )

    peptide = NNAlignVariable(name=peptide_column_name)
    peptide.fit(lf)
    lf = peptide.build_nn_align_features(lf)

    lf.collect(engine="streaming").write_csv(output_file_path)
    log.info(f"NetMHCpan-4.1 features saved in {output_file_path}.")
