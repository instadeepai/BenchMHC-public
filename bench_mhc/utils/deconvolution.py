"""Module used to define functions related to deconvolution (i.e. MA annotation)."""

import polars as pl

from bench_mhc.constants import LazyOrDataFrame
from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.logging import system

log = system.get(__name__)


def apply_deconvolution(
    df: LazyOrDataFrame, predictions_identifier: str, deconvolution_identifier: str
) -> LazyOrDataFrame:
    """Given scores (e.g. target predictions), apply deconvolution to the dataset.

    For a given unique value out of the column named as 'deconvolution_identifier': the sample with
    the highest score is kept.

    The resulting deconvoluted dataset is sorted according to 'deconvolution_identifier'.

    Args:
        df: The input dataset of length n_samples.
        predictions_identifier: The column name for the column containing the predictions to use for
            deconvolution.
        deconvolution_identifier: The column name to use for deconvolution.

    Returns:
        The deconvoluted dataset, sorted according to 'deconvolution_identifier'.

    Raises:
        KeyError: if the 'deconvolution_identifier' or 'predictions_identifier' columns are not
            available in the dataset
    """
    available_columns = set(df.collect_schema().names())

    for column_name in (deconvolution_identifier, predictions_identifier):
        if column_name not in available_columns:
            raise KeyError(
                f"The column '{column_name}' to use for deconvolution is not available in the "
                f"dataset. Available columns: {format_iterable(available_columns)}"
            )

    log.info(
        f"Deconvolution applied on probabilities '{predictions_identifier}': the higher the "
        "better."
    )

    lf = (
        df.lazy()
        .filter(
            pl.col(predictions_identifier)
            == pl.col(predictions_identifier).max().over(deconvolution_identifier)
        )
        .sort(deconvolution_identifier)
    )

    return lf if isinstance(df, pl.LazyFrame) else lf.collect()
