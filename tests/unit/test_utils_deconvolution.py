"""Unit tests related to bench_mhc/utils/deconvolution.py."""

import numpy as np
import polars as pl
import pytest

from bench_mhc.constants import LazyOrDataFrame
from bench_mhc.utils.deconvolution import apply_deconvolution


@pytest.mark.parametrize(
    "predictions_identifier",
    ["predictions", "random_predictions_identifier_not_in_dataset"],
)
@pytest.mark.parametrize(
    "deconvolution_identifier",
    ["MA_bag_identifier", "random_deconvolution_identifier_not_in_dataset"],
)
@pytest.mark.parametrize("ma_dataframe", ["lazy", "eager"], indirect=True)
def test_apply_deconvolution(
    predictions_identifier: str,
    deconvolution_identifier: str,
    ma_dataframe: LazyOrDataFrame,
) -> None:
    """Test that 'apply_deconvolution' works as expected."""
    if deconvolution_identifier == "random_deconvolution_identifier_not_in_dataset":
        error_msg = (
            f"The column '{deconvolution_identifier}' to use for deconvolution is not available in "
            "the dataset."
        )
        with pytest.raises(KeyError, match=error_msg):
            apply_deconvolution(
                df=ma_dataframe,
                predictions_identifier=predictions_identifier,
                deconvolution_identifier=deconvolution_identifier,
            )

    elif predictions_identifier == "random_predictions_identifier_not_in_dataset":
        error_msg = (
            f"The column '{predictions_identifier}' to use for deconvolution is not available in "
            "the dataset."
        )
        with pytest.raises(KeyError, match=error_msg):
            apply_deconvolution(
                df=ma_dataframe,
                predictions_identifier=predictions_identifier,
                deconvolution_identifier=deconvolution_identifier,
            )

    else:
        expected_df = pl.DataFrame(
            data={
                "allele": ["HLA__C0102", "HLA__A0201", "HLA__DRB11610", "HLA__DPA10103=DPB10101"],
                "peptide": ["AAAA", "AAAA", "BBBB", "CCCC"],
                "sample_alias": ["exp1", "exp2", "exp1", "exp2"],
                "random": [5, 9, 4, 6],
                "predictions": [1.0, 0.2, 0.9, 0.5],
                "hit": [1, 0, 0, 1],
                "binding_affinity": [-1, -1, -1, -1],
                "MA_bag_identifier": [
                    "AAAA__exp1",
                    "AAAA__exp2",
                    "BBBB__exp1",
                    "CCCC__exp2",
                ],
            }
        )

        deconvoluted_lf = apply_deconvolution(
            df=ma_dataframe,
            predictions_identifier=predictions_identifier,
            deconvolution_identifier=deconvolution_identifier,
        )

        deconvoluted_df = deconvoluted_lf.lazy().collect()

        assert deconvoluted_df.equals(expected_df)

        assert np.all(
            sorted(deconvoluted_df.select(deconvolution_identifier).to_numpy())
            == deconvoluted_df.select(deconvolution_identifier).to_numpy()
        )
