"""Unit tests for the command lines in bench_mhc/cli/train.py."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import torch
from lightning import Trainer

from bench_mhc.cli.train import _annotate_ma_data
from bench_mhc.cli.train import _check_labels_are_consistent_w_deconvolution_identifier
from bench_mhc.cli.train import _compute_rescaling_params
from bench_mhc.cli.train import _rescale_ma_positive_data
from bench_mhc.dataset.main import DataModule
from bench_mhc.dataset.main import Dataset
from bench_mhc.model.mhc1_nn_align import MHC1NNAlignLightningModule
from bench_mhc.utils.mode import Mode


def test_check_labels_are_consistent_w_deconvolution_identifier(
    ma_dataframe: pl.DataFrame,
) -> None:
    """Check _check_labels_are_consistent_w_deconvolution_identifier works as expected."""
    deconvolution_identifier = "MA_bag_identifier"

    ma_dataframe.insert_column(
        0, pl.Series("hit_w_non_unique_labels", [1, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    )
    match_msg = (
        "The MA data is not properly formatted as the following values of the deconvolution "
        f"identifier '{deconvolution_identifier}' have more than one label: "
        f"'AAAA__exp1'."
    )
    with pytest.raises(ValueError, match=match_msg):
        _check_labels_are_consistent_w_deconvolution_identifier(
            ma_df=ma_dataframe,
            deconvolution_identifier=deconvolution_identifier,
            deconvolution_output_name="hit_w_non_unique_labels",
        )

    _check_labels_are_consistent_w_deconvolution_identifier(
        ma_df=ma_dataframe,
        deconvolution_identifier=deconvolution_identifier,
        deconvolution_output_name="hit",
    )


@pytest.mark.parametrize("w_shift_parameter", [50, 75])
@pytest.mark.parametrize("z_score_threshold", [3.0, 1.0])
@pytest.mark.parametrize("epoch", [3, 10])
def test_compute_rescaling_params(
    model_directory: Path,
    tmp_path: Path,
    mhc1_nn_align_configuration: dict[str, dict[str, Any]],
    mhc1_nn_align_dataset: Dataset,
    reference_path: Path,
    w_shift_parameter: int,
    z_score_threshold: float,
    epoch: int,
) -> None:
    """Check _compute_rescaling_params works as expected."""
    deconvolution_output_name = "hit"
    model_path = model_directory / "test_compute_rescaling_params"

    mock_predictions = [
        {deconvolution_output_name: torch.tensor([0.6, 0.7, 0.9, 1.0]).reshape(-1, 1)},
        {deconvolution_output_name: torch.tensor([0.4, 0.3, 0.3, 0.4]).reshape(-1, 1)},
    ]
    alleles = ["HLA__A0107", "H2__Kb"]

    reference_data = pl.scan_csv(reference_path).with_columns(
        pl.lit(sorted(alleles)[0]).alias("allele")
    )
    reference_data.collect().write_csv(reference_path)

    trainer = Trainer(default_root_dir=model_path, logger=False)

    reference_data_module = DataModule(
        transformed_data_directory=tmp_path / "test_compute_rescaling_params",
        inputs=mhc1_nn_align_dataset.inputs,
        outputs=mhc1_nn_align_dataset.outputs,
        batch_size=16,
        num_workers=0,
        prefetch_factor=None,
        predict_file_path=reference_path,
    )

    allele2value = {
        allele: mhc1_nn_align_dataset.inputs["allele"]
        .transform(pl.DataFrame({"allele": [allele]}))["allele"]
        .to_numpy()
        for allele in alleles
    }

    with patch("bench_mhc.model.mhc1_nn_align.MODEL_DIRECTORY", model_directory):
        model = MHC1NNAlignLightningModule(
            model_path=model_path,
            configuration=mhc1_nn_align_configuration,
            inputs=mhc1_nn_align_dataset.inputs,
            outputs=mhc1_nn_align_dataset.outputs,
        )

        with patch.object(model.model, "forward", side_effect=mock_predictions):
            expected_allele2stats = {}
            for i, allele in enumerate(allele2value):
                reference_scores = mock_predictions[i][f"{deconvolution_output_name}"].numpy()
                z_scores = (reference_scores - np.mean(reference_scores)) / np.std(reference_scores)
                valid_indices = np.logical_and(
                    z_scores > -z_score_threshold, z_scores < z_score_threshold
                )
                expected_allele2stats[allele] = {
                    "p_bar": np.mean(reference_scores[valid_indices]).item(),
                    "sigma": np.std(reference_scores[valid_indices]).item(),
                }

            # Compute w, p_tilde and sigma_tilde
            w = 1 / (1 + np.exp((epoch - w_shift_parameter) / 10))
            p_u_bar = np.mean(
                [output2stats["p_bar"] for output2stats in expected_allele2stats.values()]
            ).item()
            sigma_u = np.mean(
                [output2stats["sigma"] for output2stats in expected_allele2stats.values()]
            ).item()

            expected_allele2rescaling_params = {
                allele: {
                    "p_tilde": w * expected_allele2stats[allele]["p_bar"] + (1 - w) * p_u_bar,
                    "sigma_tilde": w * expected_allele2stats[allele]["sigma"] + (1 - w) * sigma_u,
                }
                for allele in expected_allele2stats
            }

            allele2rescaling_params = _compute_rescaling_params(
                model=model,
                trainer=trainer,
                reference_data_module=reference_data_module,
                allele2value=allele2value,
                deconvolution_output_name=deconvolution_output_name,
                w_shift_parameter=w_shift_parameter,
                z_score_threshold=z_score_threshold,
                epoch=epoch,
            )

    assert allele2rescaling_params == expected_allele2rescaling_params


@pytest.mark.parametrize("rescale_predictions", [False, True])
def test_annotate_ma_data(
    model_directory: Path,
    tmp_path: Path,
    mhc1_nn_align_configuration: dict[str, dict[str, Any]],
    mhc1_nn_align_dataset: Dataset,
    ma_dataframe: pl.DataFrame,
    rescale_predictions: bool,
) -> None:
    """Check _annotate_ma_data works as expected."""
    deconvolution_output_name = "hit"
    deconvolution_identifier = "MA_bag_identifier"

    variables = mhc1_nn_align_dataset.inputs + mhc1_nn_align_dataset.outputs

    ma_positive_df = variables.transform(ma_dataframe.filter(pl.col("hit") == 1))
    ma_negative_df = variables.transform(ma_dataframe.filter(pl.col("hit") == 0))

    ma_positive_data_module = DataModule(
        transformed_data_directory=tmp_path / "test_annotate_ma_data",
        inputs=mhc1_nn_align_dataset.inputs,
        outputs=mhc1_nn_align_dataset.outputs,
        batch_size=16,
        num_workers=0,
        prefetch_factor=None,
    )
    ma_positive_data_module.save_transformed_data(ma_positive_df, Mode.PREDICT)

    model_path = model_directory / "test_compute_rescaling_params"
    mock_predictions = [
        {f"{deconvolution_output_name}": torch.tensor(ma_positive_df["predictions"])},
    ]
    trainer = Trainer(default_root_dir=model_path, logger=False)

    allele2value, allele2rescaling_params = None, None
    expected_positive_data: dict = {
        "allele": ["HLA__C0102", "HLA__DPA10103=DPB10101"],
        "peptide": ["AAAA", "CCCC"],
        "sample_alias": ["exp1", "exp2"],
        "random": [5, 6],
        "predictions": [1.0, 0.5],
        "hit": [1, 1],
        "binding_affinity": [-1, -1],
        deconvolution_identifier: [
            "AAAA__exp1",
            "CCCC__exp2",
        ],
    }

    if rescale_predictions:
        expected_positive_data["predictions"][0] = 0.6
        expected_positive_data["allele"][0] = "HLA__A0201"
        expected_positive_data["random"][0] = 1

        alleles = ma_dataframe.select("allele").unique()["allele"].to_list()
        allele2value = {
            allele: mhc1_nn_align_dataset.inputs["allele"]
            .transform(pl.DataFrame({"allele": [allele]}))["allele"]
            .to_numpy()
            for allele in alleles
        }

        allele2rescaling_params = {
            allele: {
                "p_tilde": 0.0,
                "sigma_tilde": 1.0,
            }
            for allele in alleles
        }
        # HLA__A0201 will now be the selected allele out of AAAA__exp1
        allele2rescaling_params["HLA__A0201"] = {
            "p_tilde": -1.0,
            "sigma_tilde": 1.0,
        }

    expected_negative_data = {
        "allele": ["HLA__DQA10101=DQB10201", "HLA__A0201"],
        "peptide": ["BBBB", "AAAA"],
        "sample_alias": [
            "exp1",
            "exp2",
        ],
        "random": [3, 9],
        "predictions": [0.8, 0.2],
        "hit": [0, 0],
        "binding_affinity": [-1, -1],
        deconvolution_identifier: [
            "BBBB__exp1",
            "AAAA__exp2",
        ],
    }

    expected_positive_annotated_df = variables.transform(pl.DataFrame(expected_positive_data))
    expected_negative_annotated_df = variables.transform(pl.DataFrame(expected_negative_data))

    with patch("bench_mhc.model.mhc1_nn_align.MODEL_DIRECTORY", model_directory):
        model = MHC1NNAlignLightningModule(
            model_path=model_path,
            configuration=mhc1_nn_align_configuration,
            inputs=mhc1_nn_align_dataset.inputs,
            outputs=mhc1_nn_align_dataset.outputs,
        )

        with (
            patch.object(model.model, "forward", side_effect=mock_predictions),
            patch(
                "bench_mhc.cli.train.np.random.rand",
                return_value=np.array(list(range(ma_negative_df.select(pl.len()).item(), 0, -1))),
            ),
        ):
            annotated_ma_df = _annotate_ma_data(
                model=model,
                trainer=trainer,
                ma_positive_df=ma_positive_df,
                ma_negative_df=ma_negative_df,
                ma_positive_data_module=ma_positive_data_module,
                deconvolution_identifier="MA_bag_identifier",
                deconvolution_output_name=deconvolution_output_name,
                allele2value=allele2value,
                allele2rescaling_params=allele2rescaling_params,
            )

            positive_annotated_df = annotated_ma_df.filter(pl.col("hit") == 1)
            negative_annotated_df = annotated_ma_df.filter(pl.col("hit") == 0)

    assert positive_annotated_df.equals(expected_positive_annotated_df)
    assert negative_annotated_df.equals(expected_negative_annotated_df)


@pytest.mark.parametrize(
    "allele_names",
    [
        np.array(["HLA__A0107", "HLA__A0107", "H2__Kb", "H2__Kb"]),
        np.array(["HLA__A0107"]),
    ],
)
def test_rescale_ma_positive_data(
    allele_names: np.ndarray,
    mhc1_nn_align_dataset: Dataset,
) -> None:
    """Check _rescale_ma_positive_data works as expected."""
    raw_predictions = np.array([1.0, 2.0, 1.0, 2.0])

    allele2value = {
        allele: mhc1_nn_align_dataset.inputs["allele"]
        .transform(pl.DataFrame({"allele": [allele]}))["allele"]
        .to_numpy()
        for allele in allele_names
    }
    alleles = (
        mhc1_nn_align_dataset.inputs["allele"]
        .transform(pl.DataFrame({"allele": allele_names}))["allele"]
        .to_numpy()
    )

    allele2rescaling_params = {
        "HLA__A0107": {
            "p_tilde": 1.0,
            "sigma_tilde": 2.0,
        },
        "H2__Kb": {
            "p_tilde": 3.0,
            "sigma_tilde": 4.0,
        },
    }

    if len(allele_names) == 1:
        match_msg = "Length of the arrays for raw predictions and allele values are not equal."
        with pytest.raises(ValueError, match=match_msg):
            _rescale_ma_positive_data(
                raw_predictions=raw_predictions,
                alleles=alleles,
                allele2value=allele2value,
                allele2rescaling_params=allele2rescaling_params,
            )

    else:
        match_msg = (
            "No rescaling parameters provided. Please provide 'allele2value' and "
            "'allele2rescaling_params' to use rescale MA positive data."
        )
        with pytest.raises(ValueError, match=match_msg):
            _rescale_ma_positive_data(
                raw_predictions=raw_predictions,
                alleles=alleles,
                allele2value={},
                allele2rescaling_params={},
            )

        expected_rescaled_predictions = np.array(
            [(1.0 - 1.0) / 2.0, (2.0 - 1.0) / 2.0, (1.0 - 3.0) / 4.0, (2.0 - 3.0) / 4.0]
        )

        rescaled_predictions = _rescale_ma_positive_data(
            raw_predictions=raw_predictions,
            alleles=alleles,
            allele2value=allele2value,
            allele2rescaling_params=allele2rescaling_params,
        )

        assert np.allclose(rescaled_predictions, expected_rescaled_predictions)
