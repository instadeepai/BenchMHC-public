"""Integration tests for the command lines in bench_mhc/cli/predict_command.py."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import polars as pl
import polars.selectors as cs
import polars.testing as pl_testing
import pytest
import rich_click as click
from click.testing import CliRunner
from lightning import Trainer

from bench_mhc.__main__ import main
from bench_mhc.cli.calibrate_command import calibrate
from bench_mhc.cli.predict_command import predict
from bench_mhc.cli.train_command import train
from bench_mhc.constants import SEPARATOR
from bench_mhc.utils.model import get_directory_name


def load_from_hash_fn(hash_: str, cache_directory: Path) -> pl.DataFrame:
    """Mock the load_from_hash function."""
    return pl.read_parquet(cache_directory / f"{hash_}.parquet")


@pytest.fixture
def dataset_path(mhc1_dataframe: pl.DataFrame, tmp_path: Path) -> str:
    """Get the training/validation/predict file path."""
    file_path = tmp_path / "training_validation_predict.csv"

    mhc1_dataframe.write_csv(file_path)

    return str(file_path)


@pytest.mark.parametrize(
    (
        "num_workers",
        "gpu_config",
        "w_output_file_path",
        "predictions_already_exist",
        "w_multiple_models",
        "models_provided_as",
    ),
    [
        (
            None,
            {"gpus": "1", "expected_devices": [1], "expected_accelerator": "auto"},
            True,
            False,
            True,
            "multiple_flags",
        ),
        (
            2,
            {"gpus": "0", "expected_devices": [0], "expected_accelerator": "auto"},
            False,
            False,
            True,
            "comma_separated",
        ),
        (
            None,
            {"gpus": None, "expected_devices": "auto", "expected_accelerator": "cpu"},
            False,
            True,
            True,
            "text_file",
        ),
        (
            None,
            {"gpus": "1", "expected_devices": [1], "expected_accelerator": "auto"},
            True,
            False,
            False,
            "multiple_flags",
        ),
        (
            2,
            {"gpus": "0", "expected_devices": [0], "expected_accelerator": "auto"},
            False,
            False,
            False,
            "comma_separated",
        ),
        (
            None,
            {"gpus": None, "expected_devices": "auto", "expected_accelerator": "cpu"},
            False,
            True,
            False,
            "text_file",
        ),
    ],
)
def test_cli_predict(
    num_workers: int | None,
    gpu_config: dict[str, Any],
    w_output_file_path: bool,
    predictions_already_exist: bool,
    w_multiple_models: bool,
    models_provided_as: str,
    model_directory: Path,
    cache_directory: Path,
    configuration_file_path: str,
    configuration_file_path_hit_only: str,
    dataset_path: str,
    tmp_path: Path,
) -> None:
    """Test the predict command.

    We first train dummy models to produce training artifacts. Then we use these training
    artifacts to test the predict command line.

    We check the following cases:
        - predictions with single model
        - predictions with multiple models where one model has only the `hit` output and the other
            has `hit` and `binding_affinity`
        - models can be provided as multiple flags, comma-separated list, or a text file
        - `output_file_path` not provided
        - predictions already exist in the dataset
        - GPU configuration is properly handled
        - predictions with `batch_size` provided/not provided
        - predictions with different `num_workers` provided
        - 9 mers core columns are properly handled
    """
    models_paths_txt_file = tmp_path / "models_paths.txt"
    experiment_name = "test_model"
    training_parameters = [
        "--experiment_name",
        experiment_name,
        "--training_path",
        dataset_path,
        "--validation_path",
        dataset_path,
        "--random_seed",
        str(42),
    ]

    model_name = [get_directory_name(experiment_name)]
    cfg_path_list = [configuration_file_path]
    if w_multiple_models:
        model_name.append(get_directory_name(experiment_name))
        cfg_path_list.append(configuration_file_path_hit_only)

    predictions_column_prefix = "test_cli_predict"
    base_predict_parameters = [
        "--dataset_path",
        dataset_path,
        "--predictions_column_prefix",
        predictions_column_prefix,
    ] + (["--gpus", gpu_config["gpus"]] if gpu_config["gpus"] is not None else [])

    ensemble_predict_parameters = base_predict_parameters.copy()
    if num_workers is not None:
        ensemble_predict_parameters.extend(["--batch_size", "2", "--num_workers", str(num_workers)])

    expected_output_file_path = Path(dataset_path)
    if w_output_file_path:
        expected_output_file_path = tmp_path / "custom_output_file.csv"
        ensemble_predict_parameters.extend(["--output_file_path", str(expected_output_file_path)])

    model_paths = []

    runner = CliRunner()

    with (
        patch("bench_mhc.cli.train.MODEL_DIRECTORY", model_directory),
        patch("bench_mhc.model.mhc1_nn_align.MODEL_DIRECTORY", model_directory),
        patch("bench_mhc.cli.train.CACHE_DIRECTORY", cache_directory),
        patch("bench_mhc.cli.train.get_directory_name", side_effect=model_name),
    ):
        # Train the models to produce training artifacts
        for model_name_, cfg_file_path in zip(model_name, cfg_path_list, strict=False):
            training_parameters_ = training_parameters + [
                "--configuration_file_path",
                cfg_file_path,
            ]
            result = runner.invoke(train, training_parameters_)
            assert result.exit_code == 0

            model_path = model_directory / model_name_
            if models_provided_as == "multiple_flags":
                ensemble_predict_parameters.extend(["--model_path", str(model_path)])
            elif models_provided_as == "text_file":
                with open(models_paths_txt_file, "a") as f:
                    f.write(str(model_path) + "\n")

            model_paths.append(model_path)

        if models_provided_as == "text_file":
            ensemble_predict_parameters.extend(["--model_path", str(models_paths_txt_file)])
        elif models_provided_as == "comma_separated":
            ensemble_predict_parameters.extend(
                ["--model_path", ",".join([str(model_path) for model_path in model_paths])]
            )

        # Run predictions individually with single model and average them out of predict
        single_model_predictions = {
            "hit": np.zeros(6),
            "binding_affinity": np.zeros(6),
        }

        with patch(
            "bench_mhc.dataset.cache.DatasetCache.load_from_hash",
            side_effect=lambda hash_: load_from_hash_fn(hash_, cache_directory),
        ) as load_from_hash_spy:
            for model_path in model_paths:
                trainer_mock = Trainer(
                    devices="auto",
                    accelerator="cpu",
                    default_root_dir=model_path,
                    deterministic=True,
                )
                assert result.exit_code == 0

                output_file_path_ = tmp_path / model_path.with_suffix(".csv").name

                with (
                    patch(
                        "bench_mhc.cli.predict.Trainer", return_value=trainer_mock
                    ) as trainer_mock_spy,
                    patch("bench_mhc.cli.predict.CACHE_DIRECTORY", cache_directory),
                ):
                    result = runner.invoke(
                        predict,
                        base_predict_parameters
                        + [
                            "--model_path",
                            str(model_path),
                            "--output_file_path",
                            str(output_file_path_),
                        ],
                    )

                    assert result.exit_code == 0, f"Prediction failed: {result.output}"
                    # Verify that the Trainer was called with correct GPU configuration during
                    # prediction
                    trainer_mock_spy.assert_called_once_with(
                        devices=gpu_config["expected_devices"],
                        accelerator=gpu_config["expected_accelerator"],
                        default_root_dir=model_path,
                        deterministic=True,
                    )

                lf = pl.scan_csv(output_file_path_)
                single_model_predictions["hit"] += (
                    lf.select(cs.ends_with("_hit")).collect().to_numpy().squeeze()
                )
                if any(col.endswith("_binding_affinity") for col in lf.collect_schema().names()):
                    single_model_predictions["binding_affinity"] += (
                        lf.select(cs.ends_with("_binding_affinity")).collect().to_numpy().squeeze()
                    )

            if len(model_paths) > 1:
                single_model_predictions["hit"] /= 2

            if predictions_already_exist and not w_output_file_path:
                dataset_lf = pl.scan_csv(dataset_path).with_columns(
                    pl.lit(np.random.rand(6)).alias(
                        f"{predictions_column_prefix}{SEPARATOR}binding_affinity"
                    ),
                    pl.lit(np.random.rand(6)).alias(f"{predictions_column_prefix}{SEPARATOR}hit"),
                )
                dataset_lf.collect().write_csv(dataset_path)

            # Run predictions on the ensemble with predict
            # We mock the devices and accelerator instead of the Trainer to avoid using GPUs since
            # we already checked that the GPU configuration is properly handled in the single model
            # predictions
            with (
                patch(
                    "bench_mhc.cli.predict.get_devices_and_accelerator",
                    return_value=("auto", "auto"),
                ),
                patch("bench_mhc.cli.predict.CACHE_DIRECTORY", cache_directory),
            ):
                result = runner.invoke(predict, ensemble_predict_parameters)

            assert result.exit_code == 0
            assert expected_output_file_path.exists()

            lf_predictions = pl.scan_csv(expected_output_file_path)
            prediction_column_names = {
                f"{predictions_column_prefix}{SEPARATOR}{output_name}"
                for output_name in ["hit", "binding_affinity"]
            }

            nine_mers_core_column_names = {
                f"{predictions_column_name}{column_suffix}"
                for predictions_column_name in prediction_column_names
                for column_suffix in [
                    "_selected_core_indices",
                    "_majority_core_index",
                    "_majority_core",
                ]
            }

            assert set(lf_predictions.collect_schema().names()) == {
                "peptide",
                "allele",
                "hit",
                "binding_affinity",
                *prediction_column_names,
                *nine_mers_core_column_names,
            }
            # Ensure there are no NaN values in prediction columns
            assert (
                not lf_predictions.select(
                    pl.sum_horizontal(pl.col(prediction_column_names).is_nan().any())
                )
                .collect()
                .item()
            )

            # Manual average from the 2 single models predictions files
            expected_df = pl.DataFrame(single_model_predictions).select(
                pl.all().name.prefix(f"{predictions_column_prefix}{SEPARATOR}")
            )
            # Averaged done from multi-models predict command line
            actual_df = lf_predictions.select(prediction_column_names).collect()

            # Compare predict ensemble gives same results as predictions of single models averaged
            pl_testing.assert_frame_equal(
                expected_df,
                actual_df,
                check_row_order=False,
                check_column_order=False,
                check_dtypes=False,
            )

            # Called 2 times for each model:
            # - once at the first single model predict call
            # - a second time at the ensemble predict call
            assert load_from_hash_spy.call_count == (len(model_paths) * 2)

            # Check that the 9-mers core columns are not NaN
            df_nine_mers_core = lf_predictions.select(nine_mers_core_column_names).collect()

            if len(model_paths) > 1:
                # Assert that the number of 9-mers core selected for the `hit` output is 2
                # Since we have 2 models that can predict the `hit` output
                assert df_nine_mers_core.select(
                    (cs.ends_with("_selected_core_indices") & cs.contains("hit"))
                    .str.count_matches(";")
                    .sum()
                    == pl.len()
                ).item()
            else:
                # Assert that the number of 9-mers core selected for the `hit` output is 1
                # Since we have only 1 model that can predict the `hit` output
                # Hence we check that the selected core index column is integer column
                # ranging from 0 to 9
                assert df_nine_mers_core.select(
                    (cs.ends_with("_selected_core_indices") & cs.contains("hit"))
                    .is_in(range(10))
                    .all()
                ).item()

            # Assert that the number of 9-mers core selected for the `binding_affinity` output is 1
            # Since we have 1 model that can predict the `binding_affinity` output
            # Hence we check that the selected core index column is integer column
            # ranging from 0 to 9
            assert df_nine_mers_core.select(
                pl.all_horizontal(
                    (
                        (cs.ends_with("_selected_core_indices") & cs.contains("binding_affinity"))
                        | cs.contains("majority_core_index")
                    )
                    .is_in(range(10))
                    .all()
                )
            ).item()

            # Assert that the 9-mers core columns are 9-mers
            assert df_nine_mers_core.select(
                pl.all_horizontal((cs.ends_with("core").str.len_bytes() == 9).all())
            ).item()


@pytest.mark.parametrize("paths_exist", [True, False])
def test_predict_missing_options(tmp_path: Path, paths_exist: bool) -> None:
    """Test the 'predict' command does not work when options are missing.

    We also tests relevant errors are raised when the file paths do not exist.

    Args:
        tmp_path: Path to the temporary test directory.
        paths_exist: Whether parameters' paths exist.
    """
    runner = CliRunner()
    parameters = ["predict"]
    model_path = tmp_path / "test_model_path"
    result = runner.invoke(main, parameters)
    assert result.exit_code != 0
    assert "Missing option '--model_path' / '-m'" in click.unstyle(result.output)

    if paths_exist:
        model_path.mkdir(parents=True, exist_ok=True)

    parameters.extend(["--model_path", str(model_path)])
    result = runner.invoke(main, parameters)
    assert result.exit_code != 0
    if paths_exist:
        assert "Missing option '--dataset_path' / '-d'" in click.unstyle(result.output)
    else:
        assert "Invalid value for '--model_path' / '-m'" in click.unstyle(result.output)

    model_path.touch()
    dataset_path = tmp_path / "dataset.csv"
    parameters.extend(["--dataset_path", str(dataset_path)])
    if paths_exist:
        dataset_path.touch()

    result = runner.invoke(main, parameters)
    assert result.exit_code != 0
    if paths_exist:
        assert "Missing option '--predictions_column_prefix' / '-p_pref'" in click.unstyle(
            result.output
        )
    else:
        assert "Invalid value for '--dataset_path' / '-d'" in click.unstyle(result.output)


@pytest.mark.parametrize("partial_calibration", [True, False])
@pytest.mark.parametrize("apply_sigmoid", [True, False])
def test_predict_with_calibrated_model(
    tmp_path: Path,
    model_directory: Path,
    cache_directory: Path,
    configuration_file_path: str,
    configuration_file_path_hit_only: str,
    configuration_file_path_with_logits_loss: str,
    configuration_file_path_hit_only_with_logits_loss: str,
    dataset_path: str,
    reference_path: str,
    allele_mapping_path: str,
    partial_calibration: bool,
    apply_sigmoid: bool,
) -> None:
    """Test the 'predict' command with a calibrated model.

    Tests both with and without BCEWithLogitsLoss (apply_sigmoid parameter).
    """
    experiment_name = "test_model"
    training_parameters = [
        "--experiment_name",
        experiment_name,
        "--training_path",
        dataset_path,
        "--validation_path",
        dataset_path,
        "--random_seed",
        str(42),
    ]

    model_name = [get_directory_name(experiment_name), get_directory_name(experiment_name)]

    if apply_sigmoid:
        cfg_path_list = [
            configuration_file_path_with_logits_loss,
            configuration_file_path_hit_only_with_logits_loss,
        ]
    else:
        cfg_path_list = [configuration_file_path, configuration_file_path_hit_only]

    model_paths = []
    runner = CliRunner()
    with patch("bench_mhc.model.mhc1_nn_align.MODEL_DIRECTORY", model_directory):
        with (
            patch("bench_mhc.cli.train.MODEL_DIRECTORY", model_directory),
            patch("bench_mhc.cli.train.CACHE_DIRECTORY", cache_directory),
            patch("bench_mhc.cli.train.get_directory_name", side_effect=model_name),
        ):
            # Train the models to produce training artifacts
            for model_name_, cfg_file_path in zip(model_name, cfg_path_list, strict=True):
                training_parameters_ = training_parameters + [
                    "--configuration_file_path",
                    cfg_file_path,
                ]
                result = runner.invoke(train, training_parameters_)
                assert result.exit_code == 0

                model_path = model_directory / model_name_
                model_paths.append(model_path)

        # Calibrate the model and run predictions on the ensemble
        percentile_rank_directory = tmp_path / "percentile_rank"
        model_paths_option = [
            "--model_path",
            str(model_paths[0]),
            "--model_path",
            str(model_paths[1]),
        ]
        calibrate_parameters = [
            *model_paths_option,
            "--reference_path",
            reference_path,
            "--output_directory",
            str(percentile_rank_directory),
            "--peptides_only",
        ]
        if partial_calibration:
            calibrate_parameters.extend(["--alleles", "HLA__B5809"])

        predictions_column_prefix = "test_cli_predict"
        predict_parameters = [
            *model_paths_option,
            "--dataset_path",
            dataset_path,
            "--predictions_column_prefix",
            predictions_column_prefix,
            "--percentile_rank_directory",
            str(percentile_rank_directory),
        ]
        with patch("bench_mhc.cli.predict.CACHE_DIRECTORY", cache_directory):
            with (
                patch("bench_mhc.cli.calibrate.ALLELE_MAPPING_PATH", allele_mapping_path),
            ):
                result = runner.invoke(calibrate, calibrate_parameters)
                assert result.exit_code == 0

            result = runner.invoke(predict, predict_parameters)
            assert result.exit_code == 0

    # Verify predictions and percentile ranks
    df = pl.read_csv(dataset_path)

    output_names = ["hit", "binding_affinity"]
    prediction_columns = {
        f"{predictions_column_prefix}{SEPARATOR}{output_name}" for output_name in output_names
    }
    nine_mers_core_columns = {
        f"{predictions_column}{suffix}"
        for suffix in ["_selected_core_indices", "_majority_core_index", "_majority_core"]
        for predictions_column in prediction_columns
    }
    percentile_rank_columns = {
        f"{predictions_column_prefix}{SEPARATOR}pctrnk{SEPARATOR}{output_name}"
        for output_name in output_names
    }
    expected_columns = [
        "allele",
        "peptide",
        "hit",
        "binding_affinity",
        *prediction_columns,
        *nine_mers_core_columns,
    ]
    if partial_calibration:
        # Ensure that the percentile rank columns are not present in the dataset
        assert set(df.columns) == set(expected_columns)
    else:
        expected_columns.extend(percentile_rank_columns)
        assert set(df.columns) == set(expected_columns)

        # Verify that percentile ranks are inversely ordered compared to predictions per allele
        for _, df_predictions in df.group_by("allele"):
            for pred_col, pctrnk_col in zip(
                sorted(prediction_columns),
                sorted(percentile_rank_columns),
                strict=True,
            ):
                # Check that the order of values in prediction columns
                # is the inverse of percentile ranks
                assert (
                    df_predictions[pred_col].rank(method="ordinal", descending=False)
                    == df_predictions[pctrnk_col].rank(method="ordinal", descending=True)
                ).all()
