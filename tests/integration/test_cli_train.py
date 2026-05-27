"""Integration tests for the command lines in bench_mhc/cli/train_command.py."""

from pathlib import Path
from unittest.mock import ANY
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import patch

import polars as pl
import pytest
import rich_click as click
from click.testing import CliRunner
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from bench_mhc.__main__ import main
from bench_mhc.cli.train import MODEL_NAME2LIGHTNING_MODULE
from bench_mhc.cli.train_command import train
from bench_mhc.custom_objects.callbacks import ModelCheckpoint
from bench_mhc.model.mhc1_nn_align import MHC1NNAlignLightningModule
from bench_mhc.utils.deconvolution import apply_deconvolution
from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.io import load_json
from bench_mhc.utils.io import load_yml
from bench_mhc.utils.io import save_yml
from bench_mhc.utils.model import get_directory_name
from bench_mhc.utils.predictions import process_outputs


@pytest.fixture
def validation_path(mhc1_dataframe: pl.DataFrame, tmp_path: Path) -> str:
    """Get the validation file path."""
    file_path = tmp_path / "validation.csv"

    mhc1_dataframe.write_csv(file_path)

    return str(file_path)


@pytest.fixture
def ma_training_data_path(tmp_path: Path, ma_dataframe: pl.DataFrame) -> str:
    """Define the file path for multi-allelic training data used in tests."""
    file_path = tmp_path / "ma_training_data.csv"
    ma_dataframe.with_columns(
        pl.col("MA_bag_identifier").alias("MA_bag_identifier_in_train_not_in_val")
    ).write_csv(file_path)

    return str(file_path)


@pytest.fixture
def ma_validation_data_path(tmp_path: Path, ma_dataframe: pl.DataFrame) -> str:
    """Define the file path for multi-allelic validation data used in tests."""
    file_path = tmp_path / "ma_validation_data.csv"
    ma_dataframe.write_csv(file_path)

    return str(file_path)


@pytest.fixture
def wrong_configuration_file_path(tmp_path: Path, configuration_file_path: str) -> str:
    """Get file path with wrong 'model.class_name' for the tests."""
    file_path = tmp_path / "wrong_configuration.yaml"

    configuration = load_yml(configuration_file_path)

    configuration["model"]["class_name"] = "WrongModel"

    save_yml(configuration, file_path)

    return str(file_path)


@pytest.mark.parametrize("random_seed", [None, 42])
@pytest.mark.parametrize("wrong_class_name", [True, False])
def test_train(
    model_directory: Path,
    cache_directory: Path,
    training_path: str,
    validation_path: str,
    configuration_file_path: str,
    wrong_configuration_file_path: str,
    random_seed: int | None,
    wrong_class_name: bool,
) -> None:
    """Check that the train command works as expected.

    We check the following cases:
    - the model is trained and its artifacts are properly saved,
    - the random seed is properly set,
    - an error is raised if an unknown class is provided for the model.
    """
    experiment_name = "test_model"
    parameters = [
        "--experiment_name",
        experiment_name,
        "--configuration_file_path",
        wrong_configuration_file_path if wrong_class_name else configuration_file_path,
        "--training_path",
        training_path,
        "--validation_path",
        validation_path,
    ]

    model_name = get_directory_name(experiment_name)

    if random_seed is not None:
        parameters.extend(["--random_seed", str(random_seed)])

    runner = CliRunner()

    with patch("bench_mhc.cli.train.MODEL_DIRECTORY", model_directory):
        model_checkpoint_kwargs = {
            "dirpath": str(model_directory / model_name / "checkpoints"),
            "filename": "best",
            "save_last": True,
            "monitor": "val/loss",
            "mode": "min",
            "verbose": True,
            "enable_version_counter": False,
        }
        model_checkpoint_callback_mock = ModelCheckpoint(**model_checkpoint_kwargs)
        early_stopping_callback_mock = EarlyStopping(
            # Use different values as the ones in the configuration to ensure we assert
            # the call from the mock and not from the return value of the mock
            patience=5,
            monitor="val/hit/average_precision",
        )

        with (
            patch("bench_mhc.cli.train.CACHE_DIRECTORY", cache_directory),
            patch("bench_mhc.model.mhc1_nn_align.MODEL_DIRECTORY", model_directory),
            patch("bench_mhc.cli.train.set_random_seed", return_value=42) as set_random_seed_spy,
            patch("bench_mhc.cli.train.get_directory_name", return_value=model_name),
            patch(
                "lightning.pytorch.callbacks.EarlyStopping",
                return_value=early_stopping_callback_mock,
            ) as early_stopping_spy,
            patch(
                "bench_mhc.utils.callbacks.ModelCheckpoint",
                return_value=model_checkpoint_callback_mock,
            ) as model_checkpoint_spy,
            patch(
                "bench_mhc.model.mhc1_nn_align.MHC1NNAlignLightningModule.save_hparams",
                autospec=True,
                side_effect=MHC1NNAlignLightningModule.save_hparams,
            ) as save_hparams_spy,
            patch(
                "bench_mhc.utils.logging.experiment.CSVLogger.log_hyperparams",
                autospec=True,
                side_effect=CSVLogger.log_hyperparams,
            ) as log_hyperparams_spy,
            patch(
                "bench_mhc.dataset.cache.DatasetCache.compute_hash",
                side_effect=[
                    "hash_value_train_inputs",
                    "hash_value_train_outputs",
                    "hash_value_train",
                    "hash_value_val",
                ],
            ) as compute_hash_spy,
        ):
            if wrong_class_name:
                match_msg = (
                    f"The provided 'WrongModel' is not a valid model class name. "
                    f"Valid model names: {format_iterable(MODEL_NAME2LIGHTNING_MODULE.keys())}."
                )
                with pytest.raises(ValueError, match=match_msg):
                    runner.invoke(train, parameters, catch_exceptions=False)

            else:
                result = runner.invoke(train, parameters, catch_exceptions=False)
                assert result.exit_code == 0, f"The command 'train' failed - {result}"

                if random_seed is not None:
                    set_random_seed_spy.assert_called_once_with(random_seed)
                else:
                    set_random_seed_spy.assert_called_once_with(None)

                model_path = model_directory / model_name
                assert model_path.exists()
                assert model_path.is_dir()
                assert {"hparams.json", "checkpoints"}.issubset(
                    {file_path.name for file_path in model_path.iterdir()}
                )

                expected_num_epochs = 4
                hparams = load_json(model_path / "hparams.json")
                assert hparams["random_seed"] == 42
                assert hparams["configuration"]["training"]["epochs"] == expected_num_epochs
                assert hparams["training_path"] == training_path
                assert hparams["validation_path"] == validation_path
                assert hparams["model_path"] == str(model_path.relative_to(model_directory))
                for key in ("configuration", "outputs"):
                    assert key in hparams
                assert "inputs" in hparams["configuration"]["variables"]

                early_stopping_spy.assert_called_once_with(monitor="val/hit/loss", patience=1)
                model_checkpoint_spy.assert_called_once_with(**model_checkpoint_kwargs)

                assert hparams["checkpoint"]["last_epoch"] == expected_num_epochs - 1
                assert hparams["checkpoint"]["monitored_metric"] == "val/loss"
                assert hparams["checkpoint"]["best_metric_value"] >= 0
                assert hparams["checkpoint"]["best_epoch"] in {0, 1, 2, 3}

                assert {
                    file_path.name for file_path in (model_path / "checkpoints").rglob("*")
                } == {"best.ckpt", "last.ckpt"}

                # save_hparams is called at module initialization, at each epoch, and at the end
                assert save_hparams_spy.call_count == 1 + expected_num_epochs + 1

                # compute_hash is called 4 times: once for train and once for val,
                # and once for the inputs and outputs variables
                assert compute_hash_spy.call_count == 4
                assert (cache_directory / "hash_value_train.parquet").exists()
                assert (cache_directory / "hash_value_val.parquet").exists()
                assert (cache_directory / "hash_value_train_inputs.json").exists()
                assert (cache_directory / "hash_value_train_outputs.json").exists()

                # log_hyperparams is called once once in the training loop initialization, once at
                # the end of each epoch and once at the end of the training
                assert log_hyperparams_spy.call_count == 1 + expected_num_epochs + 1


@pytest.mark.parametrize("is_ma_validation_path_provided", [False, True])
@pytest.mark.parametrize("deconvolution_output_name", ["hit", "not_defined_output_variable"])
@pytest.mark.parametrize(
    "deconvolution_identifier",
    [
        "MA_bag_identifier",
        "MA_bag_identifier_in_train_not_in_val",
        "not_defined_deconvolution_identifier",
    ],
)
@pytest.mark.parametrize("sa_warmup_epochs", [1, 5])
@pytest.mark.parametrize("use_prediction_score_rescaling", [True, False])
@pytest.mark.parametrize("is_reference_path_provided", [False, True])
@patch(
    "bench_mhc.cli.train.apply_deconvolution",
    side_effect=apply_deconvolution,
)
@patch(
    "bench_mhc.cli.train.process_outputs",
    side_effect=process_outputs,
)
def test_train_iterative_annotation(
    process_outputs_spy: MagicMock,
    apply_deconvolution_spy: MagicMock,
    model_directory: Path,
    cache_directory: Path,
    training_path: str,
    validation_path: str,
    ma_training_data_path: str,
    ma_validation_data_path: str,
    configuration_file_path: str,
    is_ma_validation_path_provided: bool,
    deconvolution_output_name: str,
    deconvolution_identifier: str,
    sa_warmup_epochs: int,
    use_prediction_score_rescaling: bool,
    is_reference_path_provided: bool,
    reference_path: str,
) -> None:
    """Check that the train command works when iterative annotation is used.

    We check the following cases:
    - an error is raised if wrong set of arguments is provided,
    - the random seed is properly set,
    - the model is trained and its artifacts are properly saved,
    - predictions and deconvolution are properly applied,
    - prediction score rescaling is properly applied when `use_prediction_score_rescaling`,
    - early stopping properly stops the training.
    """
    experiment_name = "test_model_iterative_annotation"
    random_seed = 42

    parameters = [
        "--experiment_name",
        experiment_name,
        "--configuration_file_path",
        configuration_file_path,
        "--training_path",
        training_path,
        "--validation_path",
        validation_path,
        "--ma_training_path",
        ma_training_data_path,
        "--deconvolution_output_name",
        deconvolution_output_name,
        "--deconvolution_identifier",
        deconvolution_identifier,
        "--sa_warmup_epochs",
        str(sa_warmup_epochs),
        "--batch_size",
        "16",
        "--random_seed",
        str(random_seed),
    ]

    if is_ma_validation_path_provided:
        parameters.extend(["--ma_validation_path", ma_validation_data_path])

    if use_prediction_score_rescaling:
        parameters.extend(["--use_prediction_score_rescaling"])

    if is_reference_path_provided:
        parameters.extend(["--reference_path", str(reference_path)])

    model_name = get_directory_name(experiment_name)

    # Mock EarlyStopping callback to stop the training after 2 epochs
    early_stopping_callback_mock = MagicMock()
    early_stopping_callback_mock.monitor = "val/loss"
    early_stopping_callback_mock.mode = "min"
    early_stopping_callback_mock.best_score = 0.5
    type(early_stopping_callback_mock).stopped_epoch = PropertyMock(side_effect=[0, 0, 2, 2])

    runner = CliRunner()
    with (
        patch("bench_mhc.cli.train.MODEL_DIRECTORY", model_directory),
        patch("bench_mhc.model.mhc1_nn_align.MODEL_DIRECTORY", model_directory),
        patch("bench_mhc.cli.train.CACHE_DIRECTORY", cache_directory),
        patch("bench_mhc.cli.train.set_random_seed", return_value=42) as set_random_seed_spy,
        patch("bench_mhc.cli.train.get_directory_name", return_value=model_name),
        patch(
            "bench_mhc.cli.train.get_early_stopping_callback",
            return_value=early_stopping_callback_mock,
        ),
        patch(
            "bench_mhc.model.mhc1_nn_align.MHC1NNAlignLightningModule.save_hparams",
            autospec=True,
            side_effect=MHC1NNAlignLightningModule.save_hparams,
        ) as save_hparams_spy,
        patch(
            "bench_mhc.utils.logging.experiment.CSVLogger.log_hyperparams",
            autospec=True,
            side_effect=CSVLogger.log_hyperparams,
        ) as log_hyperparams_spy,
        patch(
            "bench_mhc.dataset.cache.DatasetCache.compute_hash",
            side_effect=[
                "hash_value_train_inputs",
                "hash_value_train_outputs",
                "hash_value_train",
                "hash_value_val",
            ],
        ) as compute_hash_spy,
    ):
        if sa_warmup_epochs > 3:
            match_msg = (
                "Iterative annotation: 'training.epochs=4' but it must be greater than the number "
                f"of epochs for SA warmup training: {sa_warmup_epochs}."
            )
            with pytest.raises(ValueError, match=match_msg):
                runner.invoke(train, parameters, catch_exceptions=False)

        elif deconvolution_output_name == "not_defined_output_variable":
            match_msg = (
                f"Iterative annotation: 'deconvolution_output_name={deconvolution_output_name}' is "
                f"not available in model outputs: {format_iterable(['hit', 'binding_affinity'])}."
            )
            with pytest.raises(ValueError, match=match_msg):
                runner.invoke(train, parameters, catch_exceptions=False)

        elif deconvolution_identifier == "not_defined_deconvolution_identifier":
            match_msg = (
                f"Iterative annotation: 'deconvolution_identifier={deconvolution_identifier}' is "
                f"not available in provided MA training dataset: {ma_training_data_path}."
            )
            with pytest.raises(ValueError, match=match_msg):
                runner.invoke(train, parameters, catch_exceptions=False)

        elif (
            deconvolution_identifier == "MA_bag_identifier_in_train_not_in_val"
            and is_ma_validation_path_provided
        ):
            match_msg = (
                f"Iterative annotation: 'deconvolution_identifier={deconvolution_identifier}' is "
                f"not available in provided MA validation dataset: {ma_validation_data_path}."
            )
            with pytest.raises(ValueError, match=match_msg):
                runner.invoke(train, parameters, catch_exceptions=False)

        elif use_prediction_score_rescaling and not is_reference_path_provided:
            match_msg = (
                "Iterative annotation: You must provide 'reference_path' to use prediction score "
                "rescaling."
            )
            with pytest.raises(ValueError, match=match_msg):
                runner.invoke(train, parameters, catch_exceptions=False)

        else:
            result = runner.invoke(train, parameters, catch_exceptions=False)
            assert result.exit_code == 0, f"The command 'train' failed - {result}"

            set_random_seed_spy.assert_called_once_with(random_seed)

            num_alleles = (
                pl.scan_csv(ma_training_data_path)
                .select(pl.col("allele").n_unique())
                .collect()
                .item()
            )

            expected_num_iterative_annotation_epochs = 2
            process_outputs_expected_call_count = expected_num_iterative_annotation_epochs
            apply_deconvolution_spy_expected_call_count = expected_num_iterative_annotation_epochs
            if use_prediction_score_rescaling:
                process_outputs_expected_call_count += (
                    expected_num_iterative_annotation_epochs * num_alleles
                )

            if is_ma_validation_path_provided:
                process_outputs_expected_call_count += expected_num_iterative_annotation_epochs
                apply_deconvolution_spy_expected_call_count += (
                    expected_num_iterative_annotation_epochs
                )

            assert process_outputs_spy.call_count == process_outputs_expected_call_count

            assert apply_deconvolution_spy.call_count == apply_deconvolution_spy_expected_call_count

            model_path = model_directory / model_name
            assert model_path.exists()
            assert model_path.is_dir()
            assert {"hparams.json", "checkpoints"}.issubset(
                {file_path.name for file_path in model_path.iterdir()}
            )
            hparams = load_json(model_path / "hparams.json")
            assert hparams["random_seed"] == random_seed
            assert hparams["configuration"]["training"]["epochs"] == 4
            assert hparams["training_path"] == training_path
            assert hparams["validation_path"] == validation_path
            assert hparams["model_path"] == str(model_path.relative_to(model_directory))
            for key in ("configuration", "outputs"):
                assert key in hparams
            assert "inputs" in hparams["configuration"]["variables"]

            assert hparams["checkpoint"]["last_epoch"] == (
                expected_num_iterative_annotation_epochs + sa_warmup_epochs - 1
            )

            # For SA training phase, save_hparams is called at module initialization, at each epoch,
            # and at the end of the training phase.
            # For SA + MA training phase, save_hparams is called twice (end of epoch and end of
            # training) per epoch.
            assert save_hparams_spy.call_count == (
                (1 + sa_warmup_epochs + 1) + expected_num_iterative_annotation_epochs * 2
            )

            # For SA training phase, log_hyperparams is called once in the training loop
            # initialization, once at the end of each epoch, and once at the end of the training.
            # For SA + MA training phase, log_hyperparams is called for each epoch:
            # - in the prediction loop initialization for MA training positive data
            # - in the prediction loop initialization for MA validation positive data (if provided),
            # - in the prediction loop initialization for reference data (if used),
            # - in the training loop initialization for SA + MA data,
            # - at the end of the epoch on SA + MA data,
            # - at the end of the training (hence each epoch).
            assert log_hyperparams_spy.call_count == (
                1
                + sa_warmup_epochs
                + 1
                + expected_num_iterative_annotation_epochs
                * (
                    1
                    + int(is_ma_validation_path_provided)
                    + (num_alleles * int(use_prediction_score_rescaling))
                    + 3
                )
            )

            assert compute_hash_spy.call_count == 4
            assert (cache_directory / "hash_value_train.parquet").exists()
            assert (cache_directory / "hash_value_val.parquet").exists()
            assert (cache_directory / "hash_value_train_inputs.json").exists()
            assert (cache_directory / "hash_value_train_outputs.json").exists()

            expected_iterative_annotation_metadata = {
                "epochs": 3,
                "deconvolution_identifier": deconvolution_identifier,
                "deconvolution_output_name": deconvolution_output_name,
                "ma_training_path": ma_training_data_path,
                "ma_validation_path": ma_validation_data_path
                if is_ma_validation_path_provided
                else None,
            }

            transformed_data_directory = cache_directory / model_name / "transformed_data"

            if use_prediction_score_rescaling and reference_path is not None:
                assert "predict.parquet" in {
                    file_path.name
                    for file_path in (
                        transformed_data_directory / "iterative_annotation" / "reference"
                    ).iterdir()
                }
                expected_iterative_annotation_metadata["prediction_score_rescaling"] = {
                    "reference_path": reference_path,
                    "w_shift_parameter": 75,
                    "z_score_threshold": 3.0,
                }

            assert hparams["iterative_annotation"] == expected_iterative_annotation_metadata

            assert (
                transformed_data_directory
                / "iterative_annotation"
                / "ma"
                / "train"
                / "positive"
                / "predict.parquet"
            ).exists()
            assert (
                transformed_data_directory
                / "iterative_annotation"
                / "ma"
                / "train"
                / "negative"
                / "predict.parquet"
            ).exists()

            if is_ma_validation_path_provided:
                assert (
                    transformed_data_directory
                    / "iterative_annotation"
                    / "ma"
                    / "val"
                    / "positive"
                    / "predict.parquet"
                ).exists()
                assert (
                    transformed_data_directory
                    / "iterative_annotation"
                    / "ma"
                    / "val"
                    / "negative"
                    / "predict.parquet"
                ).exists()

            assert {"train.parquet", "val.parquet"}.issubset(
                {
                    file_path.name
                    for file_path in (
                        transformed_data_directory / "iterative_annotation" / "sa_ma"
                    ).iterdir()
                }
            )


@pytest.mark.parametrize("paths_exist", [True, False])
def test_train_missing_options(tmp_path: Path, paths_exist: bool) -> None:
    """Test the 'train' command does not work when options are missing.

    We also tests relevant errors are raised when the file paths do not exist.

    Args:
        tmp_path: Path to the temporary test directory.
        paths_exist: Whether parameters' paths exist.
    """
    runner = CliRunner()
    parameters = ["train"]
    result = runner.invoke(main, parameters)
    assert result.exit_code != 0
    assert "Missing option '--experiment_name'" in click.unstyle(result.output)

    parameters.extend(["--experiment_name", "test_model"])
    result = runner.invoke(main, parameters)
    assert result.exit_code != 0
    assert "Missing option '--configuration_file_path' / '-c'" in click.unstyle(result.output)

    configuration_file_path = tmp_path / "configuration.yml"
    parameters.extend(["--configuration_file_path", str(configuration_file_path)])
    if paths_exist:
        configuration_file_path.touch()

    result = runner.invoke(main, parameters)
    assert result.exit_code != 0
    if paths_exist:
        assert "Missing option '--training_path' / '-t'" in click.unstyle(result.output)
    else:
        assert "Invalid value for '--configuration_file_path' / '-c'" in click.unstyle(
            result.output
        )

    configuration_file_path.touch()
    training_path = tmp_path / "training.csv"
    parameters.extend(["--training_path", str(training_path)])
    if paths_exist:
        training_path.touch()

    result = runner.invoke(main, parameters)
    assert result.exit_code != 0
    if paths_exist:
        assert "Missing option '--validation_path' / '-v'" in click.unstyle(result.output)
    else:
        assert "Invalid value for '--training_path' / '-t'" in click.unstyle(result.output)

    training_path.touch()
    validation_path = tmp_path / "validation.csv"
    parameters.extend(["--validation_path", str(validation_path)])

    result = runner.invoke(main, parameters)
    assert result.exit_code != 0
    assert "Invalid value for '--validation_path' / '-v'" in click.unstyle(result.output)

    validation_path.touch()
    ma_training_path = tmp_path / "ma_training.csv"
    parameters.extend(["--ma_training_path", str(ma_training_path)])
    if paths_exist:
        ma_training_path.touch()

    result = runner.invoke(main, parameters)
    assert result.exit_code != 0
    if not paths_exist:
        assert "Invalid value for '--ma_training_path' / '-ma_t'" in click.unstyle(result.output)

    ma_training_path.touch()
    ma_validation_path = tmp_path / "ma_validation.csv"
    parameters.extend(["--ma_validation_path", str(ma_validation_path)])

    result = runner.invoke(main, parameters)
    assert result.exit_code != 0
    assert "Invalid value for '--ma_validation_path' / '-ma_v'" in click.unstyle(result.output)


@pytest.mark.parametrize(
    "gpu_config",
    [
        {"gpus": "no_field", "expected_devices": "auto", "expected_accelerator": "auto"},
        {"gpus": [2], "expected_devices": [2], "expected_accelerator": "auto"},
        {"gpus": [0, 1], "expected_devices": [0, 1], "expected_accelerator": "auto"},
        {"gpus": None, "expected_devices": "auto", "expected_accelerator": "cpu"},
    ],
)
def test_train_gpu_configuration(
    model_directory: Path,
    cache_directory: Path,
    training_path: str,
    validation_path: str,
    configuration_file_path: str,
    gpu_config: dict,
) -> None:
    """Test that GPU configuration works correctly.

    We test the following GPU configurations:
    - gpus: "auto" - Uses all available GPUs
    - gpus: [2] - Uses GPU with index 2
    - gpus: [0, 1] - Uses specific GPU indices
    - gpus: null - Falls back to CPU training
    """
    experiment_name = "test_model"
    parameters = [
        "--experiment_name",
        experiment_name,
        "--configuration_file_path",
        configuration_file_path,
        "--training_path",
        training_path,
        "--validation_path",
        validation_path,
    ]

    model_name = get_directory_name(experiment_name)

    configuration = load_yml(configuration_file_path)
    if gpu_config["gpus"] != "no_field":
        # Modify configuration to include GPU settings
        configuration["training"]["gpus"] = gpu_config["gpus"]
    else:
        del configuration["training"]["gpus"]

    save_yml(configuration, configuration_file_path)

    runner = CliRunner()

    with (
        patch("bench_mhc.cli.train.MODEL_DIRECTORY", model_directory),
        patch("bench_mhc.cli.train.CACHE_DIRECTORY", cache_directory),
        patch("bench_mhc.model.mhc1_nn_align.MODEL_DIRECTORY", model_directory),
        patch("bench_mhc.cli.train.set_random_seed", return_value=42),
        patch("bench_mhc.cli.train.get_directory_name", return_value=model_name),
        patch("bench_mhc.cli.train.Trainer") as trainer_mock_spy,
        patch("bench_mhc.model.mhc1_nn_align.save_json") as save_json_mock,
    ):
        result = runner.invoke(train, parameters)
        assert result.exit_code == 0

        # Verify that the Trainer was called with correct GPU configuration
        trainer_mock_spy.assert_called_once_with(
            devices=gpu_config["expected_devices"],
            accelerator=gpu_config["expected_accelerator"],
            default_root_dir=ANY,
            deterministic=ANY,
            max_epochs=ANY,
            logger=ANY,
            log_every_n_steps=ANY,
            callbacks=ANY,
        )

        # Verify save_json was called with correct GPU configuration
        save_json_mock.assert_called()
        if gpu_config["gpus"] != "no_field":
            assert (
                save_json_mock.call_args[0][0]["configuration"]["training"]["gpus"]
                == gpu_config["gpus"]
            )
        else:
            assert "gpus" not in save_json_mock.call_args[0][0]["configuration"]["training"]
