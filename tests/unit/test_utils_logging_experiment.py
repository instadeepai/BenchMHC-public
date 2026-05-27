"""Unit tests related to bench_mhc/utils/logging/experiment.py."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from lightning.pytorch.loggers import CSVLogger

from bench_mhc.dataset.main import Dataset
from bench_mhc.model.mhc1_nn_align import MHC1NNAlignLightningModule
from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.logging.experiment import LOGGER_NAME2LOGGER_CLASS
from bench_mhc.utils.logging.experiment import get_logger


@pytest.fixture
def logger_type2configuration() -> dict[str, dict[str, Any]]:
    """Get mapping from logger type to logger configuration."""
    return {
        "csv": {"class_name": "CSVLogger"},
        "mlflow": {
            "class_name": "MLflowLogger",
            "tags": ["tag_1", "tag_2"],
        },
        "unknown": {
            "class_name": "UnknownLogger",
        },
    }


@pytest.mark.parametrize("logger_type", ["csv", "mlflow", "unknown"])
@pytest.mark.parametrize("ma_training_file", [None, "ma_train.csv"])
@pytest.mark.parametrize("ma_validation_file", [None, "ma_validation.csv"])
@pytest.mark.parametrize("experiment_name", [None, "my-experiment"])
def test_get_logger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_directory: Path,
    mhc1_nn_align_configuration: dict[str, dict[str, Any]],
    mhc1_nn_align_dataset: Dataset,
    logger_type2configuration: dict[str, dict[str, Any]],
    logger_type: str,
    ma_training_file: str | None,
    ma_validation_file: str | None,
    experiment_name: str | None,
) -> None:
    """Test get_logger works as expected."""
    if experiment_name is not None:
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", experiment_name)
    else:
        monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)

    model_path = model_directory / "test_get_logger_model"
    with (
        patch("bench_mhc.model.mhc1_nn_align.MODEL_DIRECTORY", model_directory),
        patch("bench_mhc.utils.logging.experiment.IntStepMLFlowLogger") as mlflow_logger_class_mock,
    ):
        model = MHC1NNAlignLightningModule(
            model_path=model_path,
            configuration=mhc1_nn_align_configuration,
            inputs=mhc1_nn_align_dataset.inputs,
            outputs=mhc1_nn_align_dataset.outputs,
        )

        model.configuration["training"]["logger"] = logger_type2configuration[logger_type]

        training_path, validation_path = tmp_path / "train.csv", tmp_path / "validation.csv"

        ma_training_path = None if ma_training_file is None else tmp_path / ma_training_file
        ma_validation_path = None if ma_validation_file is None else tmp_path / ma_validation_file

        match_msg = (
            f"Unknown 'training.logger.class_name' in the configuration: UnknownLogger. "
            f"Possible values: {format_iterable(LOGGER_NAME2LOGGER_CLASS)}. "
        )
        if logger_type == "unknown":
            with pytest.raises(ValueError, match=match_msg):
                _ = get_logger(
                    model=model,
                    training_path=training_path,
                    validation_path=validation_path,
                    ma_training_path=ma_training_path,
                    ma_validation_path=ma_validation_path,
                )

        else:
            if logger_type == "csv":
                csv_logger = get_logger(
                    model=model,
                    training_path=training_path,
                    validation_path=validation_path,
                    ma_training_path=ma_training_path,
                    ma_validation_path=ma_validation_path,
                )

                assert isinstance(csv_logger, CSVLogger)
                assert csv_logger.save_dir == str(model_path)

            elif logger_type == "mlflow":
                mlflow_logger_mock = mlflow_logger_class_mock.return_value
                mlflow_logger_mock.run_id = "test-run-id"

                if experiment_name is None:
                    with pytest.raises(
                        ValueError,
                        match="MLFLOW_EXPERIMENT_NAME environment variable is not set.",
                    ):
                        _ = get_logger(
                            model=model,
                            training_path=training_path,
                            validation_path=validation_path,
                            ma_training_path=ma_training_path,
                            ma_validation_path=ma_validation_path,
                        )

                else:
                    mlflow_logger = get_logger(
                        model=model,
                        training_path=training_path,
                        validation_path=validation_path,
                        ma_training_path=ma_training_path,
                        ma_validation_path=ma_validation_path,
                    )

                    assert mlflow_logger == mlflow_logger_mock
                    expected_tags = ["train", training_path.name, validation_path.name]
                    if ma_training_path is not None:
                        expected_tags.append(ma_training_path.name)
                    if ma_validation_path is not None:
                        expected_tags.append(ma_validation_path.name)
                    expected_tags.extend(["tag_1", "tag_2"])
                    mlflow_logger_class_mock.assert_called_once()
                    call_kw = mlflow_logger_class_mock.call_args[1]
                    assert call_kw["experiment_name"] == "my-experiment"
                    assert call_kw["run_name"] == model.model_path.name
                    assert set(call_kw["tags"].keys()) == set(expected_tags)
                    assert model.hparams.get("mlflow_run_id") == "test-run-id"
