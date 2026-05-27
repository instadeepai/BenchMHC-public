"""Unit tests related to bench_mhc/utils/model.py."""

import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import lightning
import pytest
from click.testing import CliRunner

from bench_mhc.cli.train_command import train
from bench_mhc.utils.io import save_json
from bench_mhc.utils.model import get_directory_name
from bench_mhc.utils.model import load_model_from_path
from bench_mhc.utils.model import should_apply_sigmoid_for_percentile_rank


@pytest.fixture
def date() -> datetime:
    """Generate a date for tests."""
    return datetime(1991, 2, 20)


@pytest.fixture
def random_uuid() -> uuid.UUID:
    """Generate a random UUID for tests."""
    return uuid.uuid4()


@patch("bench_mhc.utils.model.datetime")
def test_get_directory_name(date: datetime, random_uuid: uuid.UUID) -> None:
    """Test get_directory_name works as expected.

    Args:
        date: The date for the test.
        random_uuid: The random UUID for the test.
    """
    with (
        patch("bench_mhc.utils.model.datetime.now", return_value=date),
        patch("bench_mhc.utils.model.uuid.uuid4", return_value=random_uuid),
    ):
        directory_name = get_directory_name("test_experiment")

    assert (
        directory_name
        == f"test_experiment__{date.strftime('%Y%m%d_%H%M%S')}__{str(random_uuid)[:3]}"
    )


@pytest.mark.parametrize("checkpoint_name", ["best", "epoch=11", "invalid_name"])
def test_load_model_from_path(
    model_directory: Path,
    cache_directory: Path,
    training_path: str,
    configuration_file_path: str,
    checkpoint_name: str,
) -> None:
    """Test loading a trained model from a specified path."""
    experiment_name = "test_load_model_from_checkpoint"
    parameters = [
        "--experiment_name",
        experiment_name,
        "--configuration_file_path",
        configuration_file_path,
        "--training_path",
        training_path,
        "--validation_path",
        training_path,
    ]

    model_name = get_directory_name(experiment_name)

    runner = CliRunner()

    with (
        patch("bench_mhc.cli.train.MODEL_DIRECTORY", model_directory),
        patch("bench_mhc.cli.train.CACHE_DIRECTORY", cache_directory),
        patch("bench_mhc.model.mhc1_nn_align.MODEL_DIRECTORY", model_directory),
        patch("bench_mhc.cli.train.get_directory_name", return_value=model_name),
    ):
        runner.invoke(train, parameters, catch_exceptions=False)

        model_path = model_directory / model_name
        if checkpoint_name != "best":
            ckpt_path_dst = model_path / "checkpoints" / f"{checkpoint_name}.ckpt"
            ckpt_path_src = model_path / "checkpoints" / "best.ckpt"
            ckpt_path_src.rename(ckpt_path_dst)

        if checkpoint_name == "invalid_name":
            with pytest.raises(FileNotFoundError, match="No best checkpoint file found at"):
                load_model_from_path(model_path)
        else:
            model = load_model_from_path(model_path)

            assert isinstance(model, lightning.pytorch.LightningModule)


def test_load_model_from_path_missing_hparams(tmp_path: Path) -> None:
    """Test that FileNotFoundError is raised if hparams.json is missing."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="hparams"):
        load_model_from_path(model_dir)


def test_load_model_from_path_missing_checkpoint(tmp_path: Path) -> None:
    """Test that FileNotFoundError is raised if best.ckpt is missing."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    hparams_path = model_dir / "hparams.json"
    hparams_path.touch()

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        load_model_from_path(model_dir)


@pytest.mark.parametrize(
    ("loss_class_name", "expected_result"),
    [
        ("BCEWithLogitsLoss", True),
        ("BCELoss", False),
        ("MSELoss", False),
        ("CrossEntropyLoss", False),
    ],
)
def test_should_apply_sigmoid_with_single_output(
    tmp_path: Path, loss_class_name: str, expected_result: bool
) -> None:
    """Test should_apply_sigmoid with different loss functions for a single output."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    hparams = {
        "configuration": {
            "variables": {
                "outputs": {
                    "hit": {
                        "class_name": "BinaryOutput",
                        "loss": {
                            "class_name": loss_class_name,
                            "reduction": "sum",
                        },
                    }
                }
            }
        }
    }
    save_json(hparams, model_dir / "hparams.json")

    assert should_apply_sigmoid_for_percentile_rank(model_dir) is expected_result


@pytest.mark.parametrize(
    "output_with_logits_loss",
    ["hit", "binding_affinity"],
)
def test_should_apply_sigmoid_with_multiple_outputs_one_with_logits_loss(
    tmp_path: Path, output_with_logits_loss: str
) -> None:
    """Test should_apply_sigmoid returns True when any output uses BCEWithLogitsLoss."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Create hparams.json with mixed losses
    hparams = {
        "configuration": {
            "variables": {
                "outputs": {
                    "hit": {
                        "class_name": "BinaryOutput",
                        "loss": {
                            "class_name": (
                                "BCEWithLogitsLoss"
                                if output_with_logits_loss == "hit"
                                else "BCELoss"
                            ),
                            "reduction": "sum",
                        },
                    },
                    "binding_affinity": {
                        "class_name": "NumericOutput",
                        "loss": {
                            "class_name": (
                                "BCEWithLogitsLoss"
                                if output_with_logits_loss == "binding_affinity"
                                else "MSELoss"
                            ),
                            "reduction": "sum",
                        },
                    },
                }
            }
        }
    }
    save_json(hparams, model_dir / "hparams.json")

    assert should_apply_sigmoid_for_percentile_rank(model_dir) is True


def test_should_apply_sigmoid_without_hparams(tmp_path: Path) -> None:
    """Test should_apply_sigmoid returns False when hparams.json is missing."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert should_apply_sigmoid_for_percentile_rank(model_dir) is False


def test_should_apply_sigmoid_with_no_logits_loss_multiple_outputs(tmp_path: Path) -> None:
    """Test that should_apply_sigmoid returns False when no output uses BCEWithLogitsLoss."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    hparams = {
        "configuration": {
            "variables": {
                "outputs": {
                    "hit": {
                        "class_name": "BinaryOutput",
                        "loss": {"class_name": "BCELoss", "reduction": "sum"},
                    },
                    "binding_affinity": {
                        "class_name": "NumericOutput",
                        "loss": {"class_name": "MSELoss", "reduction": "sum"},
                    },
                }
            }
        }
    }
    save_json(hparams, model_dir / "hparams.json")

    assert should_apply_sigmoid_for_percentile_rank(model_dir) is False
