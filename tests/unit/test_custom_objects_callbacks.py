"""Unit tests related to bench_mhc/custom_objects/callbacks.py."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint as PytorchModelCheckpoint

from bench_mhc.custom_objects.callbacks import ModelCheckpoint
from bench_mhc.model.mhc1_nn_align import MHC1NNAlignLightningModule


@pytest.fixture
def trainer() -> MagicMock:
    """Create a mock trainer."""
    trainer = MagicMock()
    trainer.current_epoch = 0
    trainer._should_skip_saving_checkpoint.return_value = False
    return trainer


@pytest.fixture
def pl_module() -> MagicMock:
    """Create a mock lightning module."""
    module = MagicMock(spec=MHC1NNAlignLightningModule)
    module.hparams = {}

    return module


def test_model_checkpoint_init_min_mode() -> None:
    """Test initialization with min mode."""
    callback = ModelCheckpoint(monitor="val_loss", mode="min")
    assert callback.monitor_op == torch.lt
    assert callback.best_epoch == 0


def test_model_checkpoint_init_max_mode() -> None:
    """Test initialization with max mode."""
    callback = ModelCheckpoint(monitor="val_acc", mode="max")
    assert callback.monitor_op == torch.gt
    assert callback.best_epoch == 0


@pytest.mark.parametrize(
    ("monitor", "mode", "current_score", "new_score", "expected_best_epoch"),
    [
        ("val_loss", "min", 0.5, 0.4, 1),
        ("val_loss", "min", 0.5, 0.6, 0),
        ("val_acc", "max", 0.8, 0.9, 1),
        ("val_acc", "max", 0.8, 0.7, 0),
    ],
)
@pytest.mark.parametrize("save_on_train_epoch_end", [True, False])
def test_model_checkpoint(
    trainer: MagicMock,
    pl_module: MagicMock,
    monitor: str,
    mode: str,
    current_score: float,
    new_score: float,
    expected_best_epoch: int,
    save_on_train_epoch_end: bool,
) -> None:
    """Test custom ModelCheckpoint."""
    callback = ModelCheckpoint(
        monitor=monitor, mode=mode, save_on_train_epoch_end=save_on_train_epoch_end
    )
    callback.current_score = torch.tensor(current_score)
    callback.best_model_score = torch.tensor(current_score)

    callback.on_train_epoch_end(trainer, pl_module)

    assert pl_module.hparams["checkpoint"] == {
        "best_epoch": 0,
        "last_epoch": 0,
        "best_metric_value": current_score,
        "monitored_metric": monitor,
    }

    should_improve = expected_best_epoch == 1

    trainer.current_epoch = 1
    callback.current_score = torch.tensor(new_score)
    callback.best_model_score = torch.tensor(new_score if should_improve else current_score)

    with (
        patch(
            "bench_mhc.custom_objects.callbacks.PytorchModelCheckpoint.on_train_epoch_end",
            return_value=lambda *args: PytorchModelCheckpoint.on_train_epoch_end(*args),
        ) as on_train_epoch_end_spy,
    ):
        callback.on_train_epoch_end(trainer, pl_module)

    on_train_epoch_end_spy.assert_called_once_with(trainer, pl_module)

    expected_previous_best_model_score = torch.tensor(
        new_score if should_improve else current_score
    )
    assert callback.best_epoch == expected_best_epoch
    assert callback.previous_best_model_score == expected_previous_best_model_score

    assert pl_module.hparams["checkpoint"] == {
        "best_epoch": expected_best_epoch,
        "last_epoch": 1,
        "best_metric_value": new_score if should_improve else current_score,
        "monitored_metric": monitor,
    }


def test_model_checkpoint_state_dict() -> None:
    """Test the state_dict method."""
    callback = ModelCheckpoint(monitor="val_loss", mode="min")
    callback.current_score = torch.tensor([0.5])
    callback.best_model_score = torch.tensor([0.4])
    callback.best_epoch = 2
    state = callback.state_dict()

    assert "best_epoch" in state
    assert state["best_epoch"] == 2

    for key in super(ModelCheckpoint, callback).state_dict().keys():
        assert key in state
