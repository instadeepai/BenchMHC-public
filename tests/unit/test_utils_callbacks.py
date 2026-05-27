"""Unit Tests related to bench_mhc/utils/callbacks.py."""

from pathlib import Path
from typing import Any

import pytest
from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelSummary

from bench_mhc.constants import ROOT_DIRECTORY
from bench_mhc.custom_objects.callbacks import ModelCheckpoint
from bench_mhc.utils.callbacks import get_early_stopping_callback
from bench_mhc.utils.callbacks import load_callbacks_from_config
from bench_mhc.utils.callbacks import maybe_add_model_checkpoint_callback
from bench_mhc.utils.callbacks import override_callbacks


class TestLoadCallbacksFromConfig:
    """Module to test use cases of load_callbacks_from_config."""

    def test_load_model_checkpoint_with_args(self) -> None:
        """Test loading multiple callbacks with arguments."""
        config: list[dict[str, int | str]] = [
            {
                "class_name": "ModelCheckpoint",
                "dirpath": "checkpoints",
                "filename": "best-{epoch}-{val_loss:.2f}",
            },
            {
                "class_name": "EarlyStopping",
                "monitor": "val_loss",
                "patience": 5,
            },
        ]

        callbacks = load_callbacks_from_config(config)

        assert len(callbacks) == 2
        model_callback = callbacks[0]
        assert isinstance(model_callback, ModelCheckpoint)
        assert model_callback.dirpath == str(ROOT_DIRECTORY / "checkpoints")
        assert model_callback.filename == "best-{epoch}-{val_loss:.2f}"

        early_stopping_callback = callbacks[1]
        assert isinstance(early_stopping_callback, EarlyStopping)
        assert early_stopping_callback.monitor == "val_loss"
        assert early_stopping_callback.patience == 5

    def test_load_learning_rate_monitor_no_args(self) -> None:
        """Test loading LearningRateMonitor with no arguments."""
        config = [{"class_name": "LearningRateMonitor"}]
        callbacks = load_callbacks_from_config(config)
        assert len(callbacks) == 1
        callback = callbacks[0]
        assert isinstance(callback, LearningRateMonitor)

    def test_load_nonexistent_callback(self) -> None:
        """Test loading a callback that doesn't exist raises AttributeError."""
        config = [{"class_name": "NonExistentCallback"}]
        with pytest.raises(AttributeError):
            load_callbacks_from_config(config)

    def test_load_invalid_config(self) -> None:
        """Test loading a callback with an invalid configuration type raises TypeError."""
        config = [{"class_name": "EarlyStopping", "dirpath": "invalid_config"}]
        with pytest.raises(Exception, match="Error instantiating callback"):
            load_callbacks_from_config(config)

    def test_load_model_checkpoint_various_args(self) -> None:
        """Test loading ModelCheckpoint with a variety of argument types."""
        config = [
            {
                "class_name": "ModelCheckpoint",
                "dirpath": ".",
                "monitor": "val/loss",
                "save_top_k": 3,
                "every_n_epochs": 2,
                "save_last": True,
                "filename": "epoch={epoch}-val_loss={val/loss:.2f}",
            }
        ]
        callbacks = load_callbacks_from_config(config)
        assert len(callbacks) == 1
        callback = callbacks[0]
        assert isinstance(callback, ModelCheckpoint)
        assert callback.dirpath == str(ROOT_DIRECTORY)
        assert callback.monitor == "val/loss"
        assert callback.save_top_k == 3
        assert callback.every_n_epochs == 2
        assert callback.save_last is True
        assert callback.filename == "epoch={epoch}-val_loss={val/loss:.2f}"


def test_maybe_add_model_checkpoint_callback(tmp_path: Path) -> None:
    """Ensure maybe_add_model_checkpoint_callback works as expected.

    We test the following cases:
    - the configuration is empty
    - the configuration does not contain ModelCheckpoint callback
    - the configuration contains a ModelCheckpoint callback with missing default parameters
    """
    model_path = tmp_path / "test_maybe_add_model_checkpoint_callback_model_path"
    configuration: dict[str, Any] = {"training": {}}
    maybe_add_model_checkpoint_callback(configuration, model_path)

    expected_configuration = {
        "callbacks": [
            {
                "class_name": "ModelCheckpoint",
                "dirpath": str(model_path / "checkpoints"),
                "enable_version_counter": False,
                "filename": "best",
                "save_last": True,
                "monitor": "val/loss",
                "mode": "min",
                "verbose": True,
            }
        ]
    }
    assert configuration["training"] == expected_configuration

    maybe_add_model_checkpoint_callback(configuration, model_path)
    assert configuration["training"] == expected_configuration

    configuration = {"training": {"callbacks": [{"class_name": "EarlyStopping"}]}}

    maybe_add_model_checkpoint_callback(configuration, model_path)
    assert configuration["training"] == {
        "callbacks": [
            {"class_name": "EarlyStopping"},
            {
                "class_name": "ModelCheckpoint",
                "dirpath": str(model_path / "checkpoints"),
                "enable_version_counter": False,
                "filename": "best",
                "save_last": True,
                "monitor": "val/loss",
                "mode": "min",
                "verbose": True,
            },
        ]
    }

    configuration = {
        "training": {
            "callbacks": [
                {"class_name": "ModelCheckpoint", "monitor": "val/accuracy", "mode": "max"}
            ]
        }
    }
    maybe_add_model_checkpoint_callback(configuration, model_path)
    assert configuration["training"] == {
        "callbacks": [
            {
                "class_name": "ModelCheckpoint",
                "dirpath": str(model_path / "checkpoints"),
                "enable_version_counter": False,
                "filename": "best",
                "save_last": True,
                "monitor": "val/accuracy",
                "mode": "max",
                "verbose": True,
            },
        ]
    }


def test_get_early_stopping_callback() -> None:
    """Test getting the EarlyStopping callback from the list of callbacks."""
    callbacks: list[Callback] = [
        ModelCheckpoint(monitor="val_loss"),
    ]
    assert get_early_stopping_callback(callbacks) is None

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5)
    callbacks.append(early_stopping_callback)

    assert get_early_stopping_callback(callbacks) == early_stopping_callback


def test_override_callbacks() -> None:
    """Test overriding callbacks."""
    callbacks = [
        ModelCheckpoint(monitor="val_loss"),
        EarlyStopping(monitor="val_loss", patience=5),
        ModelSummary(),
    ]
    new_callbacks: list[Callback] = [EarlyStopping(monitor="val_loss", patience=18)]

    updated_callbacks = override_callbacks(callbacks, new_callbacks, (ModelSummary,))
    assert len(updated_callbacks) == 2
    assert updated_callbacks[0] == callbacks[0]
    assert updated_callbacks[1] == new_callbacks[0]
