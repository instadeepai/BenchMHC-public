"""Unit tests related to model/base.py."""

from pathlib import Path

import pytest
import torch.nn as nn

from bench_mhc.model.base import BaseLightningModule
from bench_mhc.utils.io import save_json


def test_base_lightning_module() -> None:
    """Test that the base lightning module can be instantiated."""

    class TempModule(BaseLightningModule):
        @property
        def model_directory(self) -> Path:
            return Path("/tmp/test")

        def _save_json(self, data: dict, path: Path) -> None:
            """Saves a dictionary to a JSON file."""
            save_json(data, path)

        def _create_model(self) -> nn.Module:
            return nn.Linear(1, 1)

    module = TempModule(
        model_path=Path("test"),
        configuration={"training": {"optimizer": {"class_name": "Adam", "lr": 0.001}}},
        inputs={},
        outputs={},
    )

    assert module.model_directory == Path("/tmp/test")
    assert isinstance(module._create_model(), nn.Linear)


def test_base_lightning_module_error() -> None:
    """Check that the instantiation fails if one of the abstract methods is not implemented."""
    with pytest.raises(TypeError) as excinfo:
        BaseLightningModule(  # type: ignore[abstract]
            model_path=Path("dummy/path"),
            configuration={"training": {"optimizer": {}}},
            inputs={},
            outputs={},
        )

    error_message = str(excinfo.value)
    assert "Can't instantiate abstract class BaseLightningModule" in error_message
    assert "_create_model" in error_message
    assert "model_directory" in error_message
    assert "_save_json" in error_message
