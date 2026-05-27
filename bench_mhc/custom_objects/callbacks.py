"""Module to define custom objects relative to callbacks."""

from typing import Any

import torch
from lightning import LightningModule
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint as PytorchModelCheckpoint
from torch import Tensor

from bench_mhc.utils.logging import system

log = system.get(__name__)


class ModelCheckpoint(PytorchModelCheckpoint):
    """Extends the PyTorch Lightning ModelCheckpoint callback.

    This class inherits from `lightning.pytorch.callbacks.ModelCheckpoint` and adds functionality
    to track the epoch at which the best model score (according to the specified monitoring metric)
    was achieved.  It overrides `on_train_epoch_end` and `on_validation_end` to store the
    `current_epoch` as `best_epoch` when a new best score is reached. The `state_dict` method
    is also overridden to include `best_epoch` in the saved state.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the ModelCheckpoint callback.

        Args:
            *args: Variable length argument list to pass to the parent class.
            **kwargs: Arbitrary keyword arguments to pass to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.monitor_op = {"min": torch.lt, "max": torch.gt}[self.mode]

        self.previous_best_model_score: Tensor | None = None
        self.best_epoch = 0

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Override 'on_train_epoch_end' to maybe update the best epoch.

        The best epoch is updated if the current score is better than the previous best score.

        The checkpoint metadata is saved in the LightningModule's `hparams` attribute.

        Args:
            trainer: The PyTorch Lightning Trainer instance.
            pl_module: The LightningModule instance.
        """
        super().on_train_epoch_end(trainer, pl_module)

        if self.previous_best_model_score is None or (
            self.current_score is not None
            and self.monitor_op(self.current_score, self.previous_best_model_score)
        ):
            self.best_epoch = trainer.current_epoch
            self.previous_best_model_score = self.best_model_score

        pl_module.hparams["checkpoint"] = {
            "best_epoch": self.best_epoch,
            # To align with 0-indexing of epochs
            "last_epoch": trainer.current_epoch,
            "best_metric_value": round(self.best_model_score.item(), 4)
            if self.best_model_score
            else None,
            "monitored_metric": self.monitor,
        }

        pl_module.save_hparams()

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the model checkpoint callback, including the best epoch.

        Returns:
            A dictionary containing the state of the callback. This dictionary
            includes all the state information from the parent class, plus the
            `best_epoch`.
        """
        state_dict = super().state_dict()

        return {
            **state_dict,
            "best_epoch": self.best_epoch,
        }
