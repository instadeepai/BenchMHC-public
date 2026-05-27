"""Base Lightning module for all models in the framework."""

import copy
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

import lightning
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.loggers import Logger
from torch import Tensor
from torchmetrics import MetricCollection

from bench_mhc.utils.format import format_dict
from bench_mhc.utils.logging import system
from bench_mhc.utils.logging.mlflow import IntStepMLFlowLogger
from bench_mhc.variables.variables import Outputs
from bench_mhc.variables.variables import Variables

log = system.get(__name__)


class BaseLightningModule(lightning.LightningModule, ABC):
    """Base Lightning module.

    See https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    for the required methods.

    Attributes:
        relative_model_path: The path where model artifacts are saved.
            The path is relative to MODEL_DIRECTORY.
        configuration: The configuration used to train the model.
        inputs: The input variables.
        outputs: The output variables.
        output2loss: A mapping from output name to loss function.
        train_metrics: A mapping from output name to training metrics.
        val_metrics: A mapping from output name to validation metrics.
    """

    def __init__(
        self,
        model_path: Path,
        configuration: dict[str, Any],
        inputs: Variables | dict[str, Any],
        outputs: Outputs | dict[str, Any],
    ) -> None:
        """Initialize the base Lightning module.

        Args:
            model_path: The path where model artifacts are saved.
                It should be a path relative to MODEL_DIRECTORY.
            configuration: The configuration used to train the model.
            inputs: The input variables.
            outputs: The output variables.
        """
        super().__init__()

        if model_path.is_absolute():
            try:
                relative_model_path = model_path.relative_to(self.model_directory)
            except ValueError as error:
                raise ValueError(
                    f"The provided path {model_path} is not relative to "
                    f"MODEL_DIRECTORY={self.model_directory}."
                ) from error
        else:
            relative_model_path = Path(model_path)

        self.relative_model_path = relative_model_path
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.configuration = configuration

        # Required for cls.load_from_checkpoint().
        if not isinstance(inputs, Variables):
            inputs = Variables.from_dict(inputs)
        if not isinstance(outputs, Outputs):
            outputs = Outputs.from_dict(outputs)

        self.inputs = inputs
        self.outputs = outputs
        self.model = self._create_model()

        self.output2loss = {output.name: output.build_loss() for output in self.outputs}

        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()
        for output in self.outputs:
            self.train_metrics[output.name] = MetricCollection(
                {
                    f"train/{metric_name}": metric
                    for metric_name, metric in output.build_metrics().items()
                }
            )
            self.val_metrics[output.name] = MetricCollection(
                {
                    f"val/{metric_name}": metric
                    for metric_name, metric in output.build_metrics().items()
                }
            )

        self.save_hyperparameters(ignore=["inputs", "outputs", "model_path"])
        self.save_hyperparameters(
            {
                "outputs": self.outputs.to_dict(),
                "model_path": self.relative_model_path,
            }
        )

        # Strip variables.inputs from the hparams sent to MLflow: the configuration dict saved
        # by save_hyperparameters contains configuration["variables"]["inputs"], which we don't
        # want logged as MLflow params. We must update both _hparams and _hparams_initial because
        # Lightning sends _hparams_initial (not _hparams) to the logger at trainer.fit() time.
        # self.configuration remains unmodified so the model retains full access to input defs.
        # Inputs are still written to hparams.json on disk via save_hparams() below.
        config_for_hparams = copy.deepcopy(self.configuration)
        config_for_hparams.get("variables", {}).pop("inputs", None)
        self._hparams["configuration"] = config_for_hparams
        self._hparams_initial["configuration"] = copy.deepcopy(config_for_hparams)

        self.save_hparams()

    @property
    @abstractmethod
    def model_directory(self) -> Path:
        """The root directory for saving models."""
        pass  # pragma: no cover

    @abstractmethod
    def _save_json(self, data: dict, path: Path) -> None:
        """Saves a dictionary to a JSON file. To be implemented by subclasses."""
        pass  # pragma: no cover

    @property
    def model_path(self) -> Path:
        """Model path."""
        return self.model_directory / self.relative_model_path

    @property
    def hparams_path(self) -> Path:
        """Path to the hparams file."""
        return self.model_path / "hparams.json"

    def save_hparams(self) -> None:
        """Save model hyperparameters.

        We also log the updated hyper-parameters to the logger.
        MLflow does not allow overwriting params, so we skip re-logging for it.
        """
        if isinstance(self.logger, Logger) and not isinstance(self.logger, IntStepMLFlowLogger):
            self.logger.log_hyperparams(dict(self.hparams))

        # Re-inject inputs into the on-disk copy: self.hparams["configuration"] has inputs
        # stripped (to avoid MLflow logging), but hparams.json must contain them for
        # load_from_checkpoint to work.
        hparams_to_save = copy.deepcopy(self.hparams)
        hparams_to_save.setdefault("configuration", {})
        hparams_to_save["configuration"].setdefault("variables", {})
        hparams_to_save["configuration"]["variables"]["inputs"] = self.inputs.to_dict()

        log.info(f"Saving model hyper-parameters in {self.hparams_path}.")
        self._save_json(dict(hparams_to_save), self.hparams_path)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward the inputs through the model.

        Args:
            inputs: The input tensors.

        Returns:
            Output tensors.
        """
        return self.model(inputs)

    def training_step(
        self, batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Training step for the model.

        Args:
            batch: A batch of training data.

        Returns:
            The training loss.
        """
        inputs, targets = batch
        outputs = self(inputs=inputs)

        loss = torch.tensor(0.0, device=self.device)
        logs: dict[str, int | Tensor] = {"step": self.current_epoch}
        for output in self.outputs:
            per_output_loss = self.output2loss[output.name](
                outputs[output.name], targets[output.name].float()
            )
            logs[f"train/{output.name}/loss"] = per_output_loss
            loss += per_output_loss

            self.train_metrics[output.name].update(outputs[output.name], targets[output.name])

        logs["train/loss"] = loss
        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train/batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(
        self, batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ) -> None:
        """Validation step for the model.

        Args:
            batch: A batch of validation data.
        """
        inputs, targets = batch
        outputs = self(inputs)

        loss = torch.tensor(0.0, device=self.device)
        logs: dict[str, int | Tensor] = {"step": self.current_epoch}
        for output in self.outputs:
            per_output_loss = self.output2loss[output.name](
                outputs[output.name], targets[output.name].float()
            )
            logs[f"val/{output.name}/loss"] = per_output_loss
            loss += per_output_loss

            self.val_metrics[output.name].update(outputs[output.name], targets[output.name])

        logs["val/loss"] = loss
        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _log_metrics(self, metrics: nn.ModuleDict, prog_bar: bool = True) -> None:
        """Compute, log and reset training or validation metrics."""
        for output in self.outputs:
            self.log_dict(
                {**metrics[output.name].compute(), "step": self.current_epoch},
                on_step=False,
                on_epoch=True,
                prog_bar=prog_bar,
            )
            metrics[output.name].reset()

    def on_train_epoch_end(self) -> None:
        """Step called at the very end of the epoch loop."""
        self._log_metrics(self.train_metrics, prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        """Step called at the very end of the validation loop of the epoch."""
        self._log_metrics(self.val_metrics)

    def on_train_end(self) -> None:
        """Step called at the end of training to ensure hparams are saved.

        We ensure that self._hparams_initial is updated with the latest hyper-parameters. This is
        notably useful in iterative annotation to avoid logging non-updated hyper-parameters:
        ModelCheckpoint.on_train_epoch_end gets called after self.on_train_epoch_end, and we cannot
        call self.save_hyperparameters within ModelCheckpoint.on_train_epoch_end (cf.
        https://github.com/Lightning-AI/pytorch-lightning/issues/12624), so we call it here.
        """
        # Exclude "checkpoint" from _hparams_initial: the ModelCheckpoint callback writes
        # checkpoint/best_epoch and checkpoint/last_epoch into self.hparams, and these values
        # change across trainer.fit() calls (e.g. during iterative annotation). Since MLflow
        # forbids overwriting params, re-logging _hparams_initial with a changed best_epoch
        # would raise MlflowException. Checkpoint info is still written to hparams.json via
        # save_hparams() which reads from self.hparams (where checkpoint is present).
        hparams_copy = copy.deepcopy(self.hparams)
        hparams_copy.pop("checkpoint", None)
        self._hparams_initial = hparams_copy
        self.save_hparams()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Loads PyTorch optimizer from the configuration dictionary.

        Returns:
            An instantiated optimizer object.

        Raises:
            AttributeError: If an optimizer name specified in the configuration
                does not exist in `torch.optim`.
        """
        config = copy.deepcopy(self.configuration["training"]["optimizer"])
        optimizer_name = config.pop("class_name")

        try:
            optimizer_class = getattr(optim, optimizer_name)
        except AttributeError:
            raise AttributeError(
                f"Optimizer '{optimizer_name}' not found in torch.optim."
            ) from None

        try:
            return optimizer_class(self.model.parameters(), **config)
        except Exception as error:
            raise Exception(
                f"Error instantiating optimizer {optimizer_name} with kwargs: "
                f"{format_dict(config)}"
            ) from error

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Create the model instance.

        This method must be implemented by subclasses to create the specific
        model architecture.

        Returns:
            The model instance.
        """
        pass  # pragma: no cover
