"""Module to define custom objects relative to losses."""

import copy
import sys
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn

from bench_mhc.utils.format import format_dict
from bench_mhc.utils.logging import system

log = system.get(__name__)


class MaskedUnlabeledLoss(nn.Module):
    """Wrapper for PyTorch loss functions to handle masking for unlabeled samples."""

    def __init__(self, base_loss: nn.Module, mask_value: float) -> None:
        """Initialize the loss wrapper.

        Args:
            base_loss: The base loss function (e.g. nn.BCELoss).
            mask_value: The value in `targets` that indicates unlabeled samples.
        """
        super().__init__()
        self.base_loss = base_loss
        self.mask_value = mask_value

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss while ignoring samples where targets == mask_value.

        If the reduction of the base loss is 'none', a vector of the same size as targets is
        returned, with 0s on unlabeled samples and per-sample loss on labeled samples.

        Autocast is disabled to support mixed precision with 'cuda'.

        Args:
            predictions: Model predictions.
            targets: Ground truth labels (with mask_value for unlabeled samples).

        Returns:
            A torch.Tensor: if 'base_loss.reduction' = 'none', a vector with 0s for unlabeled
            samples and per-sample loss on labeled samples, otherwise, the reduced loss computed
            only on labeled samples.
        """
        mask = targets != self.mask_value

        if not mask.any():
            if self.base_loss.reduction == "none":
                return torch.zeros_like(targets, dtype=predictions.dtype, device=predictions.device)
            else:
                return torch.tensor(0.0, dtype=predictions.dtype, device=predictions.device)

        masked_predictions = predictions[mask]
        masked_targets = targets[mask]

        device_type = predictions.device.type
        with (
            torch.autocast(enabled=False, device_type=device_type)
            if device_type != "mps"
            else nullcontext()
        ):
            loss = self.base_loss(masked_predictions.float(), masked_targets)

            if self.base_loss.reduction == "none":
                full_loss = torch.zeros_like(
                    targets, dtype=predictions.dtype, device=predictions.device
                )
                full_loss[mask] = loss

                return full_loss

        return loss


def load_loss_from_config(config: dict[str, Any]) -> nn.Module:
    """Load custom or PyTorch Lightning loss from a configuration dictionary.

    Args:
        config: A dictionary representing a loss.
            The dictionary must contain a key "class_name" specifying the name of the loss class
            (as a string). The remaining keys in the dictionary are treated as keyword arguments to
            the loss.

    Returns:
        An instantiated loss object.

    Raises:
        AttributeError: If the loss name specified in the configuration does not exist in
            `torch.nn` and `bench_mhc.custom_objects.losses.py`.
        ValueError: If the kwargs don't match the loss class.
    """
    config = copy.deepcopy(config)

    loss_class_name = config.pop("class_name")

    try:
        loss_class = getattr(nn, loss_class_name)
        log.info(f"Loss {loss_class_name} found in torch.nn.")
    except AttributeError:
        log.info(f"Loss {loss_class_name} not found in torch.nn...")
        try:
            custom_module = sys.modules[__name__]
            loss_class = getattr(custom_module, loss_class_name)
            log.info(f"Loss {loss_class_name} found in bench_mhc.custom_objects.losses.py.")
        except AttributeError:
            raise AttributeError(
                f"Loss {loss_class_name} not found in torch.nn and "
                "bench_mhc.custom_objects.losses.py."
            ) from None

    try:
        loss = loss_class(**config)
    except Exception as error:
        raise Exception(
            f"Error instantiating loss {loss_class_name} with kwargs: {format_dict(config)}"
        ) from error

    return loss
