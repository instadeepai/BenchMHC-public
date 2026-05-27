"""Unit tests related to bench_mhc/custom_objects/losses.py."""

import re
import sys
from unittest.mock import patch

import pytest
import torch
from torch import nn

import bench_mhc.custom_objects.losses as custom_losses
from bench_mhc.custom_objects.losses import MaskedUnlabeledLoss
from bench_mhc.custom_objects.losses import load_loss_from_config
from bench_mhc.utils.format import format_dict


@pytest.fixture
def default_label_value_to_mask() -> int:
    """Define the default label value to mask in MaskedUnlabeledLoss tests."""
    return -1


class MyCustomLoss(nn.Module):
    """Fake custom loss."""

    def __init__(self, penalty_factor: float = 0.1) -> None:
        """Initialise the fake custom loss."""
        super().__init__()
        self.penalty_factor = penalty_factor

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the fake custom loss."""
        mse_loss = nn.MSELoss(reduction="mean")(predictions, targets)
        return mse_loss * (1 - self.penalty_factor)


@pytest.mark.parametrize(
    "loss",
    [nn.MSELoss(reduction="none"), nn.MSELoss(reduction="mean"), nn.MSELoss(reduction="sum")],
)
@pytest.mark.parametrize("no_labeled_samples", [True, False])
def test_masked_unlabeled_numeric_loss(
    loss: nn.Module,
    no_labeled_samples: bool,
    default_label_value_to_mask: int,
) -> None:
    """Check the MaskedUnlabeledLoss with the MSELoss.

    We also check it works with no labeled samples.
    """
    y_true = torch.Tensor(
        [-1.0, -1.0, -1.0, -1.0] if no_labeled_samples else [3.0, 1.0, 0.0, -1.0]
    ).unsqueeze(-1)
    y_pred = torch.Tensor([1.0, 1.0, 1.0, 0.0]).unsqueeze(-1)

    expected_mask = y_true != default_label_value_to_mask
    masked_y_true = y_true[expected_mask]
    masked_y_pred = y_pred[expected_mask]

    if no_labeled_samples:
        true_loss = torch.Tensor([0])
    else:
        true_loss = (masked_y_true - masked_y_pred) ** 2
    if loss.reduction == "none":
        expected_total_loss = torch.zeros_like(y_true)
        expected_total_loss[expected_mask] = true_loss
    elif loss.reduction == "mean":
        expected_total_loss = true_loss.mean()
    else:
        expected_total_loss = true_loss.sum()

    masked_unlabeled_loss = MaskedUnlabeledLoss(loss, default_label_value_to_mask)
    total_loss = masked_unlabeled_loss(y_pred, y_true)

    torch.testing.assert_close(expected_total_loss, total_loss)


@pytest.mark.parametrize(
    "loss",
    [nn.BCELoss(reduction="none"), nn.BCELoss(reduction="mean"), nn.BCELoss(reduction="sum")],
)
@pytest.mark.parametrize("no_labeled_samples", [True, False])
def test_masked_unlabeled_binary_loss(
    loss: nn.Module,
    no_labeled_samples: bool,
    default_label_value_to_mask: int,
) -> None:
    """Check the MaskedUnlabeledLoss with the BCELoss.

    We also check it works with no labeled samples.
    """
    y_true = torch.Tensor([-1, -1, -1, -1] if no_labeled_samples else [1, 0, 1, -1]).unsqueeze(-1)
    y_pred = torch.Tensor([0.8, 0.1, 0.3, 0.2]).unsqueeze(-1)

    expected_mask = y_true != default_label_value_to_mask
    masked_y_true = y_true[expected_mask]
    masked_y_pred = y_pred[expected_mask]

    if no_labeled_samples:
        true_loss = torch.Tensor([0])
    else:
        true_loss = nn.functional.binary_cross_entropy(
            masked_y_pred, masked_y_true, reduction="none"
        )
    if loss.reduction == "none":
        expected_total_loss = torch.zeros_like(y_true)
        expected_total_loss[expected_mask] = true_loss
    elif loss.reduction == "mean":
        expected_total_loss = true_loss.mean()
    else:
        expected_total_loss = true_loss.sum()

    masked_unlabeled_loss = MaskedUnlabeledLoss(loss, default_label_value_to_mask)
    total_loss = masked_unlabeled_loss(y_pred, y_true)

    torch.testing.assert_close(expected_total_loss, total_loss)


def test_load_loss_from_config() -> None:
    """Check load_loss_from_config works as expected."""
    loss = load_loss_from_config({"class_name": "MSELoss", "reduction": "sum"})
    assert isinstance(loss, nn.MSELoss)
    assert loss.reduction == "sum"

    with patch.dict(sys.modules, {custom_losses.__name__: sys.modules[__name__]}):
        loss = load_loss_from_config({"class_name": "MyCustomLoss", "penalty_factor": 0.3})
        assert isinstance(loss, MyCustomLoss)
        assert loss.penalty_factor == 0.3

    match_msg = "Loss WrongLoss not found in torch.nn and bench_mhc.custom_objects.losses.py."
    with pytest.raises(AttributeError, match=match_msg):
        load_loss_from_config({"class_name": "WrongLoss"})

    match_msg = re.escape(
        "Error instantiating loss MSELoss with kwargs: "
        f"{format_dict({'wrong_param': 'wrong_value'})}"
    )
    with pytest.raises(Exception, match=match_msg):
        load_loss_from_config({"class_name": "MSELoss", "wrong_param": "wrong_value"})
