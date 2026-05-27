"""Unit tests related to bench_mhc/custom_objects/metrics.py."""

import re
import sys
from unittest.mock import patch

import pytest
import torch
from torchmetrics import MeanSquaredError
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy

import bench_mhc.custom_objects.metrics as custom_metrics
from bench_mhc.custom_objects.metrics import MaskedUnlabeledMetric
from bench_mhc.custom_objects.metrics import load_metric_from_config
from bench_mhc.utils.format import format_dict


@pytest.fixture
def default_label_value_to_mask() -> int:
    """Define the default label value to mask in MaskedUnlabeledMetric tests."""
    return -1


class MyCustomMetric(Metric):
    """Fake custom metric."""

    def __init__(self, penalty_factor: float = 0.1) -> None:
        """Initialise the fake custom metric."""
        super().__init__()
        self.mse_metric = MeanSquaredError()
        self.penalty_factor = penalty_factor

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric with predictions and targets.

        Args:
            preds: Predictions.
            target: Targets.
        """
        self.mse_metric.update(preds, target)

    def compute(self) -> torch.Tensor:
        """Compute the final fake custom metric."""
        return self.mse_metric.compute() * (1 - self.penalty_factor)

    def reset(self) -> None:
        """Reset the state of the metric."""
        self.mse_metric.reset()


@pytest.mark.parametrize(
    "metric",
    [MeanSquaredError()],
)
@pytest.mark.parametrize("no_labeled_samples", [True, False])
def test_masked_unlabeled_numeric_metric(
    metric: Metric,
    no_labeled_samples: bool,
    default_label_value_to_mask: int,
) -> None:
    """Check the MaskedUnlabeledMetric with the MeanSquaredError.

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
        expected_metric = torch.tensor(0.0)
    else:
        expected_metric = ((masked_y_true - masked_y_pred) ** 2).mean()

    masked_unlabeled_metric = MaskedUnlabeledMetric(metric, default_label_value_to_mask)

    total_metric = masked_unlabeled_metric(y_pred, y_true)

    torch.testing.assert_close(expected_metric, total_metric)

    masked_unlabeled_metric.reset()
    masked_unlabeled_metric.update(y_pred, y_true)
    total_metric = masked_unlabeled_metric.compute()

    torch.testing.assert_close(expected_metric, total_metric)


@pytest.mark.parametrize(
    "metric",
    [BinaryAccuracy()],
)
@pytest.mark.parametrize("no_labeled_samples", [True, False])
def test_masked_unlabeled_binary_metric(
    metric: Metric,
    no_labeled_samples: bool,
    default_label_value_to_mask: int,
) -> None:
    """Check the MaskedUnlabeledMetric with the BinaryAccuracy.

    We also check it works with no labeled samples.
    """
    y_true = torch.Tensor([-1, -1, -1, -1] if no_labeled_samples else [1, 0, 1, -1]).unsqueeze(-1)
    y_pred = torch.Tensor([0.8, 0.1, 0.3, 0.2]).unsqueeze(-1)

    if no_labeled_samples:
        expected_metric = torch.tensor(0.0)
    else:
        expected_metric = torch.tensor(2 / 3)

    masked_unlabeled_metric = MaskedUnlabeledMetric(metric, default_label_value_to_mask)

    total_metric = masked_unlabeled_metric(y_pred, y_true)

    torch.testing.assert_close(expected_metric, total_metric)

    masked_unlabeled_metric.reset()
    masked_unlabeled_metric.update(y_pred, y_true)
    total_metric = masked_unlabeled_metric.compute()

    torch.testing.assert_close(expected_metric, total_metric)


def test_load_metric_from_config() -> None:
    """Check load_metric_from_config works as expected."""
    metric, metric_name = load_metric_from_config(
        {"class_name": "MeanSquaredError", "squared": False}
    )
    assert isinstance(metric, MeanSquaredError)
    assert not metric.squared
    assert metric_name == "MeanSquaredError"

    metric, metric_name = load_metric_from_config(
        {"class_name": "MeanSquaredError", "squared": False, "name": "mean_squared_error"}
    )
    assert isinstance(metric, MeanSquaredError)
    assert not metric.squared
    assert metric_name == "mean_squared_error"

    with patch.dict(sys.modules, {custom_metrics.__name__: sys.modules[__name__]}):
        metric, metric_name = load_metric_from_config(
            {"class_name": "MyCustomMetric", "penalty_factor": 0.3}
        )
        assert isinstance(metric, MyCustomMetric)
        assert metric.penalty_factor == 0.3
        assert metric_name == "MyCustomMetric"

    match_msg = (
        "Metric WrongMetric not found in torchmetrics modules and "
        "bench_mhc.custom_objects.metrics.py."
    )
    with pytest.raises(AttributeError, match=match_msg):
        load_metric_from_config({"class_name": "WrongMetric"})

    match_msg = re.escape(
        "Error instantiating metric MeanSquaredError with kwargs: "
        f"{format_dict({'wrong_param': 'wrong_value'})}"
    )
    with pytest.raises(Exception, match=match_msg):
        load_metric_from_config({"class_name": "MeanSquaredError", "wrong_param": "wrong_value"})
