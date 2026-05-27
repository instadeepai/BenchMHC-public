"""Module to define custom objects relative to losses."""

import copy
import sys
from typing import Any

import torch
import torchmetrics
from torchmetrics import Metric

from bench_mhc.utils.format import format_dict
from bench_mhc.utils.logging import system

log = system.get(__name__)


class MaskedUnlabeledMetric(Metric):
    """Wrapper for torchmetrics.Metric to mask unlabeled samples before computing the metric.

    Attributes:
        base_metric: The base metric (e.g. torchmetrics.AveragePrecision).
        mask_value: The value in `target` that indicates unlabeled samples.
        _has_updates: Whether the metric has been already updated.
    """

    def __init__(self, base_metric: Metric, mask_value: float, **kwargs: Any):
        """Initialize the metric wrapper.

        Args:
            base_metric: The base metric (e.g. torchmetrics.AveragePrecision).
            mask_value: The value in `target` that indicates unlabeled samples.
            kwargs: Additional keyword arguments to pass to the base metric.
        """
        super().__init__(**kwargs)
        self.base_metric = base_metric
        self.mask_value = mask_value
        self._has_updates = False

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric with filtered predictions and targets.

        Args:
            preds: Predictions.
            target: Targets.
        """
        mask = target != self.mask_value

        if mask.any():
            self.base_metric.update(preds[mask], target[mask])
            self._has_updates = True

    def compute(self) -> torch.Tensor:
        """Compute the final metric."""
        # Handle the case where no updates have been done because no samples were labeled
        if not self._has_updates:
            return torch.tensor(0.0, device=self.device)

        return self.base_metric.compute()

    def reset(self) -> None:
        """Reset the state of the metric."""
        self.base_metric.reset()
        self._has_updates = False


def load_metric_from_config(config: dict[str, Any]) -> tuple[Metric, str]:
    """Load custom or PyTorch Lightning metric from a configuration dictionary.

    Args:
        config: A dictionary representing a metric.
            The dictionary must contain a key "class_name" specifying the name of the metric class
            (as a string). The remaining keys in the dictionary are treated as keyword arguments to
            the metric.

    Returns:
        A tuple with an instantiated metric object and the metric name.

    Raises:
        AttributeError: If the metric name specified in the configuration does not exist in
            `torchmetrics` and `bench_mhc.custom_objects.metrics.py`.
        ValueError: If the kwargs don't match the metric class.
    """
    config = copy.deepcopy(config)

    metric_class_name = config.pop("class_name")
    metric_name = config.pop("name") if "name" in config else metric_class_name

    metric_class = _get_metric_class(metric_class_name)

    try:
        metric = metric_class(**config)
    except Exception as error:
        raise Exception(
            f"Error instantiating metric {metric_class_name} with kwargs: {format_dict(config)}"
        ) from error

    return metric, metric_name


def _get_metric_class(metric_class_name: str) -> Metric:
    """Get an instantiated metric object from its name.

    Args:
        metric_class_name: Name of the metric class.

    Returns:
        An instantiated metric object.

    Raises:
        AttributeError: If the metric name specified in the configuration does not exist in
            `torchmetrics` and `bench_mhc.custom_objects.metrics.py`.
    """
    for module in [torchmetrics, torchmetrics.classification, sys.modules[__name__]]:
        try:
            metric_class = getattr(module, metric_class_name)
            log.info(f"Metric {metric_class_name} found in {module.__name__}.")
            return metric_class

        except AttributeError:
            log.info(f"Metric {metric_class_name} not found in {module.__name__}...")

    raise AttributeError(
        f"Metric {metric_class_name} not found in torchmetrics modules and "
        f"bench_mhc.custom_objects.metrics.py."
    ) from None
