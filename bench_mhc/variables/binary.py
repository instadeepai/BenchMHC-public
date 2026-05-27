"""Module to define the binary variable."""

from typing import Any

import polars as pl
from polars.datatypes.classes import NumericType
from torch import nn
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryAveragePrecision

from bench_mhc.constants import LazyOrDataFrame
from bench_mhc.custom_objects.losses import MaskedUnlabeledLoss
from bench_mhc.custom_objects.losses import load_loss_from_config
from bench_mhc.custom_objects.metrics import MaskedUnlabeledMetric
from bench_mhc.custom_objects.metrics import load_metric_from_config
from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.logging import system
from bench_mhc.variables.base import OutputMixin
from bench_mhc.variables.base import skip_if_fitted
from bench_mhc.variables.numeric import NumericVariable

log = system.get(__name__)


class BinaryVariable(NumericVariable):
    """Class for the binary variable.

    Attributes:
        name: Name to identify the variable.
        column: Optional column linked to the variable.
            If not provided, the 'name' is used.
        is_fitted: Indicates if the variable has been fitted.
        default_value: Optional default value for NaN.
    """

    @skip_if_fitted
    def fit(self, df: LazyOrDataFrame) -> None:
        """Fit the binary variable.

        Args:
            df: Dataset.

        Raises:
            ValueError: If the column does not contain binary values.
        """
        super().fit(df)

        lf = df.lazy()

        if lf.collect_schema()[self.column] in {pl.String, pl.Int32}:
            lf = lf.with_columns(pl.col(self.column).cast(pl.Float32))

        lf = lf.fill_nan(None)

        unique_values = set(
            lf.select(pl.col(self.column).drop_nulls().unique())
            .collect(engine="streaming")[self.column]
            .to_list()
        )
        if self.default_value in unique_values - {0, 1}:
            unique_values.remove(self.default_value)
        if unique_values != {0, 1}:
            raise ValueError(
                f"The column '{self.column}' for variable '{self.name}' contains the following "
                f"unique values: {format_iterable(unique_values)} and is hence not binary."
            )

    @property
    def polars_type(self) -> type[NumericType]:
        """Type of the variable values in polars."""
        return pl.Int32


class BinaryOutput(OutputMixin, BinaryVariable):
    """Class for the binary output variable.

    Attributes:
        name: Name to identify the variable.
        column: Optional column linked to the variable.
            If not provided, the 'name' is used.
        is_fitted: Indicates if the variable has been fitted.
        default_value: Optional default value for NaN.
        loss_config: Optional loss configuration associated with the variable.
        metrics_config: Optional metrics configuration associated with the variable.
    """

    def __init__(
        self,
        name: str,
        column: str | None = None,
        default_value: float | None = None,
        loss: dict[str, Any] | None = None,
        metrics: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialise the BinaryOutput.

        Args:
            name: Name to identify the variable.
            column: Optional column linked to the variable.
                If not provided, the 'name' is used.
            default_value: Optional default value for NaN.
            loss: Optional loss configuration associated with the variable.
                If not provided and the variable is an output, a BCE loss is defined.
            metrics: Optional metrics configuration associated with the variable.
                If not provided and the variable is an output, binary accuracy and average
                precision metrics are defined.
        """
        super().__init__(name=name, column=column, default_value=default_value)
        self.loss_config = loss
        self.metrics_config = metrics

        if self.default_value in {0, 1}:
            log.warning(
                f"Default value {self.default_value} provided for binary output '{self.name}'. "
                "NaN values will be set to this default value and considered in the loss."
            )

    @property
    def output_size(self) -> int:
        """Number of units in the output layer."""
        return 1

    @property
    def default_loss(self) -> nn.Module:
        """Build the binary cross entropy loss."""
        return nn.BCELoss(reduction="mean")

    @property
    def default_metrics(self) -> dict[str, Metric]:
        """Build the binary accuracy and the average precision metrics."""
        return {
            f"{self.name}/binary_accuracy": BinaryAccuracy(),
            f"{self.name}/average_precision": BinaryAveragePrecision(),
        }

    def build_loss(self) -> nn.Module:
        """Build the binary output loss."""
        loss = (
            load_loss_from_config(self.loss_config)
            if self.loss_config is not None
            else self.default_loss
        )

        if self.default_value is not None:
            return MaskedUnlabeledLoss(loss, mask_value=self.default_value)
        else:
            return loss

    def build_metrics(self) -> dict[str, Metric]:
        """Build the metrics for binary classification."""
        metrics: dict[str, Metric] = {}
        if self.metrics_config is None:
            metrics.update(self.default_metrics)
        else:
            for metric_config in self.metrics_config:
                metric_class, metric_name = load_metric_from_config(metric_config)
                metrics.update({f"{self.name}/{metric_name}": metric_class})

        if self.default_value is not None:
            return {
                metric_name: MaskedUnlabeledMetric(metric, mask_value=self.default_value)
                for metric_name, metric in metrics.items()
            }
        else:
            return metrics
