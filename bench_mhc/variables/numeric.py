"""Module to define the numeric variable."""

from typing import Any

import polars as pl
from polars.datatypes.classes import NumericType
from torch import nn
from torchmetrics import MeanSquaredError
from torchmetrics import Metric

from bench_mhc.constants import LazyOrDataFrame
from bench_mhc.custom_objects.losses import MaskedUnlabeledLoss
from bench_mhc.custom_objects.losses import load_loss_from_config
from bench_mhc.custom_objects.metrics import MaskedUnlabeledMetric
from bench_mhc.custom_objects.metrics import load_metric_from_config
from bench_mhc.utils.logging import system
from bench_mhc.variables.base import BaseVariable
from bench_mhc.variables.base import DefaultValueMixin
from bench_mhc.variables.base import OutputMixin
from bench_mhc.variables.base import maybe_raised_not_fitted
from bench_mhc.variables.base import skip_if_fitted

log = system.get(__name__)


class NumericVariable(DefaultValueMixin, BaseVariable):
    """Class for the numeric variable.

    Attributes:
        name: Name to identify the variable.
        column: Optional column linked to the variable.
            If not provided, the 'name' is used.
        is_fitted: Indicates if the variable has been fitted.
        default_value: Optional default value for NaN.
    """

    def __init__(
        self,
        name: str,
        column: str | None = None,
        default_value: float | None = None,
    ) -> None:
        """Initialise the NumericVariable.

        Args:
            name: Name to identify the variable.
            column: Optional column linked to the variable.
                If not provided, the 'name' is used.
            default_value: Optional default value for NaN.
        """
        super().__init__(name=name, column=column, default_value=default_value)

    @skip_if_fitted
    def fit(self, df: LazyOrDataFrame) -> None:
        """Fit the numeric variable.

        We convert to float to support both NaN and None for missing values, cf.
        https://docs.pola.rs/user-guide/expressions/missing-data/#null-and-nan-values.

        Args:
            df: Dataset.

        Returns:
            The fitted numeric variable.

        Raises:
            ValueError: If 'default_value' is not provided and there are missing values in the
                column.
        """
        lf = df.lazy()

        lf = lf.with_columns(pl.col(self.column).cast(pl.Float32).fill_nan(None))

        self.check_missing_values(lf)

        self.is_fitted = True

    @maybe_raised_not_fitted
    def transform(self, df: LazyOrDataFrame) -> LazyOrDataFrame:
        """Transform the numeric variable.

        1. Check 'default_value' is set if there are NaN values in the column.
        2. Fill NaN values with the default value if defined.

        Args:
            df: Dataset.

        Returns:
            The transformed data.

        Raises:
            NotFittedError if the variable is not fitted yet.
        """
        lf = df.lazy()

        lf = lf.with_columns(pl.col(self.column).cast(pl.Float32).fill_nan(None))

        self.check_missing_values(lf)

        if self.default_value is not None:
            lf = lf.with_columns(pl.col(self.column).fill_null(self.default_value))

        lf = lf.with_columns(pl.col(self.column).cast(self.polars_type).alias(self.name))

        return lf if isinstance(df, pl.LazyFrame) else lf.collect(engine="streaming")

    @property
    def polars_type(self) -> type[NumericType]:
        """Type of the variable values in polars."""
        return pl.Float32


class NumericOutput(OutputMixin, NumericVariable):
    """Class for the numeric output variable.

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
        """Initialise the NumericOutput.

        Args:
            name: Name to identify the variable.
            column: Optional column linked to the variable.
                If not provided, the 'name' is used.
            default_value: Optional default value for NaN.
            loss: Optional loss configuration associated with the variable.
                If not provided and the variable is an output, an MSE loss is defined.
            metrics: Optional metrics configuration associated with the variable.
                If not provided and the variable is an output, an MSE metric is defined.
        """
        super().__init__(name=name, column=column, default_value=default_value)
        self.loss_config = loss
        self.metrics_config = metrics

        if self.default_value:
            log.warning(
                f"Default value {self.default_value} provided for numeric output '{self.name}'. "
                "NaN values will be set to this default value and considered in the loss."
            )

    @property
    def output_size(self) -> int:
        """Number of units in the output layer."""
        return 1

    @property
    def default_loss(self) -> nn.Module:
        """Build the mean squared error loss."""
        return nn.MSELoss(reduction="mean")

    @property
    def default_metrics(self) -> dict[str, Metric]:
        """Build the mean squared error metric."""
        return {f"{self.name}/mean_squared_error": MeanSquaredError()}

    def build_loss(self) -> nn.Module:
        """Build the numeric output loss."""
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
        """Build the metrics for the numeric output."""
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
