"""Unit tests related to bench_mhc/variables/numeric.py."""

from typing import Any
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from torch import nn
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError

from bench_mhc.custom_objects.losses import MaskedUnlabeledLoss
from bench_mhc.custom_objects.metrics import MaskedUnlabeledMetric
from bench_mhc.utils.errors import NotFittedError
from bench_mhc.variables import NumericOutput
from bench_mhc.variables import NumericVariable


class TestNumericVariable:
    """Test cases for the NumericVariable class."""

    variable_name = "numeric_variable"

    @pytest.mark.parametrize("column", [None, "numeric_column"])
    @pytest.mark.parametrize("default_value", [None, -np.inf])
    def test_init_numeric_variable(self, column: str | None, default_value: float | None) -> None:
        """Test initializing the numeric variable."""
        variable = NumericVariable(
            name=self.variable_name, column=column, default_value=default_value
        )

        assert variable.name == self.variable_name
        assert variable.column == column or self.variable_name
        assert variable.default_value == default_value
        assert variable.polars_type == pl.Float32
        assert not variable.is_fitted

    @pytest.mark.parametrize("dataframe", ["lazy", "eager"], indirect=True)
    @pytest.mark.parametrize(
        "column_name", ["numeric", "numeric_str", "binary", "binary_float", "binary_str"]
    )
    @pytest.mark.parametrize("default_value", [None, -np.inf])
    def test_fit_transform_numeric_variable(
        self,
        dataframe: pl.DataFrame | pl.LazyFrame,
        default_value: float | None,
        column_name: str,
    ) -> None:
        """Test fitting and transforming a dataframe with a numeric variable.

        We also test a numeric variable works with binary data.
        """
        variable = NumericVariable(
            name=self.variable_name, column=column_name, default_value=default_value
        )

        match_msg = (
            f"The variable '{self.variable_name}' must be fit on '{column_name}' column "
            "before being used to transform data."
        )
        with pytest.raises(NotFittedError, match=match_msg):
            variable.transform(dataframe)

        variable.fit(dataframe)
        assert variable.is_fitted

        transformed_dataframe = variable.transform(dataframe)
        assert isinstance(transformed_dataframe, type(dataframe))
        assert (
            transformed_dataframe.lazy()
            .collect()[self.variable_name]
            .equals(dataframe.lazy().collect()[column_name].cast(variable.polars_type))
        )

    @pytest.mark.parametrize("dataframe", ["lazy", "eager"], indirect=True)
    @pytest.mark.parametrize("default_value", [None, -np.inf])
    @pytest.mark.parametrize(
        "column_name",
        ["numeric_nan", "numeric_str_nan", "binary_nan", "binary_float_nan", "binary_str_nan"],
    )
    def test_fit_transform_numeric_variable_nan(
        self, dataframe: pl.DataFrame | pl.LazyFrame, default_value: float | None, column_name: str
    ) -> None:
        """Test fitting and transforming a dataframe with a numeric variable with NaN values.

        We also test it works using a numeric variable on binary data.
        """
        variable = NumericVariable(
            name=self.variable_name, column=column_name, default_value=default_value
        )

        if default_value is None:
            match_msg = (
                f"The default value {default_value} for variable '{self.variable_name}' was not "
                f"provided but there are NaN values in the dataset's column '{column_name}'."
            )
            with pytest.raises(ValueError, match=match_msg):
                variable.fit(dataframe)

        else:
            variable.fit(dataframe)
            assert variable.is_fitted

            # Check that fit is not called if variable already fitted
            with patch.object(dataframe, "lazy") as lazy_spy:
                variable.fit(dataframe)

            lazy_spy.assert_not_called()

            transformed_dataframe = variable.transform(dataframe)
            assert isinstance(transformed_dataframe, type(dataframe))

            expected_column = (
                dataframe.lazy()
                .collect()[column_name]
                .cast(variable.polars_type)
                .fill_nan(default_value)
                .fill_null(default_value)
            )
            assert (
                transformed_dataframe.lazy().collect()[self.variable_name].equals(expected_column)
            )

            # Assert None and NaN get all replaced
            assert (
                transformed_dataframe.lazy()
                .select((pl.col(column_name) == default_value).sum())
                .collect()
                .item()
                == 2
            )


class TestNumericOutput:
    """Test cases for the NumericOutput class."""

    output_name = "numeric_output"

    @pytest.mark.parametrize("column", [None, "numeric_column"])
    @pytest.mark.parametrize("default_value", [None, -np.inf])
    @pytest.mark.parametrize("loss_config", [None, {"class_name": "MSELoss", "reduction": "sum"}])
    @pytest.mark.parametrize(
        "metrics_config",
        [
            None,
            [{"class_name": "MeanSquaredError", "squared": False}],
            [
                {"class_name": "MeanSquaredError"},
                {"class_name": "MeanAbsoluteError", "name": "mean_absolute_error"},
            ],
        ],
    )
    def test_numeric_output(
        self,
        column: str | None,
        default_value: float | None,
        loss_config: dict[str, Any] | None,
        metrics_config: list[dict[str, Any]] | None,
    ) -> None:
        """Test the numeric output."""
        with patch("bench_mhc.variables.numeric.log") as log_mock:
            output = NumericOutput(
                name=self.output_name,
                column=column,
                default_value=default_value,
                loss=loss_config,
                metrics=metrics_config,
            )

        assert output.name == self.output_name
        assert output.column == column or self.output_name
        assert output.default_value == default_value
        assert output.polars_type == pl.Float32
        assert not output.is_fitted
        assert output.output_size == 1

        loss = output.build_loss()

        if default_value is None:
            assert isinstance(loss, nn.MSELoss)
            assert loss.reduction == "sum" if loss_config else "mean"
        else:
            assert isinstance(loss, MaskedUnlabeledLoss)
            assert isinstance(loss.base_loss, nn.MSELoss)
            assert loss.base_loss.reduction == "sum" if loss_config else "mean"
            assert loss.mask_value == default_value

        metrics = output.build_metrics()

        expected_metric2_metric_class: dict[str, Any]
        if metrics_config is None:
            expected_metric2_metric_class = {
                f"{self.output_name}/mean_squared_error": MeanSquaredError
            }
        elif isinstance(metrics_config, list) and len(metrics_config) == 1:
            expected_metric2_metric_class = {
                f"{self.output_name}/MeanSquaredError": MeanSquaredError
            }
        else:
            expected_metric2_metric_class = {
                f"{self.output_name}/MeanSquaredError": MeanSquaredError,
                f"{self.output_name}/mean_absolute_error": MeanAbsoluteError,
            }

        for expected_metric_name, expected_metric_class in expected_metric2_metric_class.items():
            assert expected_metric_name in metrics
            metric = metrics[expected_metric_name]

            if default_value is None:
                assert isinstance(metric, expected_metric_class)
            else:
                assert isinstance(metric, MaskedUnlabeledMetric)
                assert isinstance(metric.base_metric, expected_metric_class)
                assert metric.mask_value == default_value

        if default_value:
            log_mock.warning.assert_called_once_with(
                f"Default value {default_value} provided for numeric output '{self.output_name}'. "
                "NaN values will be set to this default value and considered in the loss."
            )
