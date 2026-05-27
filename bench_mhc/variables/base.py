"""Module to define the base variable class used as interface for all variable classes."""

import copy
import functools
import inspect
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import Protocol
from typing import Self

import polars as pl
from polars import LazyFrame
from torch import nn
from torchmetrics import Metric

from bench_mhc.constants import LazyOrDataFrame
from bench_mhc.utils.errors import NotFittedError
from bench_mhc.utils.logging import system

log = system.get(__name__)


class VariableWithNameColumnDefaultValue(Protocol):
    """Protocol for classes that have a name, column, and default_value attributes."""

    name: str
    column: str
    default_value: float | None


def skip_if_fitted(method: Callable) -> Callable:
    """Decorator to skip the execution of a method if the variable is fitted.

    Args:
        method: The method to wrap.

    Returns:
        callable: The maybe skipped method.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(self, "is_fitted", False):
            log.warning(f"The '{self.name}' variable is already fitted and won't be re-fitted.")

            return self

        return method(self, *args, **kwargs)

    return wrapper


def maybe_raised_not_fitted(method: Callable) -> Callable:
    """Decorator to raise a ValueError if the variable is not fitted.

    Args:
        method: The method to wrap.

    Returns:
        callable: The method.

    Raises:
        ValueError: If variable.is_fitted is False.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not getattr(self, "is_fitted", False):
            raise NotFittedError(
                f"The variable '{self.name}' must be fit on '{self.column}' column "
                "before being used to transform data."
            )

        return method(self, *args, **kwargs)

    return wrapper


class BaseVariable(ABC):
    """Base class for all the variables.

    Attributes:
        name: Name to identify the variable.
        column: Optional column linked to the variable.
            If not provided, the 'name' is used.
        is_fitted: Indicates if the variable has been fitted.
    """

    def __init__(self, name: str, column: str | None = None) -> None:
        """Initialise the BaseVariable.

        Args:
            name: Name to identify the variable.
            column: Optional column linked to the variable.
                If not provided, the 'name' is used.
        """
        self.name = name
        self.column = column or name

        self.is_fitted = False

    @abstractmethod
    def fit(self, df: LazyOrDataFrame) -> None:
        """Abstract method to fit the variable."""

    @abstractmethod
    def transform(self, df: LazyOrDataFrame) -> LazyOrDataFrame:
        """Abstract method to transform the variable."""

    @classmethod
    def from_dict(cls, dictionary: dict) -> Self:
        """Create a variable instance from a dictionary."""
        init_params_names = set(inspect.signature(cls.__init__).parameters.keys())
        init_params_names.remove("self")
        init_params = {}
        for p_name in init_params_names:
            # Some __init__ parameters may not be serialized in the dictionary (for models with old
            # variables), hence the try/except. The backward compatibility for these parameters is
            # handled with default values in the __init__
            try:
                init_params[p_name] = dictionary.pop(p_name)
            except KeyError:
                pass

        self = cls(**init_params)

        for attr_name, attr_value in dictionary.items():
            setattr(self, attr_name, attr_value)

        return self

    def to_dict(self) -> dict:
        """Serialize the variable instance into a dictionary.

        Convert pathlib.Path attribute to str to make it JSON-serializable.
        """
        dict_ = copy.deepcopy(self.__dict__)

        for attr_name, attr_value in dict_.items():
            if isinstance(attr_value, Path):
                dict_[attr_name] = str(attr_value)

        return dict_

    @property
    def features(self) -> set[str]:
        """Features of the variable when the dataset is transformed."""
        return {self.name}


class OutputMixin(ABC):
    """Mixin for output variables."""

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Number of units in the output layer."""

    @property
    @abstractmethod
    def default_loss(self) -> nn.Module:
        """Build the torch default loss."""

    @property
    @abstractmethod
    def default_metrics(self) -> dict[str, Metric]:
        """Build the torch default metrics."""

    @abstractmethod
    def build_loss(self) -> nn.Module:
        """Build the torch loss."""

    @abstractmethod
    def build_metrics(self) -> dict[str, Metric]:
        """Build the torch metrics."""


class DefaultValueMixin:
    """Mixin which adds a `default_value` and the logic to check missing values."""

    def __init__(
        self,
        default_value: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialise the DefaultValueMixin.

        Args:
            default_value: Optional default value for NaN.
            *args: Variable length argument list to pass to the parent class.
            **kwargs: Arbitrary keyword arguments to pass to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.default_value = default_value

    def check_missing_values(self: VariableWithNameColumnDefaultValue, lf: LazyFrame) -> None:
        """Raise if there are NaNs but no default_value was provided.

        Args:
            lf: Dataset.

        Raises:
            ValueError: If 'default_value' is not provided and there are missing values in the
                column.
        """
        null_count = (
            lf.select(pl.col(self.column).is_null().sum()).collect(engine="streaming").item()
        )
        if null_count > 0:
            num_samples = lf.select(pl.len()).collect(engine="streaming").item()
            log.warning(
                f"There are {null_count:,} ({null_count/num_samples*100:.2f}%) "
                f"NaN values in '{self.name}'."
            )
            if self.default_value is None:
                raise ValueError(
                    f"The default value None for variable '{self.name}' was not provided "
                    f"but there are NaN values in the dataset's column '{self.column}'."
                )
