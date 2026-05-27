"""Module to define the input or output variables."""

from collections.abc import Iterator
from collections.abc import Sequence
from typing import Any
from typing import Self

from bench_mhc.constants import LazyOrDataFrame
from bench_mhc.utils.format import format_iterable
from bench_mhc.variables import OUTPUT_VARIABLE_NAME2VARIABLE_CLASS
from bench_mhc.variables import VARIABLE_NAME2VARIABLE_CLASS
from bench_mhc.variables import OutputClassesUnion
from bench_mhc.variables import VariableClassesUnion


class Variables:
    """Class to define variables.

    Attributes:
        name2class_mapping: Mapping from variable class name to variable class.
        variables: The variables.
    """

    name2class_mapping = VARIABLE_NAME2VARIABLE_CLASS
    variables: Sequence[VariableClassesUnion]

    def __init__(self, variables: list[VariableClassesUnion]) -> None:
        """Initialize the variables."""
        self.variables = variables

    def __iter__(self) -> Iterator[VariableClassesUnion]:
        """Iterate over the variables."""
        return iter(self.variables)

    def __len__(self) -> int:
        """Return the number of variables."""
        return len(self.variables)

    def __getitem__(self, name: str) -> Any:
        """Retrieve a variable thanks to its name.

        Args:
            name: The name of the variable to retrieve.

        Raises:
            KeyError: If the name does not correspond to a variable available.
        """
        for variable in self:
            if variable.name == name:
                return variable

        raise KeyError(
            f"Variable with name '{name}' not found. The available variables are the following: "
            f"{format_iterable(variable.name for variable in self)}"
        )

    def __add__(self, other: Self) -> Self:
        """Add two Variables instances.

        It concatenates `variables` from both instances into the `variables` attribute of a newly
        created instance.

        Args:
            other: The other Variables instance to add.

        Raises:
            TypeError: If the quantity to add is not a Variables instance.
        """
        if not isinstance(other, Variables):
            raise TypeError(f"must be {self.__class__.__name__}, not {type(other)}")

        new = self.__new__(self.__class__)
        new.variables = list(self.variables) + list(other.variables)

        return new

    @classmethod
    def from_dict(cls, dictionary: dict[str, Any]) -> Self:
        """Create a Variables instance from a dictionary.

        Args:
            dictionary: The dictionary to create a Variables instance from.

        Returns:
            The Variables instance.
        """
        variables = [
            cls.name2class_mapping[variable.pop("class_name")].from_dict(
                {**variable, "name": variable_name}
            )
            for variable_name, variable in dictionary.items()
        ]
        self = cls.__new__(cls)
        self.variables = variables

        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize the Variables instance into a dictionary."""
        serialized_variables = {
            variable.name: {**variable.to_dict(), **{"class_name": variable.__class__.__name__}}
            for variable in self.variables
        }

        return serialized_variables

    def fit(self: Self, df: LazyOrDataFrame) -> None:
        """Fit the variables on a given DataFrame.

        Args:
            df: The DataFrame to fit the variables on.
        """
        for variable in self:
            variable.fit(df)

    def transform(self, df: LazyOrDataFrame) -> LazyOrDataFrame:
        """Transform the variables on a given DataFrame.

        Args:
            df: The DataFrame to fit the variables on.

        Returns:
            The transformed DataFrame.
        """
        for variable in self:
            df = df.pipe(variable.transform)

        return df

    @property
    def names(self) -> set[str]:
        """Return the set of variables' names.

        Returns:
            The set of variables' names.
        """
        return {variable.name for variable in self}

    @property
    def features(self) -> set[str]:
        """Return the set of features from all the variables.

        Returns:
            The set of features from all the variables.
        """
        return set.union(*[variable.features for variable in self])

    @property
    def is_fitted(self) -> bool:
        """Return whether the variables are all fitted."""
        return all(variable.is_fitted for variable in self)


class Outputs(Variables):
    """Class to define output variables.

    Attributes:
        name2class_mapping: Mapping from variable class name to variable class.
        variables: The output variables.
    """

    name2class_mapping = OUTPUT_VARIABLE_NAME2VARIABLE_CLASS
    variables: Sequence[OutputClassesUnion]

    def __iter__(self) -> Iterator[OutputClassesUnion]:
        """Iterate over the variables.

        We need to override this method to restrict the type to output variables.
        """
        return iter(self.variables)
