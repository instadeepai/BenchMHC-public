"""Module to define the multiclass variable."""

from pathlib import Path

import polars as pl

from bench_mhc.constants import LazyOrDataFrame
from bench_mhc.utils.io import load_txt
from bench_mhc.utils.logging import system
from bench_mhc.variables.base import BaseVariable
from bench_mhc.variables.base import DefaultValueMixin
from bench_mhc.variables.base import maybe_raised_not_fitted
from bench_mhc.variables.base import skip_if_fitted

log = system.get(__name__)


class MultiClassVariable(DefaultValueMixin, BaseVariable):
    """Class for the multiclass variable.

    Attributes:
        name: Name to identify the variable.
        column: The column in the dataframe linked to this variable.
        classes_file_path: Optional path to a .txt file containing class labels, one per line.
        classes: A sorted list of unique class labels, populated after fitting.
        class_to_idx: A mapping from class labels to integer indices, populated after fitting.
    """

    def __init__(
        self,
        name: str,
        column: str | None = None,
        classes_file_path: str | Path | None = None,
    ):
        """Initialise the MultiClassVariable.

        Args:
            name: Name to identify the variable.
            column: Optional column linked to the variable. If not provided, 'name' is used.
            classes_file_path: Optional path to a .txt file to load classes from.
                         If not provided, classes are inferred from the training data.
        """
        super().__init__(name=name, column=column)
        self.classes_file_path = Path(classes_file_path) if classes_file_path else None
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

    @skip_if_fitted
    def fit(self, df: LazyOrDataFrame) -> None:
        """Fit the variable by creating a mapping from class names to integer indices.

        Args:
            df: The dataset to fit on.

        Raises:
            FileNotFoundError: If the classes file is not found.
            ValueError: If classes_file_path is provided and the dataset contains unknown classes.
        """
        lf = df.lazy()

        self.check_missing_values(lf)

        if self.classes_file_path:
            log.info(
                f"Loading class vocabulary for '{self.name}' from file: {self.classes_file_path}"
            )
            try:
                classes = load_txt(self.classes_file_path)
            except FileNotFoundError as err:
                raise FileNotFoundError(
                    f"Classes file not found: {self.classes_file_path}"
                ) from err

            # Check that the dataset does not contain classes not listed in the file
            dataset_classes = set(
                lf.select(pl.col(self.column).unique())
                .collect(engine="streaming")[self.column]
                .to_list()
            )
            unknown_classes = dataset_classes - set(classes)
            if unknown_classes:
                raise ValueError(
                    f"Found {len(unknown_classes)} unknown class(es) in dataset for '{self.name}' "
                    f"which are not in classes_file_path: {sorted(unknown_classes)}"
                )
            self.classes = classes
        else:
            log.info(f"Inferring classes for '{self.name}' from column '{self.column}'")
            unique_values = (
                lf.select(pl.col(self.column).unique())
                .sort(self.column)
                .collect(engine="streaming")[self.column]
                .to_list()
            )
            self.classes = unique_values

        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.is_fitted = True
        log.info(f"'{self.name}' fitted with {len(self.classes)} classes.")

    @maybe_raised_not_fitted
    def transform(self, df: LazyOrDataFrame) -> LazyOrDataFrame:
        """Transform the categorical column to integer indices based on the fitted mapping.

        Args:
            df: The dataframe to transform.

        Returns:
            The transformed dataframe with an added column of integer indices.

        Raises:
            NotFittedError if the variable is not fitted yet.
        """
        lf = df.lazy()

        self.check_missing_values(lf)

        lf = lf.with_columns(
            pl.col(self.column)
            .replace_strict(self.class_to_idx, return_dtype=self.polars_type)
            .alias(self.name)
        )

        return lf if isinstance(df, pl.LazyFrame) else lf.collect(engine="streaming")

    @property
    def polars_type(self) -> type[pl.Int32]:
        """The Polars data type for the transformed variable (integer indices)."""
        return pl.Int32
