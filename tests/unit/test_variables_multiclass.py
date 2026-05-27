"""Unit tests related to bench_mhc/variables/multiclass.py."""

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from bench_mhc.utils.errors import NotFittedError
from bench_mhc.utils.io import load_txt
from bench_mhc.utils.io import save_txt
from bench_mhc.variables import MultiClassVariable


@pytest.fixture
def classes_file_path(tmp_path: Path) -> Path:
    """Create a .txt file with allele classes for MultiClassVariable tests."""
    classes = [
        "HLA__A0103",
        "HLA__A0201",
        "HLA__B0702",
        "HLA__C0102",
        "HLA__DPA10103=DPB10101",
        "HLA__DQA10101=DQB10201",
        "HLA__DRB10101",
        "HLA__DRB11610",
    ]

    classes_file_path = tmp_path / "multiclass_classes.txt"
    save_txt(classes, classes_file_path)

    return classes_file_path


class TestMultiClassVariable:
    """Test cases for the MultiClassVariable class."""

    variable_name = "multiclass_variable"
    column_name = "allele"

    @pytest.mark.parametrize("column", [None, "binary_column"])
    @pytest.mark.parametrize("classes_file_path", [None, "classes_file.txt"])
    def test_init_multiclass_variable(
        self, column: str | None, classes_file_path: str | None
    ) -> None:
        """Test initializing the multiclass variable."""
        variable = MultiClassVariable(
            name=self.variable_name,
            column=column or self.column_name,
            classes_file_path=classes_file_path,
        )

        assert variable.name == self.variable_name
        assert variable.column == (column or self.column_name)
        assert variable.polars_type == pl.Int32
        assert not variable.is_fitted
        assert variable.classes == []
        assert variable.class_to_idx == {}
        assert variable.classes_file_path == (
            Path(classes_file_path) if classes_file_path else None
        )

    @pytest.mark.parametrize("ma_dataframe", ["lazy", "eager"], indirect=True)
    @pytest.mark.parametrize("with_classes_file_path", [True, False])
    def test_fit(
        self,
        ma_dataframe: pl.DataFrame | pl.LazyFrame,
        with_classes_file_path: bool,
        classes_file_path: Path,
    ) -> None:
        """Test fitting using a classes file to define classes."""
        variable = MultiClassVariable(
            name=self.variable_name,
            column=self.column_name,
            classes_file_path=classes_file_path if with_classes_file_path else None,
        )

        variable.fit(ma_dataframe)
        assert variable.is_fitted

        expected_classes = [
            "HLA__A0103",
            "HLA__A0201",
            "HLA__B0702",
            "HLA__C0102",
            "HLA__DPA10103=DPB10101",
            "HLA__DQA10101=DQB10201",
            "HLA__DRB10101",
            "HLA__DRB11610",
        ]

        assert variable.classes == expected_classes
        assert variable.class_to_idx == {name: i for i, name in enumerate(expected_classes)}

        with patch.object(ma_dataframe, "lazy") as lazy_spy:
            variable.fit(ma_dataframe)

        lazy_spy.assert_not_called()

    @pytest.mark.parametrize("ma_dataframe", ["lazy", "eager"], indirect=True)
    def test_fit_from_file_with_unknown_classes(
        self, ma_dataframe: pl.DataFrame | pl.LazyFrame, classes_file_path: Path
    ) -> None:
        """Test that fitting from a file raises an error if the dataset contains unknown classes."""
        classes = load_txt(classes_file_path)

        # Remove 2 alleles to create a partial classes file
        partial_classes = classes[:-2]
        partial_classes_file_path = classes_file_path.parent / "partial_classes.txt"
        save_txt(partial_classes, partial_classes_file_path)

        variable = MultiClassVariable(
            name=self.variable_name,
            column=self.column_name,
            classes_file_path=partial_classes_file_path,
        )

        with pytest.raises(ValueError, match="Found 2 unknown class\\(es\\)"):
            variable.fit(ma_dataframe)

    @pytest.mark.parametrize("ma_dataframe", ["lazy", "eager"], indirect=True)
    def test_fit_with_nulls(self, ma_dataframe: pl.DataFrame | pl.LazyFrame) -> None:
        """Test that fitting with null values raises an error."""
        null_df: pl.DataFrame | pl.LazyFrame = pl.DataFrame(
            {"allele": ["HLA__A0103", None, "HLA__C0102"]}
        )
        if isinstance(ma_dataframe, pl.LazyFrame):
            null_df = null_df.lazy()

        variable = MultiClassVariable(name=self.variable_name, column=self.column_name)
        match_msg = (
            f"The default value None for variable '{self.variable_name}' was not provided "
            f"but there are NaN values in the dataset's column '{self.column_name}'."
        )
        with pytest.raises(ValueError, match=match_msg):
            variable.fit(null_df)

    @pytest.mark.parametrize("ma_dataframe", ["lazy", "eager"], indirect=True)
    def test_fit_classes_file_not_found(
        self, ma_dataframe: pl.DataFrame | pl.LazyFrame, tmp_path: Path
    ) -> None:
        """Test that fitting with a non-existent classes file raises an error."""
        non_existent_file_path = tmp_path / "non_existent.txt"

        variable = MultiClassVariable(
            name=self.variable_name,
            column=self.column_name,
            classes_file_path=non_existent_file_path,
        )

        with pytest.raises(FileNotFoundError, match="Classes file not found"):
            variable.fit(ma_dataframe)

    @pytest.mark.parametrize("ma_dataframe", ["lazy", "eager"], indirect=True)
    def test_transform(self, ma_dataframe: pl.DataFrame | pl.LazyFrame) -> None:
        """Test transforming a dataframe."""
        test_df = ma_dataframe.with_columns(
            pl.Series(
                "alleles_with_nan",
                [
                    "HLA__A0201",
                    "HLA__B0702",
                    "HLA__DQA10101=DQB10201",
                    "HLA__A0201",
                    "HLA__A0103",
                    "HLA__A0103",
                    "HLA__B0702",
                    "HLA__DQA10101=DQB10201",
                    "HLA__A0201",
                    "HLA__A0103",
                ],
            )
        )

        # First, check that transforming without fitting raises an error
        variable = MultiClassVariable(name=self.variable_name, column="alleles_with_nan")
        with pytest.raises(NotFittedError, match="The variable 'multiclass_variable' must be fit"):
            variable.transform(ma_dataframe)

        variable.fit(test_df)

        transformed_df = variable.transform(test_df)

        assert isinstance(transformed_df, type(test_df))

        expected_indices = pl.Series(
            self.variable_name, [1, 2, 3, 1, 0, 0, 2, 3, 1, 0], dtype=pl.Int32
        )
        actual_indices = transformed_df.lazy().collect()[self.variable_name]
        assert actual_indices.equals(expected_indices)
