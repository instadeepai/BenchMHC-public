"""Unit tests related to bench_mhc/variables/nn_align.py."""

from pathlib import Path
from typing import Literal
from unittest.mock import patch

import polars as pl
import polars.testing as pl_testing
import pytest

from bench_mhc.constants import PAD_TOKEN
from bench_mhc.utils.errors import NotFittedError
from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.io import save_json
from bench_mhc.variables import NNAlignVariable


@pytest.fixture
def nn_align_mapping_file_path(tmp_path: Path) -> str:
    """Define the file path for the mapping to use in tests relative to NNAlignVariable."""
    mapping = {
        "A": "SIXMER",
        "B": "EIGHTMER",
        "C": "NINEMEEER",
        "D": "ITSATENMER",
        "E": "ITSTWELVEMER",
        "F": None,
    }
    mapping_file_path = tmp_path / "nn_align_mapping.json"
    save_json(mapping, mapping_file_path)

    return str(mapping_file_path)


class TestNNAlignVariable:
    """Test cases for the NNAlignVariable class."""

    variable_name = "nn_align_variable"

    def _get_expected_transformed_dataframe(
        self,
        expected_dataframe: pl.DataFrame,
        aa2idx: dict[str, int],
        pad_token: str,
        unk_token: str | None,
    ) -> pl.DataFrame:
        """Get the expected transformed dataframe based on pad_token and unk_token."""
        suffixes = NNAlignVariable.suffixes
        column_name2new_column_name = {
            f"nn_align_{suffix}": f"{self.variable_name}_{suffix}" for suffix in suffixes
        }
        expected_transformed_dataframe = expected_dataframe.rename(column_name2new_column_name)

        for i in range(NNAlignVariable.max_num_cores):
            col_name = f"{self.variable_name}_core_{i}"
            encoded_values = [
                [
                    aa2idx.get(aa, aa2idx.get(unk_token) if unk_token is not None else None)
                    for aa in sequence.replace("-", pad_token)
                ]
                for sequence in expected_transformed_dataframe[col_name].to_list()
            ]
            expected_transformed_dataframe = expected_transformed_dataframe.with_columns(
                pl.Series(
                    col_name,
                    encoded_values,
                    dtype=pl.Array(pl.UInt8, shape=NNAlignVariable.core_len),
                )
            )

        return expected_transformed_dataframe

    @pytest.mark.parametrize("column", [None, "nn_align_column"])
    @pytest.mark.parametrize("aas", [None, ["T", "E", "S", "T"]])
    @pytest.mark.parametrize("unk_token", [None, "X"])
    @pytest.mark.parametrize("max_len", [None, 10])
    @pytest.mark.parametrize("padding_strategy", ["start", "end"])
    @pytest.mark.parametrize("mapping_file_path", [None, "path/to/mapping.json"])
    def test_init_nn_align_variable(
        self,
        column: str | None,
        aas: list[str] | None,
        unk_token: str | None,
        max_len: int | None,
        padding_strategy: Literal["start", "end"],
        mapping_file_path: str | None,
    ) -> None:
        """Test initializing the NNAlign variable."""
        variable = NNAlignVariable(
            name=self.variable_name,
            column=column,
            aas=aas,
            unk_token=unk_token,
            max_len=max_len,
            padding_strategy=padding_strategy,
            mapping_file_path=mapping_file_path,
        )

        assert variable.name == self.variable_name
        assert variable.column == column or self.variable_name

        if aas is not None:
            assert variable.aas == sorted(set(aas))
        else:
            match_msg = (
                "In order to be defined, 'aas' should be provided or the variable "
                f"'{self.variable_name}' must be fit on '{variable.column}' column."
            )
            with pytest.raises(NotFittedError, match=match_msg):
                _ = variable.aas

        assert variable.pad_token == PAD_TOKEN
        assert variable.unk_token == unk_token

        # Check 'max_len' and 'padding_strategy' are set to default values
        assert variable.max_len == variable.core_len
        assert variable.padding_strategy == "end"

        assert variable.mapping_file_path == mapping_file_path

        assert variable.polars_type == pl.UInt8
        assert not variable.is_fitted

        assert variable.max_num_cores == 10
        assert variable.core_len == 9

    def test_init_nn_align_variable_custom_pad_token(self) -> None:
        """Test initializing the NNAlign variable with a custom pad_token."""
        variable = NNAlignVariable(
            name=self.variable_name,
            pad_token="_",
        )
        assert variable.pad_token == "_"

    @pytest.mark.parametrize("dataframe", ["lazy", "eager"], indirect=True)
    @pytest.mark.parametrize("aas", [None, ["T", "E", "S", "X"]])
    @pytest.mark.parametrize("pad_token", [PAD_TOKEN, "_"])
    @pytest.mark.parametrize("unk_token", ["X", "P"])
    @pytest.mark.parametrize("use_mapping_file_path", [False, True])
    def test_fit_transform_nn_align_variable(
        self,
        aas: list[str] | None,
        pad_token: str,
        unk_token: str,
        use_mapping_file_path: bool,
        nn_align_mapping_file_path: str,
        dataframe: pl.DataFrame | pl.LazyFrame,
        expected_nn_align_processed_mhc1_df: pl.DataFrame,
    ) -> None:
        """Test fitting and transforming a dataframe with an NNAlign variable."""
        column = "nn_align_to_map" if use_mapping_file_path else "nn_align"
        variable = NNAlignVariable(
            name=self.variable_name,
            column=column,
            aas=aas,
            pad_token=pad_token,
            unk_token=unk_token,
            mapping_file_path=nn_align_mapping_file_path if use_mapping_file_path else None,
        )

        match_msg = (
            f"The variable '{self.variable_name}' must be fit on '{column}' column "
            "before being used to transform data."
        )
        with pytest.raises(NotFittedError, match=match_msg):
            variable.transform(dataframe)

        variable.fit(dataframe)

        assert variable.is_fitted

        # Check that fit is not called if variable already fitted
        with patch.object(dataframe, "lazy") as lazy_spy:
            variable.fit(dataframe)

        lazy_spy.assert_not_called()

        # Check 'max_len' and 'padding_strategy' are set to default values
        assert variable.max_len == variable.core_len
        assert variable.padding_strategy == "end"

        if aas is not None:
            expected_aas = sorted(set(aas))

        else:
            expected_aas = ["A", "E", "G", "H", "I", "L", "M", "N", "R", "S", "T", "V", "W", "X"]

        if pad_token in expected_aas:
            expected_aas.remove(pad_token)

        expected_aas.insert(0, pad_token)

        if unk_token in expected_aas:
            expected_aas.remove(unk_token)

        expected_aas.insert(1, unk_token)

        expected_aa2idx = {aa: idx for idx, aa in enumerate(expected_aas)}
        assert variable.aas == expected_aas
        assert variable.aa2idx == expected_aa2idx

        expected_dataframe = self._get_expected_transformed_dataframe(
            expected_dataframe=expected_nn_align_processed_mhc1_df,
            aa2idx=expected_aa2idx,
            pad_token=pad_token,
            unk_token=unk_token,
        )

        transformed_dataframe = variable.transform(dataframe)
        assert isinstance(transformed_dataframe, type(dataframe))

        features = variable.features
        assert features.issubset(set(transformed_dataframe.collect_schema().names()))

        transformed_dataframe = transformed_dataframe.lazy().collect()
        for feature in features:
            if "_core_" in feature and "flank" not in feature:
                expected_type: pl.Array | type[pl.UInt8] = pl.Array(
                    variable.polars_type, shape=variable.core_len
                )
            else:
                expected_type = pl.UInt8

            assert (
                transformed_dataframe[feature].dtype == expected_type
            ), f"Wrong type for {feature}"

            pl_testing.assert_series_equal(
                transformed_dataframe[feature],
                expected_dataframe[feature],
                check_dtypes=False,
            )

    def test_transform_nn_align_variable_with_unknown_amino_acids(
        self,
        dataframe: pl.DataFrame | pl.LazyFrame,
    ) -> None:
        """Test transforming a dataframe with unknown amino acids and unk_token is None."""
        variable = NNAlignVariable(
            name=self.variable_name,
            column="nn_align",
            aas=["T", "E", "S", "X"],
            pad_token=PAD_TOKEN,
            unk_token=None,
            max_len=10,
            padding_strategy="start",
            mapping_file_path=None,
        )
        variable.fit(dataframe)

        expected_aas = [PAD_TOKEN, "E", "S", "T", "X"]

        unique_chars = (
            dataframe.lazy()
            .select(pl.col("nn_align").str.split("").explode().unique())
            .filter(pl.col("nn_align").is_not_null())
            .collect(engine="streaming")
            .get_column("nn_align")
        )
        unknown_chars = set(unique_chars) - set(expected_aas)

        with pytest.raises(
            ValueError,
            match=f"Data contains unknown amino acids {format_iterable(unknown_chars)}, "
            f"but `unk_token` is not configured. Please specify a `unk_token` in the "
            "configuration file.",
        ):
            variable.transform(dataframe)
