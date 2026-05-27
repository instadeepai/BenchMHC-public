"""Unit tests related to bench_mhc/variables/aa_seq.py."""

import itertools
from typing import Literal
from unittest.mock import patch

import polars as pl
import pytest

from bench_mhc.constants import NATURAL_AAS
from bench_mhc.constants import PAD_TOKEN
from bench_mhc.utils.errors import NotFittedError
from bench_mhc.utils.format import format_iterable
from bench_mhc.variables import AASeqVariable


@pytest.fixture
def dipeptide_vocabulary() -> set[str]:
    """Di-peptide vocabulary for the test relative to AASeqVariable with a too large vocabulary."""
    return {"".join(dipeptide) for dipeptide in itertools.product(NATURAL_AAS, repeat=2)}


class TestAASeqVariable:
    """Test cases for the AASeqVariable class."""

    variable_name = "aa_seq_variable"

    @staticmethod
    def _get_expected_transformed_aa_sequence(
        aa_seq: str | None,
        aa2idx: dict[str, int],
        pad_token: str,
        unk_token: str | None,
        max_len: int,
        padding_strategy: Literal["start", "end"],
    ) -> list[int | None]:
        """Get the expected transformed sequence of AAs.

        Args:
            aa_seq: The string of AAs to transform.
            aa2idx: The mapping from AA to index.
            pad_token: The token to use for padding.
            unk_token: The token to use for padding.
            max_len: The maximum length of the transformed value.
            padding_strategy: The padding strategy to use.

        Returns:
            The value transformed in a list of int.
        """
        if aa_seq is None:
            aa_seq = pad_token

        indices = [
            aa2idx.get(aa, aa2idx.get(unk_token) if unk_token is not None else None)
            for aa in aa_seq
        ]

        if padding_strategy == "end":
            indices = indices[:max_len] + [aa2idx.get(pad_token)] * (max_len - len(aa_seq))
        else:
            if max_len <= len(aa_seq):
                indices = indices[len(aa_seq) - max_len :]
            else:
                indices = [aa2idx.get(pad_token)] * (max_len - len(aa_seq)) + indices

        return indices

    @pytest.mark.parametrize("column", [None, "aa_seq_column"])
    @pytest.mark.parametrize("aas", [None, ["T", "E", "S", "T"]])
    @pytest.mark.parametrize("unk_token", [None, "X"])
    @pytest.mark.parametrize("max_len", [None, 3])
    @pytest.mark.parametrize("padding_strategy", ["start", "end"])
    @pytest.mark.parametrize("mapping_file_path", [None, "path/to/mapping.json"])
    def test_init_aa_seq_variable(
        self,
        column: str | None,
        aas: list[str] | None,
        unk_token: str | None,
        max_len: int | None,
        padding_strategy: Literal["start", "end"],
        mapping_file_path: str | None,
    ) -> None:
        """Test initializing the AA-sequence variable."""
        variable = AASeqVariable(
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

        if max_len is not None:
            assert variable.max_len == max_len
        else:
            match_msg = (
                "In order to be defined, 'max_len' should be provided or the variable "
                f"'{self.variable_name}' must be fit on '{variable.column}' column."
            )
            with pytest.raises(NotFittedError, match=match_msg):
                _ = variable.max_len

        assert variable.padding_strategy == padding_strategy
        assert variable.mapping_file_path == mapping_file_path

        assert variable.polars_type == pl.UInt8
        assert not variable.is_fitted

    def test_init_aa_seq_variable_custom_pad_token(self) -> None:
        """Test initializing the AA-sequence variable with a custom pad_token."""
        variable = AASeqVariable(
            name=self.variable_name,
            pad_token="_",
        )
        assert variable.pad_token == "_"

    @pytest.mark.parametrize("dataframe", ["lazy", "eager"], indirect=True)
    @pytest.mark.parametrize("aas", [None, ["T", "E", "S", "T"]])
    @pytest.mark.parametrize("pad_token", [PAD_TOKEN, "_"])
    @pytest.mark.parametrize("unk_token", ["X", "P"])
    @pytest.mark.parametrize("max_len", [None, 3, 10])
    @pytest.mark.parametrize("padding_strategy", ["start", "end"])
    @pytest.mark.parametrize("use_mapping_file_path", [False, True])
    def test_fit_transform_aa_seq_variable(
        self,
        aas: list[str] | None,
        pad_token: str,
        unk_token: str,
        max_len: int | None,
        padding_strategy: Literal["start", "end"],
        use_mapping_file_path: bool,
        aa_seq_mapping_file_path: str,
        dataframe: pl.DataFrame | pl.LazyFrame,
    ) -> None:
        """Test fitting and transforming a dataframe with an AA-sequence variable."""
        column = "aa_seq_to_map" if use_mapping_file_path else "aa_seq"
        variable = AASeqVariable(
            name=self.variable_name,
            column=column,
            aas=aas,
            pad_token=pad_token,
            unk_token=unk_token,
            max_len=max_len,
            padding_strategy=padding_strategy,
            mapping_file_path=aa_seq_mapping_file_path if use_mapping_file_path else None,
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

        if max_len is not None:
            assert variable.max_len == max_len
        else:
            assert variable.max_len == 4

        if aas is not None:
            expected_aas = sorted(set(aas))

        else:
            expected_aas = ["E", "P", "S", "T"]

        if pad_token in expected_aas:
            expected_aas.remove(pad_token)

        expected_aas.insert(0, pad_token)

        if unk_token in expected_aas:
            expected_aas.remove(unk_token)

        expected_aas.insert(1, unk_token)

        expected_aa2idx = {aa: idx for idx, aa in enumerate(expected_aas)}
        assert variable.aas == expected_aas
        assert variable.aa2idx == expected_aa2idx

        expected_serie = pl.Series(
            [
                self._get_expected_transformed_aa_sequence(
                    aa_seq=aa_seq,
                    aa2idx=expected_aa2idx,
                    pad_token=pad_token,
                    unk_token=unk_token,
                    max_len=variable.max_len,
                    padding_strategy=padding_strategy,
                )
                for aa_seq in dataframe.lazy().collect()["aa_seq"].to_list()
            ]
        )

        transformed_dataframe = variable.transform(dataframe)
        assert isinstance(transformed_dataframe, type(dataframe))
        assert transformed_dataframe.lazy().collect()[self.variable_name].equals(expected_serie)

    def test_fit_aa_seq_variable_too_large_vocabulary(
        self,
        dipeptide_vocabulary: set[str],
        dataframe: pl.DataFrame | pl.LazyFrame,
    ) -> None:
        """Test a ValueError is raised if fitting AASeqVariable with a too large vocabulary."""
        variable = AASeqVariable(
            name=self.variable_name,
            column="aa_seq",
            aas=dipeptide_vocabulary,
            pad_token=PAD_TOKEN,
            unk_token=None,
            max_len=10,
            padding_strategy="start",
            mapping_file_path=None,
        )

        assert len(dipeptide_vocabulary) > pl.select(variable.polars_type.max()).item() + 1

        match_msg = (
            f"The size of the vocabulary is {len(dipeptide_vocabulary) + 1} which is too large "
            f"to be encoded in {variable.polars_type}."
        )
        with pytest.raises(ValueError, match=match_msg):
            variable.fit(dataframe)

    def test_fit_aa_seq_variable_value_unknown_in_mapping(
        self,
        aa_seq_mapping_file_path: str,
        dataframe: pl.DataFrame | pl.LazyFrame,
    ) -> None:
        """Test a ValueError is raised if fitting AASeqVariable with an unknown value to map."""
        variable = AASeqVariable(
            name=self.variable_name,
            column="aa_seq_to_map_unknown_key",
            aas=None,
            pad_token=PAD_TOKEN,
            unk_token=None,
            max_len=10,
            padding_strategy="start",
            mapping_file_path=aa_seq_mapping_file_path,
        )

        match_msg = (
            f"The following values are not in the mapping file '{aa_seq_mapping_file_path}': "
            "'unknown'"
        )
        with pytest.raises(ValueError, match=match_msg):
            variable.fit(dataframe)

    def test_transform_aa_seq_variable_with_unknown_amino_acids(
        self,
        dataframe: pl.DataFrame | pl.LazyFrame,
    ) -> None:
        """Test transforming a dataframe with unknown amino acids and unk_token is None."""
        variable = AASeqVariable(
            name=self.variable_name,
            column="aa_seq",
            aas=["T", "E", "S", "T"],
            pad_token=PAD_TOKEN,
            unk_token=None,
            max_len=10,
            padding_strategy="start",
            mapping_file_path=None,
        )
        variable.fit(dataframe)

        expected_aas = [PAD_TOKEN, "E", "S", "T"]

        unique_chars = (
            dataframe.lazy()
            .select(pl.col("aa_seq").str.split("").explode().unique())
            .filter(pl.col("aa_seq").is_not_null())
            .collect(engine="streaming")
            .get_column("aa_seq")
        )
        unknown_chars = set(unique_chars) - set(expected_aas)

        with pytest.raises(
            ValueError,
            match=f"Data contains unknown amino acids {format_iterable(unknown_chars)}, "
            f"but `unk_token` is not configured. Please specify a `unk_token` in the "
            "configuration file.",
        ):
            variable.transform(dataframe)
