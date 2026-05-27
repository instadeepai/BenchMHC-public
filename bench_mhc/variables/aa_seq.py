"""Module to define the variable for a sequence of amino acids."""

from collections.abc import Iterable
from typing import Literal

import polars as pl

from bench_mhc.constants import PAD_TOKEN
from bench_mhc.constants import LazyOrDataFrame
from bench_mhc.utils.errors import NotFittedError
from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.io import load_json
from bench_mhc.utils.logging import system
from bench_mhc.variables.base import BaseVariable
from bench_mhc.variables.base import maybe_raised_not_fitted
from bench_mhc.variables.base import skip_if_fitted

log = system.get(__name__)


class AASeqVariable(BaseVariable):
    """Class for the variable which represents a sequence of AAs.

    Attributes:
        _aas: Optional amino acids.
        pad_token: Pad token.
        unk_token: Optional unknown token.
        _max_len: Optional maximum length.
        padding_strategy: Padding strategy.
        mapping_file_path: Optional mapping file path.
            If set, the values are previously mapped with this mapping.
        aa2idx: Mapping from token to ID.
        name: Name to identify the variable.
        column: Optional column linked to the variable.
            If not provided, the 'name' is used.
        is_fitted: Indicates if the variable has been fitted.
    """

    def __init__(
        self,
        name: str,
        column: str | None = None,
        aas: Iterable[str] | None = None,
        pad_token: str = PAD_TOKEN,
        unk_token: str | None = None,
        max_len: int | None = None,
        padding_strategy: Literal["start", "end"] = "end",
        mapping_file_path: str | None = None,
    ) -> None:
        """Initialise the AASeqVariable.

        Args:
            name: Name to identify the variable.
            column: Optional column linked to the variable.
                If not provided, the 'name' is used.
            aas: Optional amino acids.
                If not provided, the 'aas' are inferred during 'fit'.
            pad_token: Optional pad token.
                If not provided, the default 'PAD_TOKEN' is used.
                The 'pad_token' will be inserted at the beginning of the vocabulary.
            unk_token: Optional unknown token.
                If provided, 'unk_token' will be inserted at the second position of the vocabulary.
            max_len: Optional maximum length.
                If not provided, the 'max_len' is inferred during 'fit'.
            padding_strategy: Optional padding strategy.
            mapping_file_path: Optional mapping file path.
                If provided, the values are previously mapped with this mapping.
        """
        super().__init__(name=name, column=column)
        self._aas = sorted(set(aas)) if aas is not None else None
        self.pad_token = pad_token
        self.unk_token = unk_token
        self._max_len = max_len
        self.padding_strategy = padding_strategy
        self.mapping_file_path = mapping_file_path

        # Built during the fit
        self.aa2idx: dict[str, int] | None = None

    @property
    def max_len(self) -> int:
        """Get maximum length of the sequence."""
        if self._max_len is None:
            raise NotFittedError(
                "In order to be defined, 'max_len' should be provided or the variable "
                f"'{self.name}' must be fit on '{self.column}' column."
            )

        return self._max_len

    @property
    def aas(self) -> Iterable[str]:
        """Get amino acid vocabulary."""
        if self._aas is None:
            raise NotFittedError(
                "In order to be defined, 'aas' should be provided or the variable "
                f"'{self.name}' must be fit on '{self.column}' column."
            )

        return self._aas

    def _maybe_map(self, lf: LazyOrDataFrame) -> LazyOrDataFrame:
        """Maybe map the values based on the provided mapping.

        Args:
            lf: LazyOrDataFrame.

        Returns:
            The maybe mapped LazyOrDataFrame.
        """
        if self.mapping_file_path is not None:
            mapping = load_json(self.mapping_file_path)

            values = set(
                lf.lazy()
                .select(pl.col(self.column))
                .unique()
                .collect(engine="streaming")[self.column]
            )
            if not values.issubset(mapping.keys()):
                raise ValueError(
                    f"The following values are not in the mapping file '{self.mapping_file_path}': "
                    f"{format_iterable(values - mapping.keys())}"
                )

            lf = lf.with_columns(pl.col(self.column).replace_strict(mapping))

        return lf

    @skip_if_fitted
    def fit(self, df: LazyOrDataFrame) -> None:
        """Fit the AA seq variable on the provided data.

        1. If `max_len` is `None`, infer it
        2. If `aas` is `None`, infer it
        3. Build mapping `aa2idx`
        4. Maybe safe insert padding and unknown tokens

        Args:
            df: Dataset.
        """
        lf = df.lazy()

        lf = self._maybe_map(lf)

        # Replace null values with PAD token
        lf = lf.with_columns(pl.col(self.column).fill_null(pl.lit(self.pad_token)))

        if self._max_len is None:
            self._max_len = (
                lf.select(pl.col(self.column).str.len_chars().max())
                .collect(engine="streaming")
                .item()
            )

        if self._aas is None:
            aas = sorted(
                lf.select(pl.col(self.column).str.split("").explode().unique())
                .collect(engine="streaming")
                .get_column(self.column)
            )
            log.info(
                f"Vocabulary for variable '{self.name}' inferred from the variable column: "
                f"{format_iterable(aas)}."
            )
        else:
            aas = list(self.aas)

        self._safe_insert_token_in_vocabulary(vocabulary=aas, token=self.pad_token, position=0)

        if self.unk_token is not None:
            self._safe_insert_token_in_vocabulary(vocabulary=aas, token=self.unk_token, position=1)

        if len(aas) > pl.select(self.polars_type.max()).item() + 1:
            raise ValueError(
                f"The size of the vocabulary is {len(aas)} which is too large to be encoded in "
                f"{self.polars_type}."
            )

        self._aas = aas

        self.aa2idx = {aa: idx for idx, aa in enumerate(self.aas)}

        self.is_fitted = True

    @maybe_raised_not_fitted
    def transform(self, df: LazyOrDataFrame) -> LazyOrDataFrame:
        """Transform the AA seq variable on the provided data.

        1. Maybe map the values based on the provided mapping
        2. Truncate and pad
        3. Tokenize
        4. Map AA to index

        Null values are replaced by sequences of pad tokens (equivalent to empty sequences).

        Args:
            df: Dataset.

        Returns:
            The transformed data.

        Raises:
            NotFittedError if the variable is not fitted yet.
        """
        lf = df.lazy()
        lf = self._maybe_map(lf)

        lf = lf.with_columns(pl.col(self.column).fill_null(pl.lit(self.pad_token)))

        if self.unk_token is None:
            self._check_unknown_amino_acids(lf)

        lf = lf.with_columns(
            self._truncate_and_pad_expr()
            .str.split(by="")
            .list.eval(
                pl.element().replace_strict(
                    self.aa2idx,
                    default=self.aa2idx.get(self.unk_token),  # type: ignore
                    return_dtype=pl.UInt8,
                )
            )
            .cast(pl.Array(self.polars_type, self.max_len))
            .alias(self.name)
        )

        return lf if isinstance(df, pl.LazyFrame) else lf.collect(engine="streaming")

    def _truncate_and_pad_expr(self) -> pl.Expr:
        """Get a polars expression to truncate and pad data.

        Returns:
            The polars expression.
        """
        if self.padding_strategy == "start":
            expr = (
                pl.col(self.column)
                .str.slice(-self.max_len)
                .str.pad_start(self.max_len, fill_char=self.pad_token)
            )
        else:
            expr = (
                pl.col(self.column)
                .str.slice(0, self.max_len)
                .str.pad_end(self.max_len, fill_char=self.pad_token)
            )

        return expr

    @staticmethod
    def _safe_insert_token_in_vocabulary(
        vocabulary: list[str], token: str, position: int = 0
    ) -> None:
        """Insert a token at a position in the vocabulary, moving the token if it already exists.

        Args:
            vocabulary: The vocabulary to expand.
            token: The new token to be inserted in the vocabulary.
            position: The position in the vocabulary where to insert the new token.
        """
        if token in vocabulary:
            vocabulary.remove(token)

        vocabulary.insert(position, token)

    @property
    def polars_type(self) -> type[pl.UInt8]:
        """Type of the AA IDs in polars."""
        return pl.UInt8

    @maybe_raised_not_fitted
    def _check_unknown_amino_acids(self, lf: pl.LazyFrame) -> None:
        """Check if the data contains unknown amino acids.

        Args:
            lf: LazyFrame containing the data to check.

        Raises:
            ValueError: If data contains unknown amino acids and `unk_token` is not configured.
        """
        unique_chars = (
            lf.select(pl.col(self.column).str.split("").explode().unique())
            .filter(pl.col(self.column).is_not_null())
            .collect(engine="streaming")
            .get_column(self.column)
        )

        unknown_chars = set(unique_chars) - self.aa2idx.keys()  # type: ignore
        if unknown_chars:
            raise ValueError(
                f"Data contains unknown amino acids {format_iterable(unknown_chars)}, but "
                f"`unk_token` is not configured. Please specify a `unk_token` in the configuration "
                "file."
            )
