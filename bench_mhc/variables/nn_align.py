"""Module to define the NNAlign variables."""

import itertools
from collections.abc import Iterable
from typing import Literal

import polars as pl

from bench_mhc.constants import PAD_TOKEN
from bench_mhc.constants import LazyOrDataFrame
from bench_mhc.utils.logging import system
from bench_mhc.variables.aa_seq import AASeqVariable
from bench_mhc.variables.base import maybe_raised_not_fitted

log = system.get(__name__)


class NNAlignVariable(AASeqVariable):
    """Class for the variable which represents the inputs required by NNAlign.

    Attributes:
        _aas: Optional amino acids.
        pad_token: Pad token.
        unk_token: Optional unknown token.
        _max_len: Maximum length.
        padding_strategy: Padding strategy.
        mapping_file_path: Optional mapping file path.
            If set, the values are previously mapped with this mapping.
        aa2idx: Mapping from token to ID.
        name: Name to identify the variable.
        column: Optional column linked to the variable.
            If not provided, the 'name' is used.
        is_fitted: Indicates if the variable has been fitted.
    """

    max_num_cores = 10
    core_len = 9
    suffixes = frozenset(
        {
            "num_possible_cores",
            "insertion_len",
            "deletion_len",
            "is_8mer_or_less",
            "is_9mer",
            "is_10mer",
            "is_11mer_or_more",
        }.union(
            set(
                itertools.chain.from_iterable(
                    (f"core_{i}", f"core_{i}_flank_left_len", f"core_{i}_flank_right_len")
                    for i in range(max_num_cores)
                )
            )
        )
    )

    def __init__(
        self,
        name: str,
        column: str | None = None,
        aas: Iterable[str] | None = None,
        pad_token: str = PAD_TOKEN,
        unk_token: str | None = None,
        max_len: int | None = None,
        padding_strategy: Literal["start", "end"] | None = None,
        mapping_file_path: str | None = None,
    ) -> None:
        """Initialize the NNAlign variable.

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
            max_len: Optional maximum length. Not used for NNAlignVariable.
            padding_strategy: Optional padding strategy. Not used for NNAlignVariable.
            mapping_file_path: Optional mapping file path.
                If provided, the values are previously mapped with this mapping.
        """
        super().__init__(
            name=name,
            column=column,
            aas=aas,
            pad_token=pad_token,
            unk_token=unk_token,
            max_len=self.core_len,
            padding_strategy="end",
            mapping_file_path=mapping_file_path,
        )

        if max_len is not None:
            log.warning(
                f"You provided 'max_len={max_len}' for the NNAlign variable '{self.name}' but it "
                "won't be used in the preprocessing and set by default to "
                f"'core_len={self.core_len}'."
            )

        if padding_strategy is not None:
            log.warning(
                f"You provided 'padding_strategy={padding_strategy}' for the NNAlign variable "
                f"'{self.name}' but it won't be used in the preprocessing and set by default to "
                f"'padding_strategy={self.padding_strategy}'.'"
            )

    @maybe_raised_not_fitted
    def transform(self, df: LazyOrDataFrame) -> LazyOrDataFrame:
        """Transform the variable by adding all columns required by NNAlign.

        It also maps AA to index for the 9mer cores.

        Args:
            df: Dataset.

        Returns:
            The transformed data.
        """
        lf = df.lazy()
        lf = self._maybe_map(lf)

        if self.unk_token is None:
            self._check_unknown_amino_acids(lf)

        lf = lf.pipe(self.build_nn_align_features).with_columns(
            [
                pl.col(f"{self.name}_core_{idx}")
                .str.split(by="")
                .list.eval(
                    pl.element().replace_strict(
                        self.aa2idx,
                        default=self.aa2idx.get(self.unk_token),  # type: ignore
                        return_dtype=pl.UInt8,
                    )
                )
                .cast(pl.Array(self.polars_type, self.core_len))
                for idx in range(self.max_num_cores)
            ]
        )

        return lf if isinstance(df, pl.LazyFrame) else lf.collect(engine="streaming")

    def build_nn_align_features(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Build the NNAlign features.

        Args:
            lf: LazyFrame.

        Returns:
            The transformed LazyFrame.
        """
        column_len = f"{self.name}_len"

        lf = (
            lf.with_columns(pl.col(self.column).replace(None, ""))
            .with_columns(
                [
                    pl.col(self.column).str.len_chars().alias(column_len),
                ]
            )
            .with_columns(
                [
                    (pl.col(column_len) < self.core_len).cast(pl.UInt8).alias("is_8mer_or_less"),
                    (pl.col(column_len) == self.core_len).cast(pl.UInt8).alias("is_9mer"),
                    (pl.col(column_len) == self.core_len + 1).cast(pl.UInt8).alias("is_10mer"),
                    (pl.col(column_len) > self.core_len + 1)
                    .cast(pl.UInt8)
                    .alias("is_11mer_or_more"),
                    (
                        pl.when(pl.col(column_len) < self.core_len)
                        .then(pl.col(column_len) + 1)
                        .when(pl.col(column_len) > self.core_len)
                        .then(self.max_num_cores)
                        .otherwise(1)
                    )
                    .cast(pl.UInt8)
                    .alias("num_possible_cores"),
                    (
                        pl.when(pl.col(column_len) < self.core_len)
                        .then(self.core_len - pl.col(column_len))
                        .otherwise(0)
                    )
                    .cast(pl.UInt8)
                    .alias("insertion_len"),
                    (
                        pl.when(pl.col(column_len) > self.core_len)
                        .then(pl.col(column_len) - self.core_len)
                        .otherwise(0)
                    )
                    .cast(pl.UInt8)
                    .alias("deletion_len"),
                ]
            )
            .with_columns(**self._build_nn_align_cores_expr())
            .drop(column_len)
            # Append the prefix and sort the NNAlign columns
            .with_columns(
                [pl.col(suffix).name.prefix(f"{self.name}_") for suffix in sorted(self.suffixes)]
            )
            .drop(self.suffixes)
        )

        return lf

    def _build_nn_align_cores_expr(self) -> dict[str, pl.Expr]:
        """Build the polars expressions for core-related features.

        Returns:
            The polars expressions for core-related features.
        """
        core_columns_9mers = self._build_cores_9mers()
        core_columns_8mers_or_less = self._build_cores_8mers_or_less()
        core_columns_10mers_or_more = self._build_cores_10mers_or_more()

        core_columns = {}

        column_len = f"{self.name}_len"

        for column_name, core_column_9mers in core_columns_9mers.items():
            core_columns[column_name] = (
                pl.when(pl.col(column_len) == self.core_len)
                .then(core_column_9mers)
                .when(pl.col(column_len) > self.core_len)
                .then(core_columns_10mers_or_more[column_name])
                .otherwise(core_columns_8mers_or_less[column_name])
            )

        return core_columns

    def _build_cores_9mers(self) -> dict[str, pl.Expr]:
        """Build the polars expressions for core-related features for 9mers.

        Returns:
            The polars expressions for core-related features for 9mers.
        """
        core_columns = {
            "core_0": pl.col(self.column),
            "core_0_flank_left_len": pl.lit(0, dtype=pl.UInt8),
            "core_0_flank_right_len": pl.lit(0, dtype=pl.UInt8),
        }
        for idx in range(1, self.max_num_cores):
            core_columns.update(
                {
                    f"core_{idx}": pl.lit(self.pad_token * self.core_len),
                    f"core_{idx}_flank_left_len": pl.lit(0, dtype=pl.UInt8),
                    f"core_{idx}_flank_right_len": pl.lit(0, dtype=pl.UInt8),
                }
            )

        return core_columns

    def _build_cores_8mers_or_less(self) -> dict[str, pl.Expr]:
        """Build the polars expressions for core-related features for kmers with k < 9.

        Returns:
            The polars expressions for core-related features for kmers with k < 9.
        """
        core_columns = {}

        insertion_len2pad_tokens = {
            insertion_len: self.pad_token * insertion_len
            for insertion_len in range(0, self.core_len + 1)
        }

        for idx in range(self.max_num_cores):
            core_columns[f"core_{idx}"] = (
                pl.when(idx < pl.col("num_possible_cores"))
                .then(
                    pl.concat_str(
                        [
                            pl.col(self.column).str.slice(0, idx),
                            pl.col("insertion_len").replace_strict(insertion_len2pad_tokens),
                            pl.col(self.column).str.slice(idx),
                        ],
                        separator="",
                    )
                )
                .otherwise(pl.lit(self.pad_token * self.core_len))
            )

            core_columns[f"core_{idx}_flank_left_len"] = pl.lit(0, dtype=pl.UInt8)
            core_columns[f"core_{idx}_flank_right_len"] = pl.lit(0, dtype=pl.UInt8)

        return core_columns

    def _build_cores_10mers_or_more(self) -> dict[str, pl.Expr]:
        """Build the polars expressions for core-related features for kmers with k > 9.

        Returns:
            The polars expressions for core-related features for kmers with k > 9.
        """
        core_columns = {}

        for idx in range(self.max_num_cores):
            core_columns[f"core_{idx}"] = pl.concat_str(
                [
                    pl.col(self.column).str.slice(0, idx),
                    pl.col(self.column).str.slice(idx + pl.col("deletion_len")),
                ],
                separator="",
            )
            if idx == 0:
                core_columns[f"core_{idx}_flank_left_len"] = pl.col("deletion_len")
            else:
                core_columns[f"core_{idx}_flank_left_len"] = pl.lit(0, dtype=pl.UInt8)

            if idx == 9:
                core_columns[f"core_{idx}_flank_right_len"] = pl.col("deletion_len")
            else:
                core_columns[f"core_{idx}_flank_right_len"] = pl.lit(0, dtype=pl.UInt8)

        return core_columns

    @property
    def features(self) -> set[str]:
        """Features of the variable when the dataset is transformed."""
        return {f"{self.name}_{suffix}" for suffix in sorted(self.suffixes)}
