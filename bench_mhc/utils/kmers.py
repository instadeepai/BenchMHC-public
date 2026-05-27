"""Utils functions to generate k-mers from proteome."""

from pathlib import Path

import polars as pl

from bench_mhc.constants import PAD_TOKEN
from bench_mhc.utils.io import load_fasta


def generate_kmers_from_proteome(
    df_proteome: pl.DataFrame,
    kmer_length: int,
    include_cysteine_info: bool = False,
    flank_size: int = 0,
) -> pl.LazyFrame:
    """Generate k-mers from a proteome DataFrame.

    This function extracts all possible k-mers (peptides of length k) from protein sequences
    in the proteome DataFrame and returns a LazyFrame containing the k-mers containing or not
    cysteine and source protein information.

    Args:
        df_proteome: DataFrame containing protein sequences with columns 'sequence' and
            'protein_id'.
        kmer_length: Length of peptides (k-mers) to extract from each protein sequence.
        include_cysteine_info: Whether to include a boolean column indicating if the peptide
            contains cysteine. Default to False.
        flank_size: Size of the flanks to include in the protein sequence. Default to 0.

    Returns:
        A LazyFrame containing k-mers with columns:
            - Mandatory columns:
                - `peptide`: k-mer sequence
                - `protein_id`: Source protein ID
            - Optional columns:
                - `with_c`: Boolean indicating if the peptide contains cysteine
                - `left_flank`: The left flank(s) associated with the peptide
                - `right_flank`: The right flank(s) associated with the peptide

    Note:
        Only proteins with sequences longer than or equal to kmer_length are processed.
    """
    lf = (
        df_proteome.lazy()
        .filter(pl.col("sequence").str.len_bytes() >= kmer_length)
        .with_columns(
            pl.int_ranges(start=0, end=pl.col("sequence").str.len_bytes() - kmer_length + 1).alias(
                "start_index"
            )
        )
        .explode("start_index")
        .with_columns(
            pl.col("sequence").str.slice(pl.col("start_index"), length=kmer_length).alias("peptide")
        )
    )

    columns_to_keep = ["peptide", "protein_id"]

    if include_cysteine_info:
        columns_to_keep += ["with_c"]
        lf = lf.with_columns(pl.col("peptide").str.contains("C").alias("with_c"))

    if flank_size != 0:
        lf = (
            # Pad directly the source protein to manage peptides on the edges
            lf.with_columns(
                pl.concat_str(
                    [
                        pl.lit(PAD_TOKEN * flank_size),
                        pl.col("sequence"),
                        pl.lit(PAD_TOKEN * flank_size),
                    ]
                ).alias("sequence"),
            )
            # Extract flanks
            .with_columns(
                pl.col("sequence")
                .str.slice((pl.col("start_index")), flank_size)
                .alias("left_flank"),
                pl.col("sequence")
                .str.slice(pl.col("start_index") + kmer_length + flank_size, flank_size)
                .alias("right_flank"),
            )
        )

        columns_to_keep += ["left_flank", "right_flank"]

    return lf.select(columns_to_keep)


def generate_kmers_sampling_pool(lf_kmers: pl.LazyFrame) -> dict[bool, dict[str, set[str]]]:
    """Generate a sampling pool for k-mers from a k-mers LazyFrame.

    Args:
        lf_kmers: LazyFrame containing k-mers with columns 'protein_id', 'peptide', and 'with_c'.

    Returns:
        A dictionary mapping cysteine presence (True/False) to a dictionary of protein IDs
            and their associated peptides.
    """
    protein_id2peptides_with_c = dict(
        lf_kmers.filter(pl.col("with_c"))
        .group_by("protein_id")
        .agg("peptide")
        .collect(engine="streaming")
        .rows()
    )
    protein_id2peptides_without_c = dict(
        lf_kmers.filter(~pl.col("with_c"))
        .group_by("protein_id")
        .agg("peptide")
        .collect(engine="streaming")
        .rows()
    )

    return {
        True: {
            protein_id: set(peptides) for protein_id, peptides in protein_id2peptides_with_c.items()
        },
        False: {
            protein_id: set(peptides)
            for protein_id, peptides in protein_id2peptides_without_c.items()
        },
    }


def load_swissprot_proteome(swissprot_db_path: Path) -> pl.DataFrame:
    """Load and format SwissProt dataset.

    The sequences containing a `*` (stop codon) are ignored.

    Args:
        swissprot_db_path: Path to the SwissProt FASTA file.

    Returns:
        SwissProt lazyframe with columns `protein_id` and `sequence`.
    """
    return (
        load_fasta(swissprot_db_path, columns=["stitle", "sequence"])
        .lazy()
        .filter(~pl.col("sequence").str.contains("*", literal=True))
        .with_columns(
            pl.col("stitle")
            .str.split_exact(" ", 1)
            .struct.rename_fields(["protein_id", "description"])
        )
        .unnest("stitle")
        .select(pl.col("protein_id"), pl.col("sequence"))
        .collect(engine="streaming")
    )


def load_human_proteome(human_proteome_path: Path) -> pl.DataFrame:
    """Load and format human proteome dataset.

    The sequences containing a `*` (stop codon) are ignored.

    Args:
        human_proteome_path: Path to the human proteome FASTA file.

    Returns:
        Human proteome lazyframe with columns `protein_id` and `sequence`.
    """
    return load_fasta(human_proteome_path, columns=["protein_id", "sequence"]).filter(
        ~pl.col("sequence").str.contains("*", literal=True)
    )
