"""Utils functions relative to blastp."""

from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import polars as pl

from bench_mhc.constants import ROOT_DIRECTORY
from bench_mhc.utils.bash import run_bash
from bench_mhc.utils.logging import system

BLASTP_OUTPUT_COLUMNS = (
    "qseqid",
    "sseqid",
    "stitle",
    "sstart",
    "send",
    "evalue",
    "bitscore",
)


log = system.get(__name__)


def build_blast_db(blast_db_name: str, blast_db_output_path: str, proteome_path: Path) -> None:
    """Build a BLAST database from a proteome FASTA file.

    Creates a protein BLAST database using makeblastdb from the specified proteome
    FASTA file.

    You can download:
    - the swissprot database from:
    https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/

    - the human proteome database from:
    https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/gencode.v48.pc_translations.fa.gz

    Args:
        blast_db_name: The name for the BLAST database to be created.
        blast_db_output_path: Directory path where the BLAST database will be built.
        proteome_path: Path to the FASTA file containing the proteome sequences.
    """
    Path(blast_db_output_path).mkdir(parents=True, exist_ok=True)

    cmd = [
        "makeblastdb",
        "-dbtype",
        "prot",
        "-out",
        blast_db_output_path,
        "-title",
        blast_db_name,
        "-in",
        str(ROOT_DIRECTORY / proteome_path),
    ]
    log.info("Building the database ...")
    run_bash(" ".join(cmd), cwd=blast_db_output_path)

    log.info(f"Database built in {blast_db_output_path}.")


def get_blastp_output(
    peptides: list[str],
    blast_db_name: str,
    blast_db_path: str,
    num_threads: int,
    blastp_output_columns: tuple[str, ...] = BLASTP_OUTPUT_COLUMNS,
    **kwargs: Any,
) -> pl.DataFrame:
    """Run blastp for the peptides list.

    Args:
        peptides: List of peptides.
        blast_db_name: Name of the blast database.
        blast_db_path: Path to where the blast database is built.
        num_threads: Number of computing threads.
        blastp_output_columns: Output column names to use when loading the result.
        kwargs: Additional arguments that can be passed to the blastp command.

    Returns:
        DataFrame with the result of the blast search. Each line corresponds to a peptide
            with its match and the matching metrics. If no match are returned from the command,
            an empty dataframe is returned with the expected columns.
    """
    # These args are the one provided in the BLAST+ documentation
    # suited for small protein sequences e.g. peptides.
    # https://www.ncbi.nlm.nih.gov/books/NBK279684/table/appendices.T.blastp_application_options/
    args = {
        "-db": blast_db_name,
        "-outfmt": '"6 ' + " ".join(BLASTP_OUTPUT_COLUMNS) + '"',
        "-num_threads": str(num_threads),
        "-max_target_seqs": "500",
        "-word_size": "2",
        "-matrix": "PAM30",
        "-window_size": "40",
        "-threshold": "1000",
        "-evalue": "10",
        "-mt_mode": "0",
    }
    args.update([("-" + key, value) for key, value in kwargs.items()])

    with NamedTemporaryFile("r+") as f:
        for peptide in peptides:
            f.write(f">{peptide}\n")
            f.write(f"{peptide}\n")
        f.seek(0)
        cmd = [f"BLASTDB={blast_db_path}", "blastp", "-query", f.name]
        for item in args.items():
            cmd.extend(item)
        log.info(
            f"Running Blastp with {num_threads} threads. "
            "This may take a while for a large number of peptides."
        )
        log.info(" ".join(cmd))
        blastp_output = run_bash(" ".join(cmd))

    # Create proper schema mapping for pl.read_csv
    schema_mapping = {
        col: pl.Int64
        if col in ("sstart", "send")
        else pl.Float64
        if col in ("evalue", "bitscore")
        else pl.String
        for col in blastp_output_columns
    }

    if blastp_output:
        return pl.read_csv(
            StringIO(blastp_output),
            separator="\t",
            schema=schema_mapping,
        )
    else:
        log.warning(
            "No match found from blastp. Returning an empty dataframe. "
            "If this is not expected, try tuning the blastp parameters for better results."
        )

        return pl.DataFrame(schema=schema_mapping)
