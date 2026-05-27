"""Module to define the entry point of the command line to assign protein-related features."""

from pathlib import Path

import rich_click as click

from bench_mhc.utils.click import arguments


@click.command()
@arguments.peptides_path
@arguments.flank_size
@arguments.human_proteome_path
@arguments.swissprot_db_path
@arguments.output_file_path
@arguments.with_columns
def assign_protein_features(
    peptides_path: Path,
    flank_size: int,
    human_proteome_path: Path,
    swissprot_db_path: Path,
    output_file_path: Path,
    with_columns: set[str] | None,
) -> None:
    """Assign source protein(s) and related features to each peptide using the reference proteome.

    In addition to the source protein, this command assigns flanks to the peptide. If a peptide
    matches to multiple source proteins, multiple flanks are returned.

    The process is the following:

    \b
    1. **Load peptides** from a CSV file (must contain `peptide` column)
    2. **Load human proteome sequences** from a FASTA file
    3. **Load SwissProt sequences** from a FASTA file
    4. For each peptide length (k) group:
       - **Generate k-mers** from human proteome sequences
       - **Match peptides** to source proteins in the human proteome, if possible
       - For **unmatched peptides**, **perform `BLASTP` search** against SwissProt and human
         proteome databases to match peptides to one source protein (or multiple if ties)
       - Assign related **flanks**
    5. Add a **feature `exact_protein_match`** to discriminate exact and imperfect matches
    6. **Output peptides and source protein features** to a CSV file

    The output CSV file contains the following columns:

    \b
    - `peptide`: The peptide sequence
    - `protein_id`: The protein ID(s) containing the peptide
    - `origin_species`: The species of origin for the peptide
      (e.g., "Homo Sapiens", "Mus musculus", etc.)
    - `exact_protein_match`: Whether the peptide has an exact match to a source protein
    - `left_flank`: The left flank(s) associated with the peptide
    - `right_flank`: The right flank(s) associated with the peptide
    - Additional columns specified with `--with_columns` (copied from input CSV)

    Here is an example of an output CSV file with a flank size of 3:

    ```
    peptide,protein_id,origin_species,exact_protein_match, left_flank, right_flank
    EESCEKSEP,ENST00000518069.2;ENST00000611914.1,Homo Sapiens,True,LCV;TLC,AQS;FLA
    LCAQSPLCV,ENST00000611914.1,Homo Sapiens,True,QSA,L--
    FMLLGTLCE,ENST00000518069.2,Homo Sapiens,True,--A,TGT
    VSCVFLAFV,ENST00000518069.2,Homo Sapiens,True,FLL,MML
    SCVFLAFVI,ENST00000611914.1,Homo Sapiens,True,SSA,---
    SCVFLAFVI,ENST00000518069.2,Homo Sapiens,False,IVA,FLL
    CVVNFNILV,ENST00000611914.1,Homo Sapiens,False,-VV,NNV
    AVAAVSCVF,ENST00000518069.2,Homo Sapiens,True,---,---
    ```

    !!! Note

        \b
        - For peptides not found in the human proteome, **BLASTP is used to find similar
          sequences in SwissProt + human proteome**, and peptides are assigned to those matches.
        - If `--with_columns` is provided, the specified columns are copied from peptides to output.
    """  # noqa: D301
    from bench_mhc.cli.assign_protein_features import assign_protein_features

    assign_protein_features(
        peptides_path=peptides_path,
        flank_size=flank_size,
        human_proteome_path=human_proteome_path,
        swissprot_db_path=swissprot_db_path,
        output_file_path=output_file_path,
        with_columns=with_columns,
    )
