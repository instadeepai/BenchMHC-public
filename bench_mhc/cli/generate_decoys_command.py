"""Module to define the entry point of the command line to generate decoys."""

from pathlib import Path

import click

from bench_mhc.utils.click import arguments


@click.command()
@arguments.hits_path
@arguments.human_proteome_path
@arguments.output_file_path
@arguments.num_decoys
@arguments.swissprot_db_path
@arguments.flank_size
@arguments.remove_multiple_matches
@arguments.remove_imperfect_matching
@arguments.nullify_imperfect_flanks
@arguments.with_columns
def generate_decoys(
    hits_path: Path,
    human_proteome_path: Path,
    output_file_path: Path,
    num_decoys: int,
    swissprot_db_path: Path,
    flank_size: int,
    remove_multiple_matches: bool,
    remove_imperfect_matching: bool,
    nullify_imperfect_flanks: bool,
    with_columns: set[str] | None,
) -> None:
    """Generate decoys from hits using proteome sequences.

    This command generates decoy peptides for a given set of hits by sampling from
    proteome sequences. The process is the following:

    \b
    1. **Load peptide hits** from a CSV file (must contain `peptide`, `protein_id`,
    `origin_species`, `allele` columns)
    2. **Load human proteome sequences** from a FASTA file
    3. **Load SwissProt sequences** from a FASTA file
    4. For each peptide length (k) group:
       - **Generate k-mers** from human proteome sequences and SwissProt sequences
       - **Sample decoy peptides** from matched proteins with k length and cysteine content
    5. **Output combined hits and decoys** to a CSV file

    The output CSV file contains the following columns:

    \b
    - `peptide`: The peptide sequence (hit or decoy)
    - `protein_id`: The protein ID(s) containing the peptide (semicolon-separated for hits)
    - `origin_species`: The species of origin for the peptide
      (e.g., "Homo Sapiens", "Mus musculus" etc.)
    - `hit`: Binary indicator (1 for original hit, 0 for decoys)
    - `allele`: The MHC allele associated with the peptide
    - Additional columns specified with `--with_columns` (copied from hits to decoys)

    Here is an example of an output CSV file:

    ```
    peptide,protein_id,origin_species,hit,allele
    EESCEKSEP,ENST00000518069.2;ENST00000611914.1,Homo Sapiens;Homo Sapiens,1,HLA-A*02:01
    LCAQSPLCV,ENST00000611914.1,Homo Sapiens,0,HLA-A*02:01
    FMLLGTLCE,ENST00000518069.2,Homo Sapiens,0,HLA-A*02:01
    VSCVFLAFV,ENST00000518069.2,Homo Sapiens,0,HLA-A*02:01
    SCVFLAFVI,ENST00000611914.1,Homo Sapiens,0,HLA-A*02:01
    SCVFLAFVI,ENST00000518069.2,Homo Sapiens,0,HLA-A*02:01
    CVVNFNILV,ENST00000611914.1,Homo Sapiens,0,HLA-A*02:01
    AVAAVSCVF,ENST00000518069.2,Homo Sapiens,0,HLA-A*02:01
    ```

    !!! Important

        The `assign-protein-features` command must be run first to map the hits
        to their source proteins, origin species and flanks. During the `generate-decoys`
        process, we assume all the hits have been mapped to their source proteins,
        origin species and flanks.

    !!! Note

        The decoy generation ensures that:

        \b
        - Decoys have the **same length** as their corresponding hits.
        - The **cysteine depletion bias is mitigated** by sampling decoys with the **same**
          cysteine content as their hits.
        - Decoys are sampled **from proteins that contain the original hit** peptide.
        - If `--flank_size` is provided, the flanks are added to the decoys.
        - If `--with_columns` is provided, the specified columns are copied from hits to decoys.
        - If `hit` column is present in the original dataset, only rows with `hit=1` are processed.
    """  # noqa: D301
    from bench_mhc.cli.generate_decoys import generate_decoys

    generate_decoys(
        hits_path=hits_path,
        human_proteome_path=human_proteome_path,
        output_file_path=output_file_path,
        num_decoys=num_decoys,
        swissprot_db_path=swissprot_db_path,
        flank_size=flank_size,
        remove_multiple_matches=remove_multiple_matches,
        remove_imperfect_matching=remove_imperfect_matching,
        nullify_imperfect_flanks=nullify_imperfect_flanks,
        with_columns=with_columns,
    )
