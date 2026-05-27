# Protein Features Assignment

The `assign_protein_features` [command](../reference/api/assign_protein_features.md) allows you to
fetch the source protein(s) of a given peptide using proteome databases and extract its flanks.

## Prerequisites

Before assigning source proteins to peptides, you need to download and
prepare the required proteome databases:

- **SwissProt database**: [link](https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/)
- **Human proteome database**: [link](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/gencode.v48.pc_translations.fa.gz)

After downloading, make sure to unzip the databases to get the fasta files.

You also need to install the [BLAST+](../guides/installation.md#requirements) library.

## Launch the generation

You can use the command line with an input csv file containing a `peptide` column.
It will generate a dataset containing the source protein(s) and its flanks.

Use the following:

```bash
assign-protein-features --peptides_path data/dummy_data.csv --flank_size 5 --human_proteome_path data/hg38.fasta --swissprot_db_path data/uniprot_sprot.fasta -o dummy_output.csv
```

!!! warning "Warning"
    Make sure to specify which columns from your input CSV you want to keep in the output using the
    `--with_columns` argument. Otherwise, only the `peptide` column and the newly generated columns
    (`protein_id`, `origin_species`, `flank_left` and `flank_right`) will be present in the output
    file.
