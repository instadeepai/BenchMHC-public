# Decoys Generation

The `generate_decoys` [command](../reference/api/generate_decoys.md) allows you to generate
decoy peptides from hit peptides using proteome databases. This is useful for creating
negative samples for model training and evaluation.

## Prerequisites

### Proteome databases

Before generating decoys, you need to download and prepare the required proteome databases:

- **SwissProt database**: [link](https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/)
- **Human proteome database**: [link](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/gencode.v48.pc_translations.fa.gz)

After downloading, make sure to unzip the databases to get the fasta files.

### Data preparation

The input CSV file for `generate-decoys` must contain the following columns:

- `peptide`: The peptide sequence
- `allele`: The MHC allele associated with the peptide
- `protein_id`: The protein ID(s) containing the peptide (semicolon-separated if multiple)
- `origin_species`: The species of origin for the peptide (e.g., "Homo sapiens", "Mus musculus")
- `exact_protein_match`: Whether the match source protein assignment is exact.

If your data only contains `peptide` and `allele` columns, you need to first run the
`assign-protein-features` [command](../guides/assign_protein_features.md) to generate the required
`protein_id`, `origin_species`, `exact_protein_match`, `left_flank` and `right_flank` columns.
Make sure to specify which columns from your input CSV you want to keep in the output using the
`--with-columns` argument.

This will create a new CSV file with the additional columns needed for decoy generation.

## Filtering options

`generate-decoys` supports the same filtering options as `refine-peptides`.
See the [filtering options](../guides/refine_peptides.md#filtering-options) section for details.

## Launch the generation

You can use the command line with an input csv file containing the required columns.
It will generate a dataset containing decoys.

Use the following:

```bash
generate-decoys --hits_path data/dummy_data.csv --human_proteome_path data/hg38.fasta --swissprot_db_path data/uniprot_sprot.fasta --num_decoys 100 -o dummy_output.csv
```
