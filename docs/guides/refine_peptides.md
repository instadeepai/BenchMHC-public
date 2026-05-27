# Peptide Refinement

The `refine-peptides` [command](../reference/api/refine_peptides.md) allows you to filter and
process a peptide dataset. It applies the same filtering
options as [`generate-decoys`](../guides/decoys_generation.md), but **without generating decoys**.

This is useful to apply protein-level filters (e.g., remove imperfect matches, remove peptides
mapped to multiple proteins, nullify imperfect flanks) to any dataset without (re-)generating
decoys. For instance, models that consume flanking sequences and source protein information
can use `refine-peptides` to prepare the reference set (see [Training on multi-allelic data with
iterative annotation](../guides/training.md#training-on-single-allelic-sa-and-multi-allelic-ma-data-with-iterative-annotation)).

## Prerequisites

### Data preparation

The input CSV file must contain the columns produced by `assign-protein-features`:

- `peptide`: The peptide sequence
- `protein_id`: The protein ID(s) containing the peptide (semicolon-separated if multiple)
- `origin_species`: The species of origin for the peptide
- `exact_protein_match`: Whether the match to the source protein is exact
- `left_flank` and `right_flank`: The flanking sequences (if flanks are relevant)

If your data only contains `peptide` (and optionally `allele`) columns, you need to first run the
`assign-protein-features` [command](../guides/assign_protein_features.md).

## Usage

```bash
refine-peptides --peptides_path data/my_dataset.csv --flank_size 5 -o data/my_refined_dataset.csv
```

### Filtering options

You can combine the following flags depending on your use case:

- `--remove_imperfect_matching`: Removes peptides that could not be exactly matched to a source
  protein in the reference proteome (i.e., `exact_protein_match == 0`).
- `--nullify_imperfect_flanks`: Replaces flanking sequences derived from imperfect alignments with
  padding tokens.
- `--remove_multiple_matches`: When a peptide maps to multiple source proteins, randomly samples
  one protein (with its associated species and flanks) instead of keeping all matches.

!!! warning "Mutually exclusive flags"
    `--remove_imperfect_matching` and `--nullify_imperfect_flanks` cannot be used together.

### Column selection

By default, only the core columns (`peptide`, `protein_id`, `origin_species`,
`exact_protein_match`, and optionally `left_flank`/`right_flank`) are included in the output. Use
`--with_columns` to carry over additional columns from the input:

```bash
refine-peptides --peptides_path data/my_dataset.csv --flank_size 5 --with_columns allele --with_columns hit -o data/my_refined_dataset.csv
```

Use `--with_columns all` to include all available columns from the input.
