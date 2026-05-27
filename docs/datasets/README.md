# Datasets

!!! info

    This documentation describes the datasets used in BenchMHC and their versioning system.

## Version Components

- **X (Hit Versioning)**: Increments when hits are modified (filtered, etc.)
- **Y (Decoy Versioning)**: Increments when decoys are changed (ratio, sampling, etc.)
- **Z (CSV Versioning)**: Increments when additional columns are added or removed

| Version | Description | Hit Changes | Decoy Changes            | CSV Changes | Usage |
|---------|-------------|-------------|--------------------------|-------------|-------|
| v0.0.0  | Original NetMHCpan data | - | -                        | - | Training (NetMHCpan-4.1); Evaluation (MS ligands, CD8 epitopes) |
| v1.0.0  | Human-unbiased NetMHCpan data | Unmatched hits removed | Hit:decoy ratio of 1:99 | Additional `protein_id` and `origin_species` columns | - |
| v1.1.0  | Human-unbiased NetMHCpan data | Unmatched hits removed | Hit:decoy ratio of 1:17  | Additional `protein_id`, `origin_species`, `left_flank` and `right_flank` columns (flank_size=5) | Training (NetMHCpan-4.1) |
| v1.2.0  | Human-unbiased NetMHCpan data | Unmatched hits removed | Hit:decoy ratio of 1:20[^1]  | Additional `protein_id`, `origin_species`, `left_flank` and `right_flank` columns (flank_size=5), multiple protein matches removed | - |
| v1.3.0  | Human-unbiased NetMHCpan data | Unmatched hits removed | Hit:decoy ratio of 1:100 | Additional `protein_id`, `origin_species`, `left_flank` and `right_flank` columns (flank_size=5), multiple protein matches removed | - |
| v2.0.0  | Human-unbiased NetMHCpan data | Unmatched hits removed and imperfect matches removed | Hit:decoy ratio of 1:20  | Additional `protein_id`, `origin_species`, `left_flank` and `right_flank` columns (flank_size=5), multiple protein matches removed | Evaluation (MS ligands) |
| v2.1.0  | Human-unbiased NetMHCpan data | Unmatched hits removed and imperfect matches removed | Hit:decoy ratio of 1:100 | Additional `protein_id`, `origin_species`, `left_flank` and `right_flank` columns (flank_size=5), multiple protein matches removed | - |

[^1]: The hit:decoy ratio of 1:20 was set to be consistent with the ratio used for NetMHCpan training.

The idea is to add a new line for each version of the dataset that describes best the specific versioning.
