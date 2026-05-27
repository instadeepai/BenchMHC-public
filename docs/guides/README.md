# Guides

!!! info

    In this documentation, you will find everything you need to get started with using our
    machine learning models and algorithms.

!!! tip "New to the terminology?"

    See the **[Glossary](glossary.md)** (SA/MA, BA/EL, pMHC, FRANK, decoys, etc.).

## Project overview

**BenchMHC** is a toolkit to reproduce and benchmark peptide–MHC presentation models: you prepare
tabular data (hits, then optional protein context and decoys), optionally add model-specific
peptide and allele features, `train` a model from YAML configuration, then `predict` and
`evaluate` on held-out data.

!!! info "Workflow"

    The usual core workflow is the following:

    ```bash
    hits dataset
      → assign-protein-features
      → generate-decoys
      → train
      → calibrate (optional)
      → predict
      → evaluate
      → generate-performance-plot
    ```

### Command-line commands

#### Core workflow

- **`assign-protein-features`**: Annotate peptides with source protein identifiers and sequence
  flanks for downstream modeling.
- **`generate-decoys`**: Build negative (decoy) examples from hits and apply protein-level filters
  (e.g. remove imperfect matches) for training or evaluation.
- **`train`**: Fit a model from training and validation CSVs using a YAML configuration.
- **`calibrate`**: Calibrate model outputs (e.g. scores or probabilities) after training.
- **`predict`**: Run inference with a trained checkpoint on new peptides.
- **`evaluate`**: Compute metrics from predictions and labels.
- **`generate-performance-plot`**: Generate performance plots from evaluation outputs.

#### Optional commands

- **`refine-peptides`**: Filter and process CSVs already produced by `assign-protein-features`
  without generating decoys.
- **`compute-nnalign-features`**: Add NetMHCpan-style NNAlign peptide features for NNAlign-based
  models.
- **`format-allele-feature`**: Normalize allele names to a consistent format.

Run `bench-mhc --help` and `{command} --help` for full options. Deeper API references live under
[API Reference](../reference/api/index.md).

### Repository layout

For a concise map of `bench_mhc/`, `configuration/`, `data/`, and `docs/`, see
[Repository layout](development/repo_layout.md) in the Development section.

## Guides

- [Glossary](glossary.md) defines domain terms used across docs (immunology and ML).
- [Installation](installation.md) contains all instructions to install the package.
- [Bucket](bucket.md) describes the structure and organization of the Google Storage bucket used for
  storing datasets, models and reference data.
- [Protein features assignment](assign_protein_features.md) explains how to fetch the source
  protein(s) and the flanks of a given peptide.
- [Peptide refinement](refine_peptides.md) explains how to filter and clean enriched peptide tables
  without generating decoys.
- [Decoys generation](decoys_generation.md) explains how to generate decoys from hits.
- [Training](training.md) explains how to train models.
- [Configuration](training_configuration.md) explains how to configure the training process.
- [Inference](inference.md) explains how to predict with and evaluate models.
- [Calibration](calibration.md) explains how to calibrate model outputs.
- [Development](development/README.md) contains everything related to development.
