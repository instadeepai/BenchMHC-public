# API Reference

This section provides detailed API documentation for the `bench-mhc` package,
combining both command-line interface (CLI) documentation and API reference for each module.

## Modules

- [Assign Protein Features Module](assign_protein_features.md): module to assign their source
  protein(s) and flanks to given peptides.
- [Generate Decoys Module](generate_decoys.md): module to generate negative (decoy) examples from
  hits.
- [Train Module](train.md): module to train a model.
- [Calibrate Module](calibrate.md): module to calibrate a model.
- [Predict Module](predict.md): module to make predictions with a trained model.
- [Evaluate Module](evaluate.md): module to evaluate predictions on a dataset.
- [Generate Performance Plot Module](generate_performance_plot.md): module to generate a performance
  plot from evaluation outputs.
- [Refine Peptides Module](refine_peptides.md): module to filter and refine peptides.
- [Compute Allele NNAlign Features Module](compute_allele_nnalign_features.md): module to compute
  NetMHCpan-style NNAlign features.
- [Format Allele Feature Module](format_allele_feature.md): module to format allele names.

## Documentation structure

Each module's documentation includes:

- Command-line arguments and options
- Usage examples
- Configuration options
- Function signatures
- Parameter descriptions
- Return value descriptions
- Example usage where applicable

## Usage

The documentation is organized to show both how to use the package via command line
and how to use it programmatically via its Python API. Each module's page contains:

1. A CLI section showing the command-line interface using mkdocs-click
2. An API Reference section showing the Python functions and their documentation

This structure allows users to:

- Quickly find command-line usage information
- Access detailed API documentation for programmatic use
- Understand the relationship between CLI commands and their underlying functions
