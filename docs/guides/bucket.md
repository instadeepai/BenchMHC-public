# BenchMHC Google Storage Bucket Structure

This document describes the organization and structure of the `gs://bench-mhc` public Google Storage
bucket, which contains datasets, models, and reference data for the BenchMHC project.

## Bucket Overview

The `gs://bench-mhc`
[bucket](https://console.cloud.google.com/storage/browser/bench-mhc)
serves as the central data repository for the BenchMHC project, containing:

- **Datasets**: Various versions of processed MHC binding data
- **Models**: Trained machine learning models for peptide-MHC binding / presentation prediction
- **Metrics**: Metrics files related to a model and a dataset
- **Reference Data**: Proteome sequences, random reference sets, and curated datasets
- **External Data**: Data from external sources like IEDB, PRIDE, and SHERPA

## Directory Structure

```bash
gs://bench-mhc/
├── data/                          # Main data directory with versioned datasets
│   ├── v0.0.0/                    # Original NetMHCpan data
│   ├── v1.0.0/                    # Human-unbiased data (1:100 ratio)
│   └── v1.1.0/                    # Human-unbiased data (1:17 ratio)
│   ...
├── metrics/                       # All metrics files produced by the evaluate command
│   ├── v0.0.0/
│   └── v2.0.0/
│   ...
├── models/                        # Model files and artifacts
├── partitioning/                  # Data partitioning information
├── proteome_fasta/                # Proteome FASTA files
│   ├── hg38.fasta                 # Human genome reference
│   └── uniprot_sprot.fasta        # UniProt Swiss-Prot database
└── random_reference_set/          # Random reference datasets
    └── 10k_9mers.csv              # 10k 9-mer peptides reference set
```

## Main Directories Descriptions

- **`data/`**: Main directory containing versioned datasets with standardized CSV format
- **`models/`**: Trained machine learning models for MHC binding prediction
- **`metrics/`**: Metrics files produced by the `evaluate` command line
- **`proteome_fasta/`**: Reference proteome sequences in FASTA format used for decoys generation
  - `hg38.fasta`: Human genome reference (GRCh38)
  - `uniprot_sprot.fasta`: UniProt Swiss-Prot protein database
- **`random_reference_set/`**:
  - `10k_9mers.csv`: 10k 9-mer randomly selected peptide reference sets
    for iterative annotation and calibration

## Data Versioning System

The `data/` directory uses a semantic versioning system (X.Y.Z) to track different versions of the datasets.
The `metrics/` directory uses the same versioning system and stores the evaluation files of the models
that were evaluated on the particular data version.

### Version Components

- **X (Hit Versioning)**: Increments when hits are modified (filtered, etc.)
- **Y (Decoy Versioning)**: Increments when decoys are changed (ratio, sampling, etc.)
- **Z (CSV Versioning)**: Increments when additional columns are added or removed

For detailed information about dataset versions and their descriptions, see the [Datasets documentation](../datasets/README.md).

### Metrics

The `metrics/` directory uses the same versioning system and stores the evaluation files of the models
that were evaluated on the particular data version.

The convention is the following: `{model_type}_{mhc_type}_{training_data_version}_{ensemble}__{cd8,ms_ligands}`

The bucket structure looks like this:

  ```bash
  metrics
  ├── v0.0.0
  │   ├── nn_align_mhc1_v_0_0_0_ensemble__cd8
  │   ├── nn_align_mhc1_v_0_0_0_ensemble__ms_ligands
  │   ├── nn_align_mhc1_v_1_1_0_ensemble__cd8
  │   └── nn_align_mhc1_v_1_1_0_ensemble__ms_ligands
  └── v2.0.0
      ├── nn_align_mhc1_v_0_0_0_ensemble__ms_ligands
      └── nn_align_mhc1_v_1_1_0_ensemble__ms_ligands
  ```
