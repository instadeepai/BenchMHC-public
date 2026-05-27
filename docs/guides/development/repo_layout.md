# Repository layout

!!! info

    BenchMHC is organized so that the Python package, experiment configuration, and local data stay
    separate.

Here is how the repository is organized:

```text

├── bench_mhc               # Application code for the `bench-mhc` CLI and training and evaluation stack
│   ├── cli                 # Command-line entry points (train, predict, dataset utilities, etc.)
│   ├── custom_objects      # Layers, losses, metrics, and callbacks used by models
│   ├── dataset             # Dataset loading, caching, and related helpers used during training and evaluation
│   ├── model               # Network definitions and training logic for supported architectures
│   ├── utils               # Shared utilities (I/O, logging, click helpers, metrics, etc.)
│   └── variables           # Feature encodings and variable definitions
├── data                    # Local directory for datasets or outputs, not versioned
├── docs                    # MkDocs source: guides, model pages, API reference, and images under `docs/media/`
├── configuration           # YAML files for training runs, plotting, and templates (for example `configuration/nnalign_sa.yml`)
├── models                  # Trained machine learning models, not versioned (created at training time)
```

!!! tip "Passing the configuration file path"

    The configuration file path is typically passed to CLI commands with `-c` /
    `--configuration_file_path`.
