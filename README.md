# BenchMHC

<!--readme-wo-license-start-->

> [!IMPORTANT]
> **Partial public release.** This repository is a partial open-source release of
> BenchMHC. Some reproductions and the full development history are **not yet included**.
> Similarly, links to specific Github issues won't be reachable.
> A complete release with the full history and all previous releases will be available
> soon at this same location.

---

[![Lint](https://github.com/instadeepai/BenchMHC-public/actions/workflows/run-linters.yaml/badge.svg)](https://github.com/instadeepai/BenchMHC-public/actions/workflows/run-linters.yaml?query=branch%3Amain)
[![Test](https://github.com/instadeepai/BenchMHC-public/actions/workflows/run-tests.yaml/badge.svg)](https://github.com/instadeepai/BenchMHC-public/actions/workflows/run-tests.yaml?query=branch%3Amain)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/theomeb/5dd7644b1c2ba953ead3da6973d5579c/raw/pytest-coverage_bench-mhc-public_main.json)

![Release](https://img.shields.io/badge/Version-v1.3.0--alpha-green.svg?logo=python&logoColor=white)

[![python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org)
[![torch](https://img.shields.io/badge/torch-2.5.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![lightning](https://img.shields.io/badge/Lightning-2.4.0-792ee5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![notebook](https://img.shields.io/badge/notebook-7.2.2-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)

[![ruff](https://img.shields.io/badge/ruff-0.7.0-3A3A3A?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/mypy-1.12.0-2C5BB4)](http://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-4.0.1-000000?logo=pre-commit&logoColor=white)](https://pre-commit.com)

---

## Introduction

This work aims to reproduce the architectures and training pipelines (referred to as "methods")
of publicly available baseline models for **peptide-MHC presentation prediction**.

> [!NOTE]
> Personalized cancer immunotherapies have emerged as a promising approach for cancer treatment by
> leveraging the ability of the immune system to recognize and eliminate cancer cells. Predicting
> which specific protein fragments (peptides) will be displayed on cell surfaces by MHC molecules
> (forming pMHC complexes) is a key challenge and a necessary first step in determining whether the
> immune system will effectively fight cancer.

<!--models-list-start-->

| Model Name                                       | MHC Type       | Reproduced                |
|--------------------------------------------------|----------------|---------------------------|
| [NetMHCpan-4.1](docs/models/netmhcpan41_v_0_0_0.md)[^1]  | MHC1           | :white_check_mark:               |
| NetMHCIIpan-4.3[^2]                                      | MHC2           | :x:                              |
| Pep2Vec[^3]                                              | MHC1 / MHC2    | :hourglass_flowing_sand: planned |
| NetMHCpanExp[^4]                                         | MHC1           | :x:                              |

<!--models-list-end-->

> [!IMPORTANT]
> This list is not exhaustive.
> Our goal is to facilitate within this codebase the reproduction of any such released training
> pipelines, and associated model checkpoints on public data.

Details for each reproduced model can be found in the [Models](./docs/models/README.md) section of
the documentation.

Based on insights gained from these reproductions, we plan to develop improved versions of these
models with enhanced performance. These improved models will have dedicated pages in the
[Models](./docs/models/README.md) section.

Our ultimate goal is to establish a benchmark for comparing these different models on consistent
data. Currently, published methods often come as tools trained on diverse datasets. Few studies have
directly compared these predictors when exclusively trained on the same data, making it challenging
to determine the most effective approach for pMHC presentation prediction. Therefore, we also intend
to re-train, evaluate, and benchmark the reproduced methods using unified training and evaluation
data. We believe this comparative analysis will help distinguish the impact of architectural choices
from the benefits of additional training data, ultimately leading to improved machine learning
predictors for pMHC presentation.

## Links

- :books: [Documentation](https://instadeepai.github.io/BenchMHC-public/)
- [Glossary](docs/guides/glossary.md): short definitions (SA/MA, BA/EL, pMHC, FRANK, decoys, etc.)

> [!TIP]
> Start with the [Guides overview](docs/guides/README.md) to get an overview of the project.

## Developed by

- <img src="./docs/media/instadeep_logo.png" alt="InstaDeep" width="15" /> [InstaDeep](https://www.instadeep.com/)

## Usage

### Installation

Follow the [_Installation_ guide](docs/guides/installation.md) to install the `bench-mhc` package.

### Command line

After installation, you can list all available commands (within a Docker container or within your
activated virtual environment) by running:

```bash
bench-mhc --help
```

![`bench-mhc --help`](./docs/media/bench_mhc_help.png)

> [!NOTE]
> For the full documentation of a specific command you can run:
>
> ```bash
> {command} --help
> ```

You will find details and examples on:

- how to `train` a model in the [Training](docs/guides/training.md) guide,
- how to `predict` and `evaluate` a model in the [Inference](docs/guides/inference.md) guide.

## Contributing

Contributing guidelines are listed in [`CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## BibTeX entry and citation info

If you use `BenchMHC` in your research, please cite the following paper:

```bash
@article{BenchMHC,
    title        = {BenchMHC: Towards a universal standard for reproducible peptide-MHC presentation prediction in neoantigen discovery},
    author       = {Borck, Théomé and Chabrol, Marion and Colombo, Daniel and Imbert, Arthur and Volzhenin, Konstantin and Perier-Dulhoste, Martin},
    year         = 2026,
    month        = {Mar},
    day          = 31,
    url          = {https://github.com/instadeepai/BenchMHC-public}
}
```

<!--readme-wo-license-end-->

## License

### Model checkpoints [![CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY_NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

The model checkpoints are licensed under **Creative Commons Non-Commercial
([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))**, see details in
[LICENSE.md](./docs/LICENSE.md#model-checkpoints).

### Code [![Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Code is licensed under the **Apache License, Version 2.0** (see [LICENSE.md](./docs/LICENSE.md#code)).

<!--footnotes-start-->

[^1]: [NetMHCpan-4.1](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/)
[^2]: [NetMHCIIpan-4.3](https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/)
[^3]: [Pep2Vec](https://github.com/Genentech/Pep2Vec)
[^4]: [NetMHCpanExp-1.0](https://services.healthtech.dtu.dk/services/NetMHCpanExp-1.0/)

<!--footnotes-end-->
