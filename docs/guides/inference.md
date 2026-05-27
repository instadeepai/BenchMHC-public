# Inference

## Prediction

The `predict` command allows you to predict with a model or an ensemble of models on a dataset.
Predictions for the different provided models will be averaged and a new column with the provided
`--predictions_column_prefix` will be added in the dataset or in the resulting output dataset.

??? note "NetMHCPan4.1 9mers core"

    If you are running predictions with a NetMHCPan4.1 model, 3 additional columns are
    written per output type (`hit`, `binding affinity`). These columns track the
    9mers core selected by the model:

    For each output, you hence have:

    - `{output_name}_selected_core_indices`: A semicolon-separated string of the core indices predicted by all N models.
    - `{output_name}_majority_core_index`: An integer representing the most common core index (the result of the majority vote).
    - `{output_name}_majority_core`: The amino acid sequence corresponding to the `{output_name}_majority_core_index`.

More details on the command with `predict --help`.

!!! tip "Model paths"

    While model paths can be directly passed with multiple -m flags:
    `--model_path /path/to/model/1 --model_path /path/to/model/2` or comma-separated list:
    `--model_path '/path/to/model/1,/path/to/model/2'`, for ensemble models with many base models
    it can be useful to gather all model paths within a `.txt` file and use the latter as argument:
    `--model_path '/path/to/file/with/models.txt'`.

!!! example "Predict with [NetMHCpan-4.1](../models/netmhcpan41_v_0_0_0.md)"

    ```bash
    # Pull the BA + SA + MA model
    gcloud storage cp -r gs://bench-mhc/models/nnalign_mhc1_ba_sa_ma_bs_1024_sgd_5en2_ensemble.txt models/nnalign_mhc1_ba_sa_ma_bs_1024_sgd_5en2_ensemble.txt

    # Predict on your dataset
    predict -m models/nnalign_mhc1_ba_sa_ma_bs_1024_sgd_5en2_ensemble.txt -d /path/to/dataset.csv --batch_size 8192 --predictions_column_prefix my_predictions
    ```

## Evaluation

The `evaluate` command allows to compute metrics on predictions computed with `predict`. The
resulting metrics are saved in an output directory provided by `--output_directory`. Per allele and
global metrics are computed.

More details on the command with `evaluate --help`.

!!! example "Evaluate [NetMHCpan-4.1](../models/netmhcpan41_v_0_0_0.md) on [MS ligands](../models/netmhcpan41_v_0_0_0.md#ms-ligands)"

    ```bash
    # Pull the BA + SA + MA model
    gcloud storage cp -r gs://bench-mhc/models/nnalign_mhc1_ba_sa_ma_bs_1024_sgd_5en2_ensemble.txt models/nnalign_mhc1_ba_sa_ma_bs_1024_sgd_5en2_ensemble.txt

    # Pull MS ligands
    gcloud storage cp -r gs://bench-mhc/data/v0.0.0/evaluation/raw_data/ms_ligands.csv data/v0.0.0/evaluation/raw_data/ms_ligands.csv

    # Predict on MS ligands
    predict -m models/nnalign_mhc1_ba_sa_ma_bs_1024_sgd_5en2_ensemble.txt -d data/v0.0.0/evaluation/raw_data/ms_ligands.csv --batch_size 8192 --predictions_column_prefix nnalign_mhc1_ba_sa_ma_bs_1024_sgd_5en2_ensemble

    # Evaluate on MS ligands
    evaluate -d data/v0.0.0/evaluation/raw_data/ms_ligands.csv --prediction_identifier nnalign_mhc1_ba_sa_ma_bs_1024_sgd_5en2_ensemble__hit --target_identifier hit --output_directory metrics
    ```
