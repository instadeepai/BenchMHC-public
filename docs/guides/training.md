# Training

## Training on single-allelic (SA) data

In order to train models, you need to:

- prepare SA training and validation datasets in `.csv` files,
- set configuration in a `.yml` file (see a [dedicated documentation](./training_configuration.md)
  or `configuration/nnalign_ba_sa.yml` as an example),
- run the following command:

```bash
train -n my_model_name -c /path/to/configuration.yml -t /path/to/training/data.csv -v /path/to/validation/data.csv
```

More details about the command with `train --help`.

!!! example "Re-training one base model of NetMHCpan-4.1 on HLA / BA + SA on split 0"

    ```bash
    # Pull the HLA / BA + SA data
    mkdir -p data/v0.0.0/training/
    gcloud storage cp -r gs://bench-mhc/data/v0.0.0/training/ba_sa_hla data/v0.0.0/training/ba_sa_hla

    # Train the model
    train -n my_new_netmhcpan41_hla_ba_sa -c configuration/nnalign_ba_sa.yml -t data/v0.0.0/training/ba_sa_hla/split_0_train.csv -v data/v0.0.0/training/ba_sa_hla/split_0_tune.csv
    ```

!!! tip "Format the allele feature"

    To avoid any edge cases with unknown allele names during inference, especially if you want to
    map allele names to their pseudo-sequences representations, you can format the allele names with
    the command line `format-allele-feature`:

    ```bash
    format-allele-feature -d data/to/be/formatted.csv -o formatted/data.csv
    ```

    More details for the command with `format-allele-feature --help`.

## Training on single-allelic (SA) and multi-allelic (MA) data with iterative annotation

The iterative annotation process described in
[NetMHCpan-4.1](https://pubmed.ncbi.nlm.nih.gov/32406916/) is available in the `train` command. To
use it, you need to:

- prepare SA training and validation datasets in `.csv` files,
- prepare MA training and (optionally) validation datasets in `.csv` files,
- set configuration in a `.yml` file (see a [dedicated documentation](./training_configuration.md)
  or `configuration/nnalign_ba_sa.yml` as an example),
- set the number of warm-up epochs you want to train on SA data only (`--sa_warmup_epochs`), the
  total number of epochs (warm-up on SA + fine-tuning on SA + MA) is specified in the configuration
  in `training.epochs`,
- define the columns used to annotate the data (`--deconvolution_identifier` and
  `--deconvolution_output_name`)
- run the following command:

```bash
train -c /path/to/configuration.yml -t /path/to/training/sa/data.csv -v /path/to/validation/sa/data.csv -ma_t /path/to/training/ma/data.csv -ma_v /path/to/validation/ma/data.csv --deconvolution_identifier MA_bag_identifier --deconvolution_output_name hit --sa_warmup_epochs 20
```

More details about the command (notably the other hyper-parameters for iterative annotation) with
`train --help`.

!!! example "Re-training one base model of NetMHCpan-4.1 on HLA / BA + SA + MA on split 0"

    ```bash
    # Pull the HLA / BA + SA data
    mkdir -p data/v0.0.0/training/
    gcloud storage cp -r gs://bench-mhc/data/v0.0.0/training/ba_sa_hla data/v0.0.0/training/

    # Pull the HLA / MA data
    gcloud storage cp -r gs://bench-mhc/data/v0.0.0/training/ma_hla data/v0.0.0/training/

    # Pull the reference data for prediction score rescaling
    mkdir random_reference_set
    gcloud storage cp -r gs://bench-mhc/random_reference_set/10k_9mers.csv random_reference_set/10k_9mers.csv

    # Train the model
    train -n my_new_netmhcpan41_hla_ba_sa_ma -c configuration/nnalign_ba_sa.yml -t data/v0.0.0/training/ba_sa_hla/split_0_train.csv -v data/v0.0.0/training/ba_sa_hla/split_0_tune.csv -ma_t data/v0.0.0/training/ma_hla/split_0_train.csv --sa_warmup_epochs 20 --deconvolution_identifier MA_bag_identifier --deconvolution_output_name hit -r random_reference_set/10k_9mers.csv
    ```
