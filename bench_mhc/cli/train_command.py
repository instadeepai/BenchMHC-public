"""Module to define the entry point of the command line to train a model."""

from pathlib import Path

import rich_click as click

from bench_mhc.utils.click import arguments


@click.command(context_settings={"show_default": True})
@arguments.experiment_name
@arguments.configuration_file_path
@arguments.training_path
@arguments.validation_path
@arguments.random_seed
@arguments.ma_training_path
@arguments.ma_validation_path
@arguments.sa_warmup_epochs
@arguments.batch_size
@arguments.deconvolution_identifier
@arguments.deconvolution_output_name
@arguments.use_prediction_score_rescaling
@arguments.reference_path
@arguments.w_shift_parameter
@arguments.z_score_threshold
def train(
    experiment_name: str,
    configuration_file_path: Path,
    training_path: Path,
    validation_path: Path,
    random_seed: int | None,
    ma_training_path: Path | None,
    ma_validation_path: Path | None,
    sa_warmup_epochs: int,
    batch_size: int,
    deconvolution_identifier: str,
    deconvolution_output_name: str,
    use_prediction_score_rescaling: bool,
    reference_path: Path | None,
    w_shift_parameter: int,
    z_score_threshold: float,
) -> None:
    """Train a model on single-allelic (SA) data or SA and multi-allelic (MA) data.

    1. If `ma_training_path` not provided, classical training is performed on SA data.

    2. If `ma_training_path` is provided, training is performed with iterative annotation on
    both SA and MA data.

    The training process with iterative annotation/deconvolution is inspired by "NNAlign_MA",
    Alvarez et al., 2019, cf. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6885703/.

    The workflow is the following:

    1. the model is first trained on SA data, using the number of epochs specified in
    `--sa_warmup_epochs`,

    2. then, for each of the additional epochs specified in `training.epochs` configuration:

        1. the MA data is annotated (or deconvoluted) with the latest checkpoint,

        2. the annotated MA data is fused with the SA data,

        3. the model is trained for one additional epoch on the fused data.

    Remarks:

    - The provided MA data should be in a long format, i.e. the allele column is "exploded" such
    that each row is a peptide-allele combination. The column identified by
    `deconvolution_identifier` is used for deconvolution.

    - If `ma_validation_path` is not provided, only `sa_validation_path` is used as validation
    data for SA pre-training and SA + MA fine-tuning.

    - To level out differences in the prediction scores between MHC alleles imposed by the
    differences in number of positive training examples and distance to the training data included
    in the SA data set, the training MA data raw predictions may be rescaled, prior to the training
    MA data annotation. This step is roughly equivalent to a per-allele calibration. This is done
    if `--use_prediction_score_rescaling` is provided.

    - A positive sample from the training MA data or a sample from the validation MA data is
    deconvoluted by taking the allele that yields the highest rescaled prediction among its bag of
    alleles. A negative sample from the training MA data is deconvoluted by randomly taking an
    allele out of its bag of alleles.

    GPU configuration can be specified as follows under the `training/gpus` key:

    \b
    - gpus: `auto` - Uses all available GPUs
    - gpus: `2` - Uses 2 GPUs
    - gpus: `[0, 1]` - Uses specific GPU indices
    - gpus: `null` - Falls back to CPU training
    """  # noqa: D301
    from bench_mhc.cli.train import train

    train(
        experiment_name=experiment_name,
        configuration_file_path=configuration_file_path,
        training_path=training_path,
        validation_path=validation_path,
        random_seed=random_seed,
        ma_training_path=ma_training_path,
        ma_validation_path=ma_validation_path,
        sa_warmup_epochs=sa_warmup_epochs,
        batch_size=batch_size,
        deconvolution_identifier=deconvolution_identifier,
        deconvolution_output_name=deconvolution_output_name,
        use_prediction_score_rescaling=use_prediction_score_rescaling,
        reference_path=reference_path,
        w_shift_parameter=w_shift_parameter,
        z_score_threshold=z_score_threshold,
    )
