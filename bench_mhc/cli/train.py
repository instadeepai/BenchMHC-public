"""Module to define the command line to train a model."""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelSummary

from bench_mhc.constants import CACHE_DIRECTORY
from bench_mhc.constants import MODEL_DIRECTORY
from bench_mhc.dataset.main import DataModule
from bench_mhc.model import MODEL_NAME2LIGHTNING_MODULE
from bench_mhc.model import ModelType
from bench_mhc.utils.callbacks import get_early_stopping_callback
from bench_mhc.utils.callbacks import load_callbacks_from_config
from bench_mhc.utils.callbacks import maybe_add_model_checkpoint_callback
from bench_mhc.utils.callbacks import override_callbacks
from bench_mhc.utils.deconvolution import apply_deconvolution
from bench_mhc.utils.device import get_devices_and_accelerator
from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.io import load_yml
from bench_mhc.utils.io import save_json
from bench_mhc.utils.logging import system
from bench_mhc.utils.logging.experiment import get_logger
from bench_mhc.utils.mode import Mode
from bench_mhc.utils.model import get_directory_name
from bench_mhc.utils.predictions import process_outputs
from bench_mhc.utils.random import set_random_seed
from bench_mhc.variables.variables import Outputs
from bench_mhc.variables.variables import Variables

log = system.get(__name__)


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
    """Train a model.

    Refer to bench_mhc.cli.train_command.py::train for the complete documentation.

    Args:
        experiment_name: The name of the experiment.
        configuration_file_path: The configuration file to use.
        training_path: The path to the training dataset.
        validation_path: The path to the validation dataset.
        random_seed: An optional random seed to use.
        ma_training_path: The optional path to MA training dataset.
            Only relevant for iterative annotation.
        ma_validation_path: The optional path to MA validation dataset.
            Only relevant for iterative annotation.
        sa_warmup_epochs: The number of SA warmup epochs to use.
            Only relevant for iterative annotation.
        batch_size: The batch size to use for inference on MA data or reference data.
            Only relevant for iterative annotation.
        deconvolution_identifier: The deconvolution identifier.
            Only relevant for iterative annotation.
        deconvolution_output_name: Name of the output to consider for deconvolution.
            Only relevant for iterative annotation.
        use_prediction_score_rescaling: Whether to use prediction score rescaling.
            Only relevant for iterative annotation.
        reference_path: The path to the reference dataset for prediction score rescaling.
            Only relevant for iterative annotation.
        w_shift_parameter: Shift value in the w calculation for the prediction score rescaling step.
            Only relevant for iterative annotation.
        z_score_threshold: Threshold to use to filter out z-score outliers for the prediction score
            rescaling step. Only relevant for iterative annotation.

    Raises:
        ValueError: If model.class_name in the configuration is invalid.
    """
    random_seed = set_random_seed(random_seed)

    model_path = MODEL_DIRECTORY / get_directory_name(experiment_name=experiment_name)
    configuration = load_yml(configuration_file_path)

    gpus = configuration["training"].get("gpus", "auto")
    devices, accelerator = get_devices_and_accelerator(gpus)

    maybe_add_model_checkpoint_callback(configuration, model_path)

    model_name = configuration["model"]["class_name"]
    try:
        lightning_module_class: type[ModelType] = MODEL_NAME2LIGHTNING_MODULE[model_name]
    except KeyError as error:
        raise ValueError(
            f"The provided '{model_name}' is not a valid model class name. "
            f"Valid model names: {format_iterable(MODEL_NAME2LIGHTNING_MODULE.keys())}."
        ) from error

    model_name = model_path.name
    log.info(f"Loading the data module for model '{model_name}'.")
    data_module = DataModule(
        inputs=Variables.from_dict(configuration["variables"]["inputs"]),
        outputs=Outputs.from_dict(configuration["variables"]["outputs"]),
        batch_size=configuration["training"]["batch_size"],
        num_workers=configuration["training"]["num_workers"],
        prefetch_factor=configuration["training"]["prefetch_factor"],
        train_file_path=training_path,
        val_file_path=validation_path,
        cache_directory=CACHE_DIRECTORY,
    )

    log.info(
        f"Fitting variables on the training dataset '{training_path}' to create vocabulary "
        "and maximal length of the 'allele' variable."
    )
    data_module.fit_variables()

    model: ModelType = lightning_module_class(
        model_path=model_path,
        configuration=configuration,
        inputs=data_module.inputs,
        outputs=data_module.outputs,
    )

    logger = get_logger(
        model=model,
        training_path=training_path,
        validation_path=validation_path,
        ma_training_path=ma_training_path,
        ma_validation_path=ma_validation_path,
    )
    callbacks = load_callbacks_from_config(configuration["training"]["callbacks"])

    if ma_training_path is not None:
        log.info(
            "Training will be performed with iterative annotation since 'ma_training_path' is "
            f"provided and set to {ma_training_path}."
        )

        _check_arguments_for_iterative_annotation(
            ma_training_path=ma_training_path,
            ma_validation_path=ma_validation_path,
            model=model,
            sa_warmup_epochs=sa_warmup_epochs,
            deconvolution_identifier=deconvolution_identifier,
            deconvolution_output_name=deconvolution_output_name,
            use_prediction_score_rescaling=use_prediction_score_rescaling,
            reference_path=reference_path,
        )

    epochs = configuration["training"]["epochs"]
    # Omit top-level "epochs": it duplicates configuration/training/epochs in MLflow. Epoch count
    # stays in configuration (and hparams.json) under configuration["training"]["epochs"].
    extra_hparams = {
        "random_seed": random_seed,
        "training_path": training_path,
        "validation_path": validation_path,
    }
    model.hparams.update(extra_hparams)
    # Also update _hparams_initial so Lightning logs these params to MLflow on the first
    # trainer.fit(). Without this, only _hparams is updated and these keys would leak into
    # MLflow only in iterative annotation (where on_train_end refreshes _hparams_initial
    # and a subsequent trainer.fit() re-sends them).
    model._hparams_initial.update(extra_hparams)
    trainer = Trainer(
        default_root_dir=model_path,
        deterministic=random_seed is not None,
        max_epochs=sa_warmup_epochs if ma_training_path is not None else epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        devices=devices,
        accelerator=accelerator,
    )

    log.info(f"Starting the training of model '{model_name}'.")
    if ma_training_path is not None:
        log.info("Iterative annotation: pre-training on SA data...")

    trainer.fit(model, datamodule=data_module)

    if ma_training_path is not None:
        transformed_data_directory = CACHE_DIRECTORY / model_name / "transformed_data"
        _train_w_iterative_annotation(
            transformed_data_directory=transformed_data_directory,
            ma_training_path=ma_training_path,
            ma_validation_path=ma_validation_path,
            sa_warmup_epochs=sa_warmup_epochs,
            epochs=epochs,
            batch_size=batch_size,
            deconvolution_identifier=deconvolution_identifier,
            deconvolution_output_name=deconvolution_output_name,
            use_prediction_score_rescaling=use_prediction_score_rescaling,
            reference_path=reference_path,
            w_shift_parameter=w_shift_parameter,
            z_score_threshold=z_score_threshold,
            model=model,
            trainer=trainer,
            data_module=data_module,
        )

    log.info(f"Model '{model_name}' trained.")


def _check_arguments_for_iterative_annotation(
    ma_training_path: Path,
    ma_validation_path: Path | None,
    model: ModelType,
    sa_warmup_epochs: int,
    deconvolution_identifier: str,
    deconvolution_output_name: str,
    use_prediction_score_rescaling: bool,
    reference_path: Path | None,
) -> None:
    """Check the arguments are properly provided for iterative annotation.

    Args:
        ma_training_path: Path to the MA training data.
        ma_validation_path: Path to the MA validation data.
        model: Model to train with iterative annotation.
        sa_warmup_epochs: Number of epochs for SA warmup training.
        deconvolution_identifier: The deconvolution identifier.
        deconvolution_output_name: Name of the deconvolution output name.
        use_prediction_score_rescaling: Whether to use prediction score rescaling.
        reference_path: Optional path to the reference dataset.

    Raises:
        ValueError:
            - If 'sa_warmup_epochs' is greater than the number of epochs provided in the
              configuration.
            - If 'deconvolution_output_name' is not available in model outputs.
            - If 'deconvolution_identifier' is not available in MA training or validation dataset.
            - If 'reference_path' is not provided when prediction score rescaling is used.
    """
    if model.configuration["training"]["epochs"] <= sa_warmup_epochs:
        raise ValueError(
            f"Iterative annotation: 'training.epochs={model.configuration['training']['epochs']}' "
            "but it must be greater than the number of epochs for SA warmup training: "
            f"{sa_warmup_epochs}."
        )

    if deconvolution_output_name not in model.outputs.names:
        raise ValueError(
            f"Iterative annotation: 'deconvolution_output_name={deconvolution_output_name}' is not "
            f"available in model outputs: {format_iterable(model.outputs.names)}."
        )

    if deconvolution_identifier not in pl.scan_csv(ma_training_path).collect_schema().names():
        raise ValueError(
            f"Iterative annotation: 'deconvolution_identifier={deconvolution_identifier}' is not "
            f"available in provided MA training dataset: {ma_training_path}."
        )

    if (
        ma_validation_path is not None
        and deconvolution_identifier not in pl.scan_csv(ma_validation_path).collect_schema().names()
    ):
        raise ValueError(
            f"Iterative annotation: 'deconvolution_identifier={deconvolution_identifier}' is not "
            f"available in provided MA validation dataset: {ma_validation_path}."
        )

    if use_prediction_score_rescaling and reference_path is None:
        raise ValueError(
            "Iterative annotation: You must provide 'reference_path' to use prediction score "
            "rescaling."
        )


def _train_w_iterative_annotation(
    transformed_data_directory: Path,
    ma_training_path: Path,
    ma_validation_path: Path | None,
    sa_warmup_epochs: int,
    epochs: int,
    batch_size: int,
    deconvolution_identifier: str,
    deconvolution_output_name: str,
    use_prediction_score_rescaling: bool,
    reference_path: Path | None,
    w_shift_parameter: int,
    z_score_threshold: float,
    model: ModelType,
    trainer: Trainer,
    data_module: DataModule,
) -> None:
    """Fine-tune on the remaining epochs with iterative annotation.

    Args:
        transformed_data_directory: The path to the transformed data.
        ma_training_path: The path to the MA training dataset.
        ma_validation_path: The optional path to the MA validation dataset.
        sa_warmup_epochs: The number of epochs used for SA warmup training.
        epochs: The total number of epochs to use for training.
        batch_size: The batch size to use for inference on MA data or reference data.
        deconvolution_identifier: The deconvolution identifier.
        deconvolution_output_name: Name of the output to consider for deconvolution.
        use_prediction_score_rescaling: Whether to use prediction score rescaling.
        reference_path: The path to the reference dataset for prediction score rescaling.
        w_shift_parameter: Shift value in the w calculation for the prediction score rescaling step.
        z_score_threshold: Threshold to use to filter out z-score outliers for the prediction score
            rescaling step.
        model: The model to fine-tune on SA + MA data.
        trainer: The trainer to use for training.
        data_module: The data module used in SA warmup training.
    """
    log.info(
        f"Iterative annotation: pre-training on SA data for {sa_warmup_epochs} epochs done, "
        f"starting fine-tuning on SA+MA data for {epochs - sa_warmup_epochs} epochs..."
    )
    iterative_annotation_metadata = {
        "iterative_annotation": {
            "epochs": epochs - sa_warmup_epochs,
            "deconvolution_identifier": deconvolution_identifier,
            "deconvolution_output_name": deconvolution_output_name,
            "ma_training_path": ma_training_path,
            "ma_validation_path": ma_validation_path,
        }
    }
    ma_training_positive_df, ma_training_negative_df, ma_training_positive_data_module = (
        _prepare_ma_data(
            transformed_data_directory=transformed_data_directory,
            ma_data_path=ma_training_path,
            variables=data_module.inputs + data_module.outputs,
            deconvolution_identifier=deconvolution_identifier,
            deconvolution_output_name=deconvolution_output_name,
            data_module=data_module,
            mode=Mode.TRAIN,
            batch_size=batch_size,
        )
    )

    ma_validation_positive_df = None
    ma_validation_negative_df = None
    ma_validation_positive_data_module = None
    if ma_validation_path is None:
        log.warning(
            "'ma_validation_path' is not provided, only SA data from 'validation_path' is "
            "used as validation data."
        )
    else:
        (
            ma_validation_positive_df,
            ma_validation_negative_df,
            ma_validation_positive_data_module,
        ) = _prepare_ma_data(
            transformed_data_directory=transformed_data_directory,
            ma_data_path=ma_validation_path,
            variables=data_module.inputs + data_module.outputs,
            deconvolution_identifier=deconvolution_identifier,
            deconvolution_output_name=deconvolution_output_name,
            data_module=data_module,
            mode=Mode.VAL,
            batch_size=batch_size,
        )

    sa_ma_data_module = DataModule(
        transformed_data_directory=transformed_data_directory / "iterative_annotation" / "sa_ma",
        inputs=data_module.inputs,
        outputs=data_module.outputs,
        batch_size=data_module.batch_size,
        num_workers=data_module.num_workers,
        prefetch_factor=data_module.prefetch_factor,
    )

    alleles = set(
        pl.scan_csv(ma_training_path).select(pl.col("allele").unique()).collect()["allele"]
    )
    reference_data_module, allele2value = _maybe_get_reference_data_module(
        transformed_data_directory=transformed_data_directory,
        use_prediction_score_rescaling=use_prediction_score_rescaling,
        reference_path=reference_path,
        data_module=data_module,
        alleles=alleles,
        batch_size=batch_size,
    )
    if use_prediction_score_rescaling and reference_data_module is not None:
        iterative_annotation_metadata["iterative_annotation"]["prediction_score_rescaling"] = {
            "reference_path": reference_path,
            "w_shift_parameter": w_shift_parameter,
            "z_score_threshold": z_score_threshold,
        }

    model.hparams.update(iterative_annotation_metadata)

    # Reset callbacks and prevent printing model summary on each fit loop
    # mypy complains about no Trainer.callbacks attributes, but this attribute is set with
    # Trainer._callback_connector._attach_model_callbacks()
    trainer.callbacks = override_callbacks(  # type: ignore
        callbacks=trainer.callbacks,  # type: ignore
        new_callbacks=load_callbacks_from_config(model.configuration["training"]["callbacks"]),
        callback_classes_to_exclude=(ModelSummary,),
    )

    for epoch in range(sa_warmup_epochs, epochs):
        should_stop_iterative_annotation = _train_one_iterative_annotation_epoch(
            model=model,
            trainer=trainer,
            data_module=data_module,
            sa_ma_data_module=sa_ma_data_module,
            ma_training_positive_df=ma_training_positive_df,
            ma_training_negative_df=ma_training_negative_df,
            ma_training_positive_data_module=ma_training_positive_data_module,
            ma_validation_positive_df=ma_validation_positive_df,
            ma_validation_negative_df=ma_validation_negative_df,
            ma_validation_positive_data_module=ma_validation_positive_data_module,
            deconvolution_identifier=deconvolution_identifier,
            deconvolution_output_name=deconvolution_output_name,
            reference_data_module=reference_data_module,
            allele2value=allele2value,
            w_shift_parameter=w_shift_parameter,
            z_score_threshold=z_score_threshold,
            epoch=epoch,
            epochs=epochs,
        )

        if should_stop_iterative_annotation:
            break


def _prepare_ma_data(
    transformed_data_directory: Path,
    ma_data_path: Path,
    variables: Variables,
    deconvolution_identifier: str,
    deconvolution_output_name: str,
    data_module: DataModule,
    mode: Mode,
    batch_size: int,
) -> tuple[pl.DataFrame, pl.DataFrame, DataModule]:
    """Prepare the multi-allelic (MA) data.

    It does the following:
    1. Transform the MA data with variables.
    2. Split the positive and negative MA samples.
    3. Save the transformed positive and negative MA samples in .parquet format.
    4. Return the DataFrames for positive and negative samples, as well as the DataModule for
       MA positive data.

    The transformed MA data is hence transformed once and for all and loaded in memory.
    The DataModule for MA positive data will be used to compute predictions on MA positive data.

    Args:
        transformed_data_directory: The path to the transformed data.
        ma_data_path: Path to the MA dataset.
        variables: Inputs and outputs.
        deconvolution_identifier: Deconvolution identifier.
        deconvolution_output_name: Output name to use for deconvolution.
        data_module: Data module of the SA data.
        mode: Mode of the MA data.
        batch_size: Batch size to use for inference on MA data.

    Returns:
        A tuple with MA positive DataFrame, MA negative DataFrame and DataModule for MA positive
        data.
    """
    log.info(f"Transforming MA data for mode '{mode}'...")
    ma_lf = variables.transform(pl.scan_csv(ma_data_path)).select(
        variables.features.union({deconvolution_identifier})
    )

    deconvolution_output_column = variables[deconvolution_output_name].column
    ma_positive_lf = ma_lf.filter(pl.col(deconvolution_output_column) == 1)
    ma_negative_lf = ma_lf.filter(pl.col(deconvolution_output_column) != 1)

    ma_transformed_data_directory = (
        transformed_data_directory / "iterative_annotation" / "ma" / mode
    )

    log.info(f"'batch_size={batch_size}' will be used to iterate over MA positive data.")
    ma_positive_data_module = DataModule(
        transformed_data_directory=ma_transformed_data_directory / "positive",
        inputs=data_module.inputs,
        outputs=data_module.outputs,
        batch_size=batch_size,
        num_workers=data_module.num_workers,
        prefetch_factor=data_module.prefetch_factor,
    )
    ma_positive_data_module.save_transformed_data(
        df=ma_positive_lf,
        mode=Mode.PREDICT,
    )
    log.info(f"MA positive data transformed and saved for mode '{mode}'...")

    ma_negative_data_module = DataModule(
        transformed_data_directory=ma_transformed_data_directory / "negative",
        inputs=data_module.inputs,
        outputs=data_module.outputs,
        batch_size=batch_size,
        num_workers=data_module.num_workers,
        prefetch_factor=data_module.prefetch_factor,
    )
    ma_negative_data_module.save_transformed_data(
        df=ma_negative_lf,
        mode=Mode.PREDICT,
    )
    log.info(f"MA negative data transformed and saved for mode '{mode}'...")

    ma_positive_df = pl.read_parquet(ma_transformed_data_directory / "positive")
    ma_negative_df = pl.read_parquet(ma_transformed_data_directory / "negative")
    _check_labels_are_consistent_w_deconvolution_identifier(
        ma_df=ma_positive_df,
        deconvolution_identifier=deconvolution_identifier,
        deconvolution_output_name=deconvolution_output_name,
    )
    _check_labels_are_consistent_w_deconvolution_identifier(
        ma_df=ma_negative_df,
        deconvolution_identifier=deconvolution_identifier,
        deconvolution_output_name=deconvolution_output_name,
    )
    log.info(f"MA positive and negative transformed data loaded in memory for mode '{mode}'.")

    return ma_positive_df, ma_negative_df, ma_positive_data_module


def _check_labels_are_consistent_w_deconvolution_identifier(
    ma_df: pl.DataFrame,
    deconvolution_identifier: str,
    deconvolution_output_name: str,
) -> None:
    """Check that each deconvolution identifier has only one label.

    Args:
        ma_df: DataFrame for MA data.
        deconvolution_identifier: Deconvolution identifier.
        deconvolution_output_name: Name of the output to consider for deconvolution.

    Raises:
        ValueError: If the MA data is not properly formatted.
    """
    identifier2num_unique_labels = {
        row[deconvolution_identifier]: row[deconvolution_output_name]
        for row in ma_df.group_by(deconvolution_identifier)
        .agg(pl.col(deconvolution_output_name).n_unique())
        .iter_rows(named=True)
    }

    wrong_deconvolution_identifiers = [
        deconvolution_identifier
        for deconvolution_identifier, num_unique_labels in identifier2num_unique_labels.items()
        if num_unique_labels > 1
    ]

    if len(wrong_deconvolution_identifiers) > 0:
        raise ValueError(
            "The MA data is not properly formatted as the following values of the deconvolution "
            f"identifier '{deconvolution_identifier}' have more than one label: "
            f"{format_iterable(wrong_deconvolution_identifiers)}."
        )


def _maybe_get_reference_data_module(
    transformed_data_directory: Path,
    use_prediction_score_rescaling: bool,
    reference_path: Path | None,
    data_module: DataModule,
    alleles: set[str],
    batch_size: int,
) -> tuple[DataModule | None, dict[str, np.ndarray]]:
    """Maybe get the DataModule for reference data.

    The reference data is defined if 'use_prediction_score_rescaling' is True and 'reference_path'
    is not None.

    Args:
        transformed_data_directory: The path to the transformed data.
        use_prediction_score_rescaling: Whether to use prediction score rescaling.
        reference_path: Path to the reference dataset.
        data_module: Data module of the SA data.
        alleles: Set of alleles for which to compute rescaling parameters.
        batch_size: Batch size to use for inference on reference data.

    Returns:
        A tuple with the optional DataModule (if prediction score rescaling is used) and the mapping
        from alleles to their transformed value.
    """
    if not (use_prediction_score_rescaling and reference_path is not None):
        return None, {}

    allele2value = {
        allele: data_module.inputs["allele"]
        .transform(pl.DataFrame({"allele": [allele]}))["allele"]
        .to_numpy()
        for allele in alleles
    }
    with tempfile.NamedTemporaryFile(suffix=".csv") as file:
        reference_w_allele_path = Path(file.name)
        # Save reference data with a placeholder (random allele value) for the allele column
        reference_data = pl.scan_csv(reference_path).with_columns(
            pl.lit(sorted(alleles)[0]).alias("allele")
        )
        reference_data.collect().write_csv(reference_w_allele_path)
        log.info(f"'batch_size={batch_size}' will be used to iterate over reference data.")
        reference_data_module = DataModule(
            transformed_data_directory=transformed_data_directory
            / "iterative_annotation"
            / "reference",
            inputs=data_module.inputs,
            outputs=data_module.outputs,
            batch_size=batch_size,
            num_workers=data_module.num_workers,
            prefetch_factor=data_module.prefetch_factor,
            predict_file_path=reference_w_allele_path,
        )
        reference_data_module.prepare_data()

    return reference_data_module, allele2value


def _train_one_iterative_annotation_epoch(
    model: ModelType,
    trainer: Trainer,
    data_module: DataModule,
    sa_ma_data_module: DataModule,
    ma_training_positive_df: pl.DataFrame,
    ma_training_negative_df: pl.DataFrame,
    ma_training_positive_data_module: DataModule,
    ma_validation_positive_df: pl.DataFrame | None,
    ma_validation_negative_df: pl.DataFrame | None,
    ma_validation_positive_data_module: DataModule | None,
    deconvolution_identifier: str,
    deconvolution_output_name: str,
    reference_data_module: DataModule | None,
    allele2value: dict[str, np.ndarray],
    w_shift_parameter: int,
    z_score_threshold: float,
    epoch: int,
    epochs: int,
) -> bool:
    """Train one epoch of iterative annotation.

    Args:
        model: The model to fine-tune on SA + MA data.
        trainer: The trainer to use for training.
        data_module: The data module used in SA warmup training.
        sa_ma_data_module: The data module to use for SA + MA fine-tuning.
        ma_training_positive_df: DataFrame for MA training positive data.
        ma_training_negative_df: DataFrame for MA training negative data.
        ma_training_positive_data_module: DataModule for the MA training positive dataset.
        ma_validation_positive_df: DataFrame for MA validation positive data.
        ma_validation_negative_df: DataFrame for MA validation negative data.
        ma_validation_positive_data_module: DataModule for the MA validation positive dataset.
        deconvolution_identifier: The deconvolution identifier.
        deconvolution_output_name: Name of the output to consider for deconvolution.
        reference_data_module: The data module for reference data.
        allele2value: Mapping from allele to its transformed value.
        w_shift_parameter: Shift value in the w calculation for the prediction score rescaling step.
        z_score_threshold: Threshold to use to filter out z-score outliers for the prediction score
            rescaling step.
        epoch: The current epoch.
        epochs: The total number of epochs to use for training.

    Returns:
        Whether iterative annotation should be stopped because of early stopping.
    """
    early_stopping_callback = get_early_stopping_callback(trainer.callbacks)  # type: ignore
    if early_stopping_callback is not None and early_stopping_callback.stopped_epoch > 0:
        log.info(
            "Early stopping has been reached at epoch "
            f"{early_stopping_callback.stopped_epoch} since "
            f"'{early_stopping_callback.monitor}' did not improve "
            f"(mode: {early_stopping_callback.mode}) from its best value: "
            f"{early_stopping_callback.best_score:.4}. "
            "The iterative annotation is stopped."
        )
        return True

    allele2rescaling_params = {}
    if reference_data_module is not None:
        log.info(
            "Computing the per-allele parameters to rescale raw predictions from "
            "positive MA training data..."
        )
        allele2rescaling_params = _compute_rescaling_params(
            model=model,
            trainer=trainer,
            reference_data_module=reference_data_module,
            allele2value=allele2value,
            deconvolution_output_name=deconvolution_output_name,
            w_shift_parameter=w_shift_parameter,
            z_score_threshold=z_score_threshold,
            epoch=epoch,
        )

    log.info("Annotating training MA data with latest checkpoint...")
    annotated_ma_training_df = _annotate_ma_data(
        model=model,
        trainer=trainer,
        ma_positive_df=ma_training_positive_df,
        ma_negative_df=ma_training_negative_df,
        ma_positive_data_module=ma_training_positive_data_module,
        deconvolution_identifier=deconvolution_identifier,
        deconvolution_output_name=deconvolution_output_name,
        allele2value=allele2value,
        allele2rescaling_params=allele2rescaling_params,
    )

    log.info("Merging training SA data with annotated training MA data...")
    features = sorted(data_module.inputs.features | data_module.outputs.features)
    sa_ma_training_df = pl.concat(
        [
            data_module.mode2dataset[Mode.TRAIN].df.select(features),
            annotated_ma_training_df.select(features),
        ],
    )
    sa_ma_data_module.save_transformed_data(
        df=sa_ma_training_df,
        mode=Mode.TRAIN,
    )

    sa_ma_validation_df = data_module.mode2dataset[Mode.VAL].df
    if (
        ma_validation_positive_df is not None
        and ma_validation_negative_df is not None
        and ma_validation_positive_data_module is not None
    ):
        log.info("Annotating validation MA data with latest checkpoint...")
        annotated_ma_validation_df = _annotate_ma_data(
            model=model,
            trainer=trainer,
            ma_positive_df=ma_validation_positive_df,
            ma_negative_df=ma_validation_negative_df,
            ma_positive_data_module=ma_validation_positive_data_module,
            deconvolution_identifier=deconvolution_identifier,
            deconvolution_output_name=deconvolution_output_name,
        )
        log.info("Merging validation SA data with annotated validation MA data...")
        sa_ma_validation_df = pl.concat(
            [
                sa_ma_validation_df.select(features),
                annotated_ma_validation_df.select(features),
            ]
        )

    sa_ma_data_module.save_transformed_data(
        df=sa_ma_validation_df,
        mode=Mode.VAL,
    )

    log.info(
        f"Iterative annotation: fine-tuning on SA+MA data for epoch " f"{epoch}/{epochs - 1}..."
    )
    trainer.fit_loop.max_epochs = epoch + 1
    trainer.fit(
        model=model,
        datamodule=sa_ma_data_module,
    )

    log.info(
        "Iterative annotation: fine-tuning on SA+MA data for epoch " f"{epoch}/{epochs - 1} done."
    )

    return False


def _compute_rescaling_params(
    model: ModelType,
    trainer: Trainer,
    reference_data_module: DataModule,
    allele2value: dict[str, np.ndarray],
    deconvolution_output_name: str,
    w_shift_parameter: int,
    z_score_threshold: float,
    epoch: int,
) -> dict[str, dict[str, float]]:
    """Compute and save the parameters to use for prediction score rescaling of the MA data.

    For each allele, the predictions of the model on the reference dataset are used to get
    z-scores' statistics.

    The process is detailed in the "Prediction Score Rescaling" section of "NNAlign_MA",
    Alvarez et al., 2019, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6885703/.
    Notations (e.g. p_bar) are the same as in the paper.

    Args:
        model: LightningModule for the model.
        trainer: Trainer for the model.
        reference_data_module: DataModule for the reference dataset.
        allele2value: Mapping from allele to its transformed value.
        deconvolution_output_name: Name of the output to consider for deconvolution.
        w_shift_parameter: Shift value in the w calculation for the prediction score rescaling step.
        z_score_threshold: Threshold value to use to filter out z-score outliers for the prediction
            score rescaling step.
        epoch: Current epoch.

    Returns:
        A mapping from an allele to its respective parameters 'p_bar' and 'sigma' to use for
            prediction score rescaling.
    """
    allele2stats: dict[str, dict[str, float]] = {}
    num_alleles = len(allele2value)

    log.info(f"Computing per-allele predictions on reference dataset for {num_alleles} alleles...")
    for i, (allele, allele_value) in enumerate(allele2value.items(), start=1):
        reference_data_module.overridden_input_feature_name2value = {"allele": allele_value}
        predictions: list[dict[str, torch.Tensor]] = trainer.predict(  # type: ignore
            model, datamodule=reference_data_module
        )
        output_name2predictions = process_outputs(predictions)
        reference_scores = output_name2predictions[deconvolution_output_name]

        # Exclude outliers from p_bar and sigma estimation
        z_scores = (reference_scores - np.mean(reference_scores)) / np.std(reference_scores)
        valid_indices = np.logical_and(z_scores > -z_score_threshold, z_scores < z_score_threshold)
        allele2stats[allele] = {
            "p_bar": np.mean(reference_scores[valid_indices]).item(),
            "sigma": np.std(reference_scores[valid_indices]).item(),
        }
        log.info(
            f"Prediction on reference dataset: {i}/{num_alleles} allele"
            f"{'s' if num_alleles > 1 else ''} done."
        )

    # Compute w, p_tilde and sigma_tilde
    w = 1 / (1 + np.exp((epoch - w_shift_parameter) / 10))
    p_u_bar = np.mean(
        [stat_name2stat_value["p_bar"] for stat_name2stat_value in allele2stats.values()]
    ).item()
    sigma_u = np.mean(
        [stat_name2stat_value["sigma"] for stat_name2stat_value in allele2stats.values()]
    ).item()

    allele2rescaling_params = {
        allele: {
            "p_tilde": w * allele2stats[allele]["p_bar"] + (1 - w) * p_u_bar,
            "sigma_tilde": w * allele2stats[allele]["sigma"] + (1 - w) * sigma_u,
        }
        for allele in allele2stats
    }

    save_json(
        allele2rescaling_params,
        model.model_path / "iterative_annotation" / f"rescaling_params_epoch={epoch}.json",
    )

    return allele2rescaling_params


def _annotate_ma_positive_data(
    model: ModelType,
    trainer: Trainer,
    ma_positive_df: pl.DataFrame,
    ma_positive_data_module: DataModule,
    deconvolution_identifier: str,
    deconvolution_output_name: str,
    allele2value: dict[str, np.ndarray] | None = None,
    allele2rescaling_params: dict[str, dict[str, float]] | None = None,
) -> pl.DataFrame:
    """Annotate MA positive data with latest model checkpoint.

    'allele2value' and 'allele2rescaling_params' are only used with training data.

    Args:
        model: LightningModule for the model.
        trainer: Trainer for the model.
        ma_positive_df: DataFrame for MA positive data.
        ma_positive_data_module: DataModule for the MA positive dataset.
        deconvolution_identifier: Deconvolution identifier.
        deconvolution_output_name: Name of the output to consider for deconvolution.
        allele2value: Optional mapping from allele to its transformed value.
        allele2rescaling_params: Optional mapping from an allele to its respective parameters
            'p_bar' and 'sigma' to use for prediction score rescaling.

    Returns:
        The DataFrame for deconvoluted (MA annotated) MA positive data.
    """
    log.info("Annotating MA positive data by selecting allele yielding maximal score...")
    predictions: list[dict[str, torch.Tensor]] = trainer.predict(  # type: ignore
        model, datamodule=ma_positive_data_module
    )
    output_name2predictions = process_outputs(predictions)
    deconvolution_output_predictions = output_name2predictions[deconvolution_output_name]

    if allele2value and allele2rescaling_params:
        log.info("Rescaling predictions of the positive training MA data...")
        deconvolution_output_predictions = _rescale_ma_positive_data(
            raw_predictions=deconvolution_output_predictions,
            alleles=ma_positive_df["allele"].to_numpy(),
            allele2value=allele2value,
            allele2rescaling_params=allele2rescaling_params,
        )

    predictions_identifier = f"output_{deconvolution_output_name}"
    ma_positive_df = ma_positive_df.with_columns(
        pl.Series(deconvolution_output_predictions).alias(predictions_identifier)
    )

    annotated_ma_positive_df = apply_deconvolution(
        df=ma_positive_df,
        predictions_identifier=predictions_identifier,
        deconvolution_identifier=deconvolution_identifier,
    )

    return annotated_ma_positive_df


def _annotate_ma_negative_data(
    ma_negative_df: pl.DataFrame,
    deconvolution_identifier: str,
) -> pl.DataFrame:
    """Annotate MA negative data with random allele from genotype.

    Args:
        ma_negative_df: DataFrame for MA negative data.
        deconvolution_identifier: Deconvolution identifier.

    Returns:
        The DataFrame for deconvoluted (MA annotated) MA negative data.
    """
    log.info("Annotating MA negative data by selecting random allele...")
    return (
        ma_negative_df.with_columns(
            pl.lit(np.random.rand(len(ma_negative_df))).alias("_random_score")
        )
        .filter(
            pl.col("_random_score") == pl.col("_random_score").max().over(deconvolution_identifier)
        )
        .drop("_random_score")
    )


def _annotate_ma_data(
    model: ModelType,
    trainer: Trainer,
    ma_positive_df: pl.DataFrame,
    ma_negative_df: pl.DataFrame,
    ma_positive_data_module: DataModule,
    deconvolution_identifier: str,
    deconvolution_output_name: str,
    allele2value: dict[str, np.ndarray] | None = None,
    allele2rescaling_params: dict[str, dict[str, float]] | None = None,
) -> pl.DataFrame:
    """Annotate MA data with latest model checkpoint.

    Args:
        model: LightningModule for the model.
        trainer: Trainer for the model.
        ma_positive_df: DataFrame for MA positive data.
        ma_negative_df: DataFrame for MA negative data.
        ma_positive_data_module: DataModule for the MA positive dataset.
        deconvolution_identifier: Deconvolution identifier.
        deconvolution_output_name: Name of the output to consider for deconvolution.
        allele2value: Optional mapping from allele to its transformed value.
        allele2rescaling_params: Optional mapping from an allele to its respective parameters
            'p_bar' and 'sigma' to use for prediction score rescaling.

    Returns:
        The deconvoluted (MA annotated) DataFrame.
    """
    annotated_ma_negative_df = _annotate_ma_negative_data(
        ma_negative_df=ma_negative_df,
        deconvolution_identifier=deconvolution_identifier,
    )
    annotated_ma_positive_df = _annotate_ma_positive_data(
        model=model,
        trainer=trainer,
        ma_positive_df=ma_positive_df,
        ma_positive_data_module=ma_positive_data_module,
        deconvolution_identifier=deconvolution_identifier,
        deconvolution_output_name=deconvolution_output_name,
        allele2value=allele2value,
        allele2rescaling_params=allele2rescaling_params,
    ).select(annotated_ma_negative_df.columns)

    return pl.concat([annotated_ma_negative_df, annotated_ma_positive_df])


def _rescale_ma_positive_data(
    raw_predictions: np.ndarray,
    alleles: np.ndarray,
    allele2value: dict[str, np.ndarray],
    allele2rescaling_params: dict[str, dict[str, float]],
) -> np.ndarray:
    """Rescale predictions from MA positive data.

    If "Prediction Score Rescaling" is used, the raw predictions of the training MA positive samples
    are re-scaled, following the process described in the "Prediction Score Rescaling" section of
    "NNAlign_MA", Alvarez et al., 2019, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6885703/.

    Args:
        raw_predictions: Array for the raw predictions of the MA positive data.
        alleles: Array for the allele transformed values, of the same size as the raw predictions'
            array.
        allele2value: Mapping from allele to its transformed value.
        allele2rescaling_params: Mapping from an allele to its respective parameters 'p_bar' and
            'sigma' to use for prediction score rescaling.

    Returns:
        The post-processed predictions of the MA data.

    Raise:
        ValueError:
            - If the arrays for the raw predictions and the alleles' transformed values are not
              of the same size.
            - If no rescaling parameters are provided.
    """
    num_samples_predictions = len(raw_predictions)
    num_samples_alleles = len(alleles)
    if num_samples_predictions != num_samples_alleles:
        raise ValueError(
            "Length of the arrays for raw predictions and allele values are not equal. Their"
            f"respective lengths are: {num_samples_predictions:_} and {num_samples_alleles:_}."
        )

    if len(allele2rescaling_params) == 0:
        raise ValueError(
            "No rescaling parameters provided. Please provide 'allele2value' and "
            "'allele2rescaling_params' to use rescale MA positive data."
        )

    rescaled_predictions = np.empty_like(raw_predictions)
    for allele, scaling_params in allele2rescaling_params.items():
        mask = np.all(alleles == allele2value[allele], axis=1)
        rescaled_predictions[mask] = (
            raw_predictions[mask] - scaling_params["p_tilde"]
        ) / scaling_params["sigma_tilde"]

    return rescaled_predictions
