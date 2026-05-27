"""Module to configure the experiment loggers."""

import os
from pathlib import Path
from types import MappingProxyType

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import MLFlowLogger

from bench_mhc.model import ModelType
from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.logging import system
from bench_mhc.utils.logging.mlflow import IntStepMLFlowLogger

log = system.get(__name__)

LOGGER_NAME2LOGGER_CLASS = MappingProxyType(
    {"CSVLogger": CSVLogger, "MLflowLogger": IntStepMLFlowLogger}
)


def get_logger(
    model: ModelType,
    training_path: Path,
    validation_path: Path,
    ma_training_path: Path | None,
    ma_validation_path: Path | None,
) -> CSVLogger | MLFlowLogger:
    """Get the experiment logger.

    Note: When using `MLflowLogger`, the underlying MLflow client auto-reads the
    `MLFLOW_WORKSPACE` env var to bind runs to a specific workspace. If unset, the
    default workspace is used.

    Args:
        model: The Lightning model for which the logger should be created.
        training_path: Path to the training data.
        validation_path: Path to the validation data.
        ma_training_path: Optional path to the MA training data.
        ma_validation_path: Optional path to the MA validation data.

    Returns:
        The experiment logger.

    Raises:
        ValueError: If the provided logger configuration is invalid.
        ValueError: If the MLFLOW_EXPERIMENT_NAME environment variable is not set while using
            MLflowLogger.
    """
    logger_configuration = model.configuration["training"].get("logger", {})
    logger_class_name = logger_configuration.get("class_name")
    if logger_class_name == "CSVLogger":
        return CSVLogger(model.model_path)

    elif logger_class_name == "MLflowLogger":
        tags = ["train", training_path.name, validation_path.name]
        if ma_training_path is not None:
            tags.append(ma_training_path.name)
        if ma_validation_path is not None:
            tags.append(ma_validation_path.name)
        tags.extend(logger_configuration.get("tags", []))

        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
        if experiment_name is None:
            raise ValueError("MLFLOW_EXPERIMENT_NAME environment variable is not set.")

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        mlflow_logger = IntStepMLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            run_name=model.model_path.name,
            tags={tag: "" for tag in tags},
            synchronous=False,
        )
        run_type = "iterative_annotation" if ma_training_path is not None else "training"
        mlflow_logger.log_hyperparams(
            {
                "run_type": run_type,
                "model_name": model.model_path.name,
                "training_path": str(training_path),
                "validation_path": str(validation_path),
            }
        )
        run_id = mlflow_logger.run_id
        if run_id is not None:
            log.info("Using MLflow to log experiment artifacts. Run ID: %s.", run_id)
            model.hparams.update({"mlflow_run_id": run_id})

        return mlflow_logger

    else:
        raise ValueError(
            f"Unknown 'training.logger.class_name' in the configuration: {logger_class_name}. "
            f"Possible values: {format_iterable(LOGGER_NAME2LOGGER_CLASS)}. "
        )
