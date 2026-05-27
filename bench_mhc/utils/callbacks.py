"""Utils functions relative to training callbacks."""

import copy
from pathlib import Path
from typing import Any

import lightning.pytorch.callbacks as callback_lib
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import EarlyStopping

from bench_mhc.custom_objects.callbacks import ModelCheckpoint
from bench_mhc.utils.format import format_dict
from bench_mhc.utils.logging import system

log = system.get(__name__)


def load_callbacks_from_config(
    config: list[dict[str, Any]],
) -> list[Callback]:
    """Loads PyTorch Lightning callbacks from a configuration dictionary.

    Args:
        config: A list of dictionaries. Each dictionary represents a callback.
            The dictionary must contain a key "class_name" specifying the name
            of the callback class (as a string). The remaining keys in the
            dictionary are treated as keyword arguments to the callback's
            constructor.

    Returns:
        A list of instantiated callback objects.

    Raises:
        AttributeError: If a callback name specified in the configuration
            does not exist in `lightning.callbacks`.

    Example:
        config = [
            {"class_name": "ModelCheckpoint", "filepath": "my/path/to/ckpt"},
            {"class_name": "EarlyStopping", "monitor": "val_loss"},
            {"class_name": "LearningRateMonitor"}, # No kwargs
        ]
        callbacks = load_callbacks_from_config(config)
    """
    config = copy.deepcopy(config)
    callbacks: list[Callback] = []
    for callback_dict in config:
        callback_name = callback_dict.pop("class_name")
        if callback_name == "ModelCheckpoint":
            log.warning("Using a custom ModelCheckpoint callback.")
            callback_class = ModelCheckpoint
        else:
            try:
                callback_class = getattr(callback_lib, callback_name)
            except AttributeError:
                raise AttributeError(
                    f"Callback '{callback_name}' not found in lightning.callbacks."
                ) from None

        try:
            callbacks.append(callback_class(**callback_dict))
        except Exception as error:
            raise Exception(
                f"Error instantiating callback {callback_name} with kwargs: "
                f"{format_dict(callback_dict)}"
            ) from error

    return callbacks


def maybe_add_model_checkpoint_callback(configuration: dict[str, Any], model_path: Path) -> None:
    """Add the ModelCheckpoint callback if not provided in the configuration.

    Any non-provided default parameters will be initialized with our default values.

    Args:
        configuration: The configuration dictionary. This dictionary is
            modified in-place. It is expected to have a "callbacks" key,
            whose value is a list of dictionaries as accepted by `load_callbacks_from_config`.
        model_path: The path to the model.
    """
    default_model_ckpt_callback_configuration = {
        "class_name": "ModelCheckpoint",
        "dirpath": str(model_path / "checkpoints"),
        "filename": "best",
        "save_last": True,
        "monitor": "val/loss",
        "mode": "min",
        "verbose": True,
        "enable_version_counter": False,
    }

    if "callbacks" not in configuration["training"]:
        configuration["training"]["callbacks"] = []

    for callback in configuration["training"]["callbacks"]:
        if callback["class_name"] == "ModelCheckpoint":
            missing_keys = default_model_ckpt_callback_configuration.keys() - callback.keys()
            if missing_keys:
                missing_default_configuration = {
                    key: default_model_ckpt_callback_configuration[key] for key in missing_keys
                }
                log.info(
                    "ModelCheckpoint callback will be initialized with provided parameters: "
                    f"{format_dict(callback)} and the following default parameters: "
                    f"{format_dict(missing_default_configuration)}"
                )

                callback.update(missing_default_configuration)

            break
    else:
        log.warning(
            "ModelCheckpoint not in the callbacks section of the configuration. "
            "It will be instantiated with default arguments."
        )

        configuration["training"]["callbacks"].append(default_model_ckpt_callback_configuration)


def get_early_stopping_callback(callbacks: list[Callback]) -> EarlyStopping | None:
    """Get the EarlyStopping callback from the list of callbacks.

    Args:
        callbacks: The list of callbacks.

    Returns:
        The EarlyStopping callback or None if it is not in the list of callbacks.
    """
    return next((callback for callback in callbacks if isinstance(callback, EarlyStopping)), None)


def override_callbacks(
    callbacks: list[Callback],
    new_callbacks: list[Callback],
    callback_classes_to_exclude: tuple[type[Callback]],
) -> list[Callback]:
    """Override callbacks with new callbacks.

    Args:
        callbacks: The list of callbacks to be maybe overridden.
        new_callbacks: The list of new callbacks.
        callback_classes_to_exclude: Callback classes to exclude.

    Returns:
        The list of callbacks with the new callbacks and without the excluded callbacks.
    """
    updated_callbacks = []

    for callback in callbacks:
        if not isinstance(callback, callback_classes_to_exclude):
            for new_callback in new_callbacks:
                if new_callback.__class__.__name__ == callback.__class__.__name__:
                    updated_callbacks.append(new_callback)
                    break
            else:
                updated_callbacks.append(callback)

    return updated_callbacks
