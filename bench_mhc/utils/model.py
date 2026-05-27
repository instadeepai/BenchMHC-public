"""Utils functions linked to models."""

import uuid
from datetime import datetime
from pathlib import Path

import lightning

from bench_mhc.constants import SEPARATOR
from bench_mhc.model import MODEL_NAME2LIGHTNING_MODULE
from bench_mhc.utils.io import load_json
from bench_mhc.utils.logging.system import get

log = get(__name__)


def get_directory_name(experiment_name: str) -> str:
    """Get the name of the directory to store experiment's artifact.

    The directory will have the following pattern:
    - if `experiment_name` is provided: {experiment_name}__{timestamp}__{uuid}

    Args:
        experiment_name: experiment name provided by the user.

    Returns:
        name of the directory to store the experiment's artifacts.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uuid_ = str(uuid.uuid4())[:3]

    directory_name = f"{experiment_name}{SEPARATOR}{timestamp}{SEPARATOR}{uuid_}"

    return directory_name


def load_model_from_path(
    model_path: str | Path,
) -> lightning.pytorch.LightningModule:
    """Loads a model from a path.

    Args:
        model_path: Path to the directory containing the model (hparams.json and checkpoints).

    Returns:
        The instantiated PyTorch Lightning module.

    Raises:
        FileNotFoundError: If required files are not found.
    """
    model_path_ = Path(model_path)

    hparams_path = model_path_ / "hparams.json"
    if not hparams_path.exists():
        raise FileNotFoundError(f"hparams.json not found at {hparams_path}")

    checkpoints_dir = model_path_ / "checkpoints"
    best_checkpoint_path = checkpoints_dir / "best.ckpt"
    # If the best checkpoint is not found, we fall back to previous checkpoints naming
    # before adding support for `best.ckpt` with https://github.com/instadeepai/BenchMHC/issues/68
    if not best_checkpoint_path.exists():
        log.info("Checkpoint with filename 'best.ckpt' not found, trying 'epoch*.ckpt' instead")
        checkpoint_paths = list(checkpoints_dir.glob("*epoch*.ckpt"))
        if len(checkpoint_paths) == 0:
            raise FileNotFoundError(f"No best checkpoint file found at {checkpoints_dir}")
        else:
            best_checkpoint_path = checkpoint_paths[0]

    configuration = load_json(hparams_path)["configuration"]

    lightning_module_class = MODEL_NAME2LIGHTNING_MODULE[configuration["model"]["class_name"]]

    # Try to get inputs from nested location
    inputs_hparams = configuration.get("variables", {}).get("inputs")

    # In case the inputs are not in the hparams, we load them from the checkpoint
    load_kwargs = {}
    if inputs_hparams is not None:
        load_kwargs["inputs"] = inputs_hparams

    return lightning_module_class.load_from_checkpoint(best_checkpoint_path, **load_kwargs)


def should_apply_sigmoid_for_percentile_rank(model_path: str | Path) -> bool:
    """Check if sigmoid should be applied before computing percentile ranks.

    This is needed when the model uses BCEWithLogitsLoss, which means it outputs
    logits instead of probabilities. For percentile rank calibration and conversion
    to work correctly, we need to convert logits to probabilities using sigmoid.

    Args:
        model_path: Path to the directory containing the model (hparams.json and checkpoints).

    Returns:
        True if sigmoid should be applied (model uses BCEWithLogitsLoss), False otherwise.
    """
    model_path_ = Path(model_path)
    hparams_path = model_path_ / "hparams.json"

    if not hparams_path.exists():
        log.warning(f"hparams.json not found at {hparams_path}. Assuming sigmoid is not needed.")
        return False

    configuration = load_json(hparams_path)["configuration"]

    # Check if any output uses BCEWithLogitsLoss
    outputs_config = configuration.get("variables", {}).get("outputs", {})

    for _output_name, output_config in outputs_config.items():
        loss_config = output_config.get("loss", {})
        if loss_config.get("class_name") == "BCEWithLogitsLoss":
            return True

    return False
