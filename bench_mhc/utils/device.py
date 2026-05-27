"""Device utilities for model training."""

from bench_mhc.utils.format import format_iterable
from bench_mhc.utils.logging import system

log = system.get(__name__)


def get_devices_and_accelerator(
    gpus: str | list[int] | None,
) -> tuple[str | int | list[int], str]:
    """Get the devices and accelerator configuration for PyTorch Lightning.

    Args:
        gpus: GPU configuration. Can be:
            - "auto": Use all available GPUs
            - list[int]: List of specific GPU indices to use (e.g. [0,1] for GPUs 0 and 1)
            - None: Use CPU

    Returns:
        A tuple containing:
            - devices: The devices configuration for PyTorch Lightning:
                - "auto": Use all available GPUs if GPUs available
                - list[int]: List of specific GPU indices to use
            - accelerator: The accelerator type:
                - "auto": Use GPU acceleration
                - "cpu": Use CPU acceleration

    Raises:
        ValueError: If the GPU configuration is invalid (not "auto", None, or a list of integers).
    """
    if gpus == "auto":
        log.warning(
            "No specific GPUs provided in the configuration, "
            "all available GPUs will be automatically used."
        )
    elif isinstance(gpus, list):
        log.info(f"The following GPUs will be used: {format_iterable(gpus)}.")
    elif gpus is None:
        log.warning(f"'gpu={gpus}' provided. CPU acceleration will be used.")
    else:
        raise ValueError(f"Invalid GPU configuration: {gpus}")

    devices = gpus if gpus is not None else "auto"
    accelerator = "auto" if gpus is not None else "cpu"

    return devices, accelerator
