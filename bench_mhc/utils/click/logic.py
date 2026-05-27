"""Module to define common logic patterns used for command lines."""

from pathlib import Path

from bench_mhc.utils.logging import system

log = system.get(__name__)


def check_abort_if_file_exists(path: str | Path | None, force: bool = False) -> None:
    """Check if existing path should be overwritten otherwise abort.

    Args:
        path: Optional path to check.
        force: Whether to skip checking if path exists.

    Raises:
        A FileExistsError if path exists and force is False.
    """
    if not force and path is not None and Path(path).exists():
        raise FileExistsError(
            f"The file {path} already exists. Use --force if you want to overwrite it."
        )
