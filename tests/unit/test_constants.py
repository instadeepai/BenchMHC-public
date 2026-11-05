"""Unit tests related to bench_mhc/constants.py."""

import importlib
from pathlib import Path

import pytest

import bench_mhc.constants


@pytest.mark.parametrize(
    ("variable", "constant"),
    [
        ("BENCH_MHC_CACHE_DIRECTORY", "CACHE_DIRECTORY"),
        ("BENCH_MHC_MODEL_DIRECTORY", "MODEL_DIRECTORY"),
    ],
)
def test_written_directories_are_overridable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str, constant: str
) -> None:
    """Test the two directories the package writes to can be placed elsewhere.

    They default under ROOT_DIRECTORY, which for an installed package is `site-packages`
    and is read-only to a process that is not root, so a container has to be able to put
    them somewhere writable without patching the module.

    Args:
        tmp_path: Directory to point the constant at.
        monkeypatch: Fixture used to set the environment variable.
        variable: The environment variable that overrides the constant.
        constant: The name of the constant it overrides.
    """
    monkeypatch.delenv(variable, raising=False)
    default = getattr(importlib.reload(bench_mhc.constants), constant)
    assert default == bench_mhc.constants.ROOT_DIRECTORY / default.name

    monkeypatch.setenv(variable, str(tmp_path / "elsewhere"))
    try:
        assert getattr(importlib.reload(bench_mhc.constants), constant) == tmp_path / "elsewhere"
    finally:
        monkeypatch.undo()
        importlib.reload(bench_mhc.constants)

    assert getattr(bench_mhc.constants, constant) == default
