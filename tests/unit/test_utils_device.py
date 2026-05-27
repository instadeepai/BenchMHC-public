"""Unit tests related to bench_mhc/utils/device.py."""

import pytest

from bench_mhc.utils.device import get_devices_and_accelerator


@pytest.mark.parametrize(
    ("gpus", "expected_devices", "expected_accelerator"),
    [
        ("auto", "auto", "auto"),
        ([1], [1], "auto"),
        ([0], [0], "auto"),
        ([0, 1], [0, 1], "auto"),
        (None, "auto", "cpu"),
    ],
)
def test_get_devices_and_accelerator(
    gpus: list[int] | None,
    expected_devices: str | list[int],
    expected_accelerator: str,
) -> None:
    """Test that get_devices_and_accelerator returns the correct configuration."""
    devices, accelerator = get_devices_and_accelerator(gpus)
    assert devices == expected_devices
    assert accelerator == expected_accelerator


def test_get_devices_and_accelerator_invalid_input() -> None:
    """Test that get_devices_and_accelerator raises ValueError for invalid input."""
    with pytest.raises(ValueError, match="Invalid GPU configuration"):
        get_devices_and_accelerator({"gpus": "0"})  # type: ignore
