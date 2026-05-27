"""Unit tests related to bench_mhc/utils/ram.py."""

import resource
from unittest.mock import mock_open
from unittest.mock import patch

import pytest

from bench_mhc.utils.ram import set_container_ram_limit


@pytest.mark.parametrize(
    ("memory_limit", "expected_calls", "cgroup_standard"),
    [
        ("1073741824", 1, 2),  # 1GB in bytes, with cgroup v2
        ("1073741824", 1, 1),  # 1GB in bytes, with cgroup v1
        ("max\n", 0, 2),  # No limit set
        ("invalid", 0, 2),  # Invalid value
    ],
)
def test_set_container_ram_limit(
    memory_limit: str,
    expected_calls: int,
    cgroup_standard: int,
) -> None:
    """Test the set_container_ram_limit function.

    Args:
        memory_limit: The memory limit value to test with.
        expected_calls: The expected number of times setrlimit should be called.
        cgroup_standard: The standard of the cgroup file.
    """
    with (
        patch("pathlib.Path.exists") as mock_exists,
        patch("builtins.open", mock_open(read_data=memory_limit)),
        patch("resource.setrlimit") as mock_setrlimit,
    ):
        mock_exists.side_effect = [cgroup_standard == 2, cgroup_standard == 1]

        set_container_ram_limit()

        assert mock_setrlimit.call_count == expected_calls
        if expected_calls > 0:
            mock_setrlimit.assert_called_once_with(
                resource.RLIMIT_AS, (int(memory_limit), int(memory_limit))
            )


def test_set_container_ram_limit_no_files() -> None:
    """Test set_container_ram_limit when no cgroup files exist."""
    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("resource.setrlimit") as mock_setrlimit,
    ):
        set_container_ram_limit()
        mock_setrlimit.assert_not_called()
