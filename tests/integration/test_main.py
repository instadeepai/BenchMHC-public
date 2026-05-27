"""Integration tests for the bench_mhc/__main__.py module."""

import click
import pytest
from click.testing import CliRunner

from bench_mhc.__main__ import main


def _assert_no_duplicated_parameter_names(params: list[click.Parameter]) -> bool:
    """Checks if a list of parameters has duplicated entries."""
    param_names = [param.name for param in params]

    return len(set(param_names)) == len(param_names)


@pytest.mark.parametrize("command", ["compute-nnalign-features", "format-allele-feature", "train"])
def test_commands(command: str) -> None:
    """Test the bench_mhc CLI commands."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert command in result.output
    assert _assert_no_duplicated_parameter_names(main.commands[command].params)
