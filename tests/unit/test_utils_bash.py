"""Module used to define all the tests linked to bench_mhc/utils/bash.py module."""

import subprocess

import pytest

from bench_mhc.utils.bash import run_bash


def test_failing_command() -> None:
    """Error (SubprocessError) is raised if the command failed."""
    command = "not_available_command"

    msg = f"The command '{command}' finished with exit code"
    with pytest.raises(subprocess.SubprocessError, match=msg):
        run_bash(command)

    with pytest.raises(subprocess.SubprocessError, match=msg):
        run_bash(command, universal_newlines=False)


def test_command() -> None:
    """Check the stdout is returned by the 'run_bash' function."""
    command = "echo 'test OK'"
    output = run_bash(command)

    assert output == "test OK"


def test_command_with_stdin_text() -> None:
    """Test command with text stdin input."""
    command = "cat"
    stdin_text = "Hello World"
    output = run_bash(command, stdin=stdin_text)

    assert output == stdin_text


def test_command_with_stdin_bytes() -> None:
    """Test command with binary stdin input."""
    command = "cat"
    stdin_bytes = b"Hello World"
    output = run_bash(command, stdin=stdin_bytes, universal_newlines=False)

    assert output == stdin_bytes


def test_command_with_stdin_bytes_and_text_mode_raises_error() -> None:
    """Test that ValueError is raised when stdin is bytes and universal_newlines is True."""
    command = "cat"
    stdin_bytes = b"Hello World"

    with pytest.raises(
        ValueError, match="'universal_newline' can't be true when the stdin is a binary stream"
    ):
        run_bash(command, stdin=stdin_bytes, universal_newlines=True)


def test_command_with_cwd() -> None:
    """Test command with custom working directory."""
    command = "pwd"
    output = run_bash(command, cwd="/tmp")

    assert "/tmp" in output


def test_command_with_env() -> None:
    """Test command with custom environment variables."""
    command = "echo $CUSTOM_VAR"
    env = {"CUSTOM_VAR": "test_value"}
    output = run_bash(command, env=env)

    assert output == "test_value"


def test_command_with_binary_output() -> None:
    """Test command with binary output mode."""
    command = "echo 'test binary'"
    output = run_bash(command, universal_newlines=False)

    assert output == b"test binary"


def test_command_with_stderr_error() -> None:
    """Test command that writes to stderr but succeeds."""
    command = "echo 'test' >&2 && echo 'success'"
    output = run_bash(command)

    assert output == "success"


def test_command_with_none_stdin() -> None:
    """Test command with None stdin (should work normally)."""
    command = "echo 'test'"
    output = run_bash(command, stdin=None)

    assert output == "test"
