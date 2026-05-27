"""Module to run bash commands."""

import subprocess
from typing import AnyStr


def run_bash(
    command: str,
    stdin: AnyStr | None = None,
    universal_newlines: bool = True,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> AnyStr:
    """Run a string command in bash.

    Args:
        command: bash command to run as string
        stdin: the executed command's standard input.
        universal_newlines: if True stdin, stdout and stderr are opened in text mode.
            Otherwise, they are opened in binary mode.
        cwd: the directory in which the command will be executed.
        env: the environment variables for the executed command

    Returns: stdout of the bash command

    Raises:
        SubprocessError: the command finished with exit code != 0
    """
    if isinstance(stdin, bytes) and universal_newlines:
        raise ValueError("'universal_newline' can't be true when the stdin is a binary stream")

    args = {
        "args": command,
        "executable": "/bin/bash",
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "shell": True,
        "cwd": cwd,
        "env": env,
        "universal_newlines": universal_newlines,
    }
    if stdin:
        args["stdin"] = subprocess.PIPE

    p = subprocess.Popen(**args)  # type: ignore
    stdout, stderr = p.communicate(stdin)

    if p.returncode != 0:
        if not universal_newlines:
            stderr = stderr.decode()
        raise subprocess.SubprocessError(
            f"The command '{command}' finished with exit code {p.returncode} - {stderr}"
        )

    return stdout.strip()
