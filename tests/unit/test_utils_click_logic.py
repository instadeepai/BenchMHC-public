"""Unit tests related to bench_mhc/utils/click/logic.py."""

from pathlib import Path

import pytest

from bench_mhc.utils.click.logic import check_abort_if_file_exists


@pytest.mark.parametrize(
    ("file_name", "file_exists"),
    [
        (None, False),
        ("check_abort_if_file_exists.csv", False),
        ("check_abort_if_file_exists.csv", True),
    ],
)
@pytest.mark.parametrize("force", [False, True])
def test_check_abort_if_file_exists(
    tmp_path: Path,
    file_name: str | None,
    file_exists: bool,
    force: bool,
) -> None:
    """Check check_abort_if_file_exists works as expected."""
    file_path = None
    if file_name is not None:
        file_path = tmp_path / file_name

        if file_exists:
            file_path.touch()

    if not force and file_path is not None and file_exists:
        msg = f"The file {file_path} already exists. Use --force if you want to overwrite it."
        with pytest.raises(FileExistsError, match=msg):
            check_abort_if_file_exists(file_path, force)

    else:
        check_abort_if_file_exists(file_path, force)
