"""Unit tests related to bench_mhc/utils/logging/system.py."""

import logging
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from bench_mhc.utils.logging.system import BackupCountFileHandler
from bench_mhc.utils.logging.system import CustomFormatter
from bench_mhc.utils.logging.system import get


@pytest.fixture
def log_directory(tmp_path: Path) -> Path:
    """Create a temporary directory for log files."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)

    return log_directory


@pytest.fixture
def log_file(log_directory: Path) -> Path:
    """Create a log file path."""
    return log_directory / "test.log"


@pytest.mark.parametrize("log_level", ["INFO", "DEBUG"])
@pytest.mark.parametrize("log_directory_", [None, "logs"])
def test_get_logger(log_level: str, log_directory_: str | None) -> None:
    """Test the get function configures loggers correctly.

    Tests that the logger is configured with:
    - The correct name
    - The specified log level from environment variable
    - The right number and type of handlers based on LOG_DIRECTORY env var:
        - Always has a StreamHandler for stdout
        - Has an additional BackupCountFileHandler if LOG_DIRECTORY is set
    - Proper CustomFormatter on all handlers

    Args:
        log_level: The log level to test ("INFO" or "DEBUG")
        log_directory_: The log directory path to test (None or "logs")
    """
    os_environ_mock = (
        {"LOG_LEVEL": log_level, "LOG_DIRECTORY": log_directory_}
        if log_directory_ is not None
        else {
            "LOG_LEVEL": log_level,
        }
    )

    expected_number_of_handlers = 1 if log_directory_ is None else 2
    name = f"test_utils_logging_system_{log_level}_{log_directory_}"

    with patch.dict(os.environ, os_environ_mock, clear=True):
        logger = get(name)
        assert logger.name == name
        assert logger.level == getattr(logging, log_level)
        assert len(logger.handlers) == expected_number_of_handlers
        assert isinstance(logger.handlers[0].formatter, CustomFormatter)

        if expected_number_of_handlers == 2:
            assert isinstance(logger.handlers[1], logging.StreamHandler)


def test_custom_formatter() -> None:
    """Test that CustomFormatter formats log records correctly."""
    formatter = CustomFormatter()

    current_time = time.time()

    message = (
        "This is an extremely long message that will need to be "
        "wrapped because it exceeds the maximum width limit of the custom formatter "
        "which is set to 110 characters. This message will be split into multiple lines "
        "to test the wrapping functionality."
    )

    expected_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
    expected_line = 42
    expected_level = "INFO"
    expected_pid = 1234

    # Time is retrieved with the time module in LogRecord
    with patch("time.time", return_value=current_time) as mock_format_time:
        record = logging.LogRecord(
            name="test_utils_logging_system",
            level=getattr(logging, expected_level),
            pathname="module/dummy.py",
            lineno=expected_line,
            msg=message,
            args=(),
            exc_info=None,
            func="dummy_function",
        )
        mock_format_time.assert_called_once()
    record.process = expected_pid

    formatted = formatter.format(record)

    expected_formatted_base_msg = (
        f"{expected_time} | {expected_level : <8} | PID {expected_pid : <5} | "
        f"dummy:dummy_function:{expected_line}          | "
    )
    expected_formatted_msg = (
        expected_formatted_base_msg
        + (
            "This is an extremely long message that will need to be wrapped because "
            "it exceeds the maximum width limit of\n"
        )
        + expected_formatted_base_msg
        + (
            " the custom formatter which is set to 110 characters. "
            "This message will be split into multiple lines to test\n"
        )
        + expected_formatted_base_msg
        + " the wrapping functionality."
    )

    assert formatted == expected_formatted_msg


def test_custom_formatter_with_exception() -> None:
    """Test that CustomFormatter handles exceptions correctly."""
    formatter = CustomFormatter()

    # Produce the error object to pass to logging.LogRecord
    try:
        raise ValueError("Test error")
    except ValueError:
        exc_info = sys.exc_info()

    current_time = time.time()
    expected_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))

    with patch("time.time", return_value=current_time) as mock_format_time:
        record = logging.LogRecord(
            name="test_utils_logging_system",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=exc_info,
        )
        mock_format_time.assert_called_once()
    record.process = 1234
    formatted = formatter.format(record)

    # Constructing expected message using exc_info components
    exc_type, exc_value, exc_traceback = exc_info
    expected_formatted = (
        f"{expected_time} | ERROR    | PID 1234  | test:None:42                     |"
        " Test message\n"
        "Traceback (most recent call last):\n"
        f'  File "{exc_traceback.tb_frame.f_code.co_filename}", line {exc_traceback.tb_lineno}, '  # type: ignore
        f"in {exc_traceback.tb_frame.f_code.co_name}\n"  # type: ignore
        f'    raise {exc_type.__name__}("{str(exc_value)}")\n'  # type: ignore
        f"{exc_type.__name__}: {str(exc_value)}"  # type: ignore
    )

    assert formatted == expected_formatted


@pytest.mark.parametrize("backup_count", [1, 2, 3])
def test_backup_count_file_handler(log_file: Path, backup_count: int) -> None:
    """Test that BackupCountFileHandler manages log files correctly."""
    for i in range(5):
        log_file.with_name(f"test_{i}.log").touch()

    handler = BackupCountFileHandler(filename=log_file, backup_count=backup_count)
    handler.clean_logs_directory()

    # Check that only the `backup_count` most recent files are kept
    remaining_files = sorted(log_file.parent.glob("*.log"))
    assert len(remaining_files) == backup_count


def test_backup_count_file_handler_with_unexisting_file(log_file: Path) -> None:
    """Test that BackupCountFileHandler correctly ignores files deleted by another process."""
    unexisting_file = Mock()
    unexisting_file.stat.return_value.st_mtime = 1.0
    unexisting_file.unlink.side_effect = FileNotFoundError
    mock_files = [unexisting_file]

    with patch("bench_mhc.utils.logging.system.Path") as mock_path:
        mock_parent = MagicMock()
        mock_parent.iterdir.return_value = mock_files
        mock_path.return_value.parent = mock_parent

        handler = BackupCountFileHandler(filename=log_file, backup_count=0)
        handler.clean_logs_directory()
