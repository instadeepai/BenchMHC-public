"""Module to configure the system / application logger for the whole project.

To use it, add the following lines at the top of your file:

```
from bench_mhc.utils.logging import system

log = system.get(__name__)
```

The logger can be used in the file by calling :
- log.debug('My debug message.')
- log.info('My info message.')
- log.warning('My warning message.')
- log.critical('My critical message.')
- log.exception('My exception message.')

The last log can be used in try / except statement and it will log the full traceback.

By default, the log level is set to INFO, if you want to run as DEBUG you can do the following:
LOG_LEVEL=DEBUG python ...
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

current_log_file: Path | None = None

# Number of logs file to keep
DEFAULT_BACKUP_COUNT = 50
BACKUP_COUNT = int(os.environ.get("LOG_BACKUP_COUNT", DEFAULT_BACKUP_COUNT))


def get(name: str) -> logging.Logger:
    """Configure the logging.

    - a StreamHandler is always used to log to stdout.
    - if the env 'LOG_DIRECTORY' is available, a handler is also used to log the file
        'LOG_DIRECTORY/bench_mhc_{timestamp}.log'

    The level is configured thanks to the environment variable 'LOG_LEVEL' (default 'INFO').

    Args:
        name: Module name used to call the 'get' method.

    Returns:
        The logger configured with the 2 handlers.
    """
    global current_log_file

    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    logger.propagate = False

    # Check if we already have a StreamHandler
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(CustomFormatter())
        logger.addHandler(stdout_handler)

    log_directory_ = os.environ.get("LOG_DIRECTORY")

    if log_directory_ is not None:
        log_directory = Path(log_directory_)

        if not current_log_file:
            current_log_file = log_directory / datetime.now().strftime("%Y%m%d-%H%M%S")
            print(f"Using {current_log_file} to save logs…")
            current_log_file.parent.mkdir(exist_ok=True, parents=True)

        # Check if we already have a FileHandler for this file
        if not any(
            isinstance(handler, BackupCountFileHandler)
            and handler.baseFilename == str(current_log_file)
            for handler in logger.handlers
        ):
            file_handler = BackupCountFileHandler(
                filename=current_log_file, backup_count=BACKUP_COUNT
            )
            file_handler.setFormatter(CustomFormatter())
            logger.addHandler(file_handler)

    return logger


class CustomFormatter(logging.Formatter):
    r"""Custom formatter to have aligned logs.

    If a \n is used in the log message it will log 2 lines

    Examples :
    2020-04-03 14:20:21 | DEBUG    | logger:log_me:57         | This is an info log
    2020-04-03 14:20:21 | CRITICAL | logger:log_me:57         | This is a critical log
    """

    message_width = 110
    cpath_width = 32
    date_format = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        """Main method to format a given record.

        Args:
            record: Record object to display.

        Returns:
            Record formatted as string.
        """
        cpath = f"{record.module}:{record.funcName}:{record.lineno}"
        cpath = cpath[-self.cpath_width :].ljust(self.cpath_width)

        date = self.formatTime(record, self.date_format)
        prefix = f"{date} | {record.levelname : <8} | PID {record.process: <5} | {cpath}"

        lines = record.getMessage().split("\n")

        # fixing max length
        limited_lines = []
        for line in lines:
            while len(line) > self.message_width:
                splitting_position = self.message_width
                substring = line[: splitting_position - 1]
                last_space_position = substring.rfind(" ")
                if last_space_position > 0:
                    splitting_position = last_space_position

                substring = line[:splitting_position]
                limited_lines.append(substring)
                line = line[splitting_position:]

            limited_lines.append(line)

        formatted_messages = []
        for line in limited_lines:
            formatted_messages.append(f"{prefix} | {line}")

        final_message = "\n".join(formatted_messages).rstrip()

        if record.exc_info and not record.exc_text:
            # Cache the traceback text to avoid converting it multiple times
            record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            if final_message[-1:] != "\n":
                final_message += "\n"

            final_message += record.exc_text

        return final_message


class BackupCountFileHandler(logging.FileHandler):
    """Extension of 'logging.FileHandler' to keep only 'backup_count' files.

    All the files will have the following pattern:
    - 'bench_mhc_{timestamp}.log' if experiment is launched locally.

    When this class is instantiated, it will delete old files based on last modified time.

    Inspired by https://stackoverflow.com/a/59565327
    """

    def __init__(self, backup_count: int, **kwargs: Any) -> None:
        """Initialize the BackupCountFileHandler.

        Args:
            backup_count: Number of files to keep.
            kwargs: Arguments to create the ReopenFileStreamLogger.
        """
        super().__init__(**kwargs)
        self.backup_count = backup_count
        self.clean_logs_directory()

    def clean_logs_directory(self) -> None:
        """Clean the log directory to keep the latest self.backup_count files."""
        all_files_sorted = sorted(
            Path(self.baseFilename).parent.iterdir(),
            # Sort based on the last modified time
            key=lambda file_path: file_path.stat().st_mtime,
            reverse=True,
        )

        for fp in all_files_sorted[self.backup_count :]:
            # In case of multiple processes running, the file
            # might have been deleted by another process.
            try:
                fp.unlink()
            except FileNotFoundError:
                pass
