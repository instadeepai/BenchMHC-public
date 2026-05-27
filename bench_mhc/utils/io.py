"""Module to define I/O functions."""

import json
from collections import defaultdict
from json import JSONEncoder
from pathlib import Path
from typing import Any

import polars as pl
import yaml


def save_txt(
    data: list[str],
    file_path: str | Path,
) -> None:
    """Save a dict into a local .txt file.

    Each item of the list will be saved on a new line of the .txt file.

    Args:
        data: A list of strings to save in the .txt file.
        file_path: The path to the local .txt file.
    """
    file_path_ = Path(file_path)
    file_path_.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path_, "w") as file:
        for item in data:
            file.write(f"{item}\n")


def load_txt(file_path: str | Path, strip_whitespaces: bool = True) -> list[str]:
    """Load a local .txt file with the whitespaces at the beginning and end stripped.

    Each line of the file is an item in the returned list.

    Args:
        file_path: Path to the .txt file.
        strip_whitespaces: Whether to strip whitespaces at the beginning and end.

    Returns:
        The lines from the .txt file.
    """
    with open(file_path) as file:
        return [item.strip() if strip_whitespaces else item.rstrip("\n") for item in file]


def save_yml(
    dict_: dict,
    file_path: str | Path,
    **kwargs: Any,
) -> None:
    """Save a dict into a local .yml file.

    Args:
        dict_: A dict to save in .yml file.
        file_path: Path to save the dict to.
        kwargs: Keyword arguments to pass to 'yaml.dump'.

    Remarks:
    - by default:
        - 'default_flow_style=False'
        - 'sort_keys=False'
    - all keyword arguments will be forwarded to 'yaml.dump' function
    """
    kwargs["default_flow_style"] = kwargs.get("default_flow_style", False)
    kwargs["sort_keys"] = kwargs.get("sort_keys", False)

    file_path_ = Path(file_path)
    file_path_.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path_, "w") as file:
        yaml.dump(dict_, file, **kwargs)


def load_yml(file_path: str | Path, **kwargs: Any) -> dict:
    """Load a local .yml file.

    Args:
        file_path: Path to the .yml file.
        kwargs: Keyword arguments passed to 'yaml.full_load' function.

    Returns:
        The dictionary loaded from the .yml file.
    """
    with open(file_path) as file:
        dict_ = yaml.full_load(file, **kwargs)

    return dict_


class JSONEncoderWPath(JSONEncoder):
    """Custom json encoder to convert Path objects to str."""

    def default(self, o: Any) -> Any:  # pylint: disable=method-hidden
        """Override the default method to handle Path objects.

        Pylint is raising an error currently, cf https://github.com/PyCQA/pylint/issues/414.
        """
        if isinstance(o, Path):
            return str(o)

        return super().default(o)


def save_json(
    dict_: dict,
    file_path: str | Path,
    **kwargs: Any,
) -> None:
    """Save a dict into a local .json file.

    Remarks:
    - by default, 'indent=4' is used for a better display.
    - a custom encoder is used to support saving pathlib.Path objects.
    - all keyword arguments will be forwarded to 'json.dump' function

    Args:
        dict_: A dict that will be saved into a .json file.
        file_path: Path to the .json file.
        kwargs: Any additional keyword arguments will be forwarded to 'json.dump'.
    """
    kwargs["indent"] = kwargs.get("indent", 4)
    kwargs["cls"] = JSONEncoderWPath

    file_path_ = Path(file_path)
    file_path_.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path_, "w") as file:
        json.dump(dict_, file, **kwargs)


def load_json(file_path: str | Path, **kwargs: Any) -> dict:
    """Load a local .yml file.

    Args:
        file_path: Path to the .json file.
        kwargs: Keyword arguments passed to 'json.load' function.

    Returns:
        The dictionary loaded from the .json file.
    """
    with open(file_path) as file:
        dict_ = json.load(file, **kwargs)

    return dict_


def load_fasta(file_path: str | Path, columns: list[str]) -> pl.DataFrame:
    """Load fasta file to DataFrame."""
    key2seq: defaultdict[str, str] = defaultdict(str)

    with open(file_path) as file:
        for line in file:
            if line[0] == ">":
                newkey = line[1:-1]
                continue

            key2seq[newkey] += line.strip()

    return pl.DataFrame(list(key2seq.items()), schema=columns, orient="row")
