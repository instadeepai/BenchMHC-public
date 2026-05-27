"""Module to define custom types for arguments in command lines."""

import os
from pathlib import Path
from typing import Any

import click

from bench_mhc.utils.io import load_txt


class RelativePath(click.Path):
    """Class similar to click.Path, but whose value is relative to a given directory.

    Attributes:
        relative_to: The directory to which the provided path is relative.
        strict: If False and the path cannot be used (e.g. does not exist) as relative to the
            given directory, we consider it absolute (or relative to the current working directory).
    """

    def __init__(self, relative_to: Path, strict: bool = True, *args: Any, **kwargs: Any) -> None:
        """Initialize the RelativePath.

        Args:
            relative_to: The directory to which the provided path is relative.
            strict: If False and the path cannot be used (e.g. does not exist) as relative to the
                given directory, we consider it absolute (or relative to the current working
                directory).
            *args: Arguments to pass to the superclass.
            **kwargs: Keyword arguments to pass to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.relative_to = relative_to
        self.strict = strict

    def convert(
        self,
        value: str | os.PathLike[str],
        param: click.core.Parameter | None,
        ctx: click.Context | None,
    ) -> str | bytes | os.PathLike[str]:
        """Convert the parameter provided by the user to a path relative to the given directory."""
        try:
            converted_value = super().convert(str(self.relative_to / value), param, ctx)

        except click.BadParameter as relative_path_error:
            if self.strict:
                self.fail(
                    f"Issue with the provided path '{value}' that should be relative to "
                    f"'{self.relative_to}': {relative_path_error}",
                )

            else:
                try:
                    converted_value = super().convert(value, param, ctx)
                except click.BadParameter as error:
                    self.fail(
                        f"Issue with the provided path '{value}': {error} "
                        f"Note that the path could not be used as relative to "
                        f"'{self.relative_to}': {relative_path_error}",
                    )

        return converted_value


class CommaSepOrFileToSet(click.ParamType):
    """Class to convert comma separated argument or .txt file to set.

    In case of a .txt file, each line will be an item in the set.

    Each item is validated according to the provided item_type.

    Attributes:
        name: The name of the type.
        item_click_type: The click type to use for each item.
    """

    name = "comma_separated_or_file_to_set"

    def __init__(self, item_click_type: click.ParamType | None = None) -> None:
        """Initialize the CommaSepOrFileToSet.

        Args:
            item_click_type: The click type to use for each item.
        """
        super().__init__()
        self.item_click_type = item_click_type

    def convert(
        self,
        value: str,
        param: click.core.Parameter | None,  # noqa: ARG002
        ctx: click.Context | None,  # noqa: ARG002
    ) -> set[Any]:
        """Convert the parameter provided by the user to a set.

        Each item is converted/validated according to the provided item_click_type.

        Args:
            value: The value provided by the user.
            param: The parameter this value is associated with.
            ctx: The context this value is being converted in.

        Returns:
            A set of validated items.

        Raises:
            click.BadParameter: If any of the items fail validation.
        """
        try:
            values = load_txt(value)
        except OSError:
            values = value.split(",")

        validated_items = set()
        for item_str in values:
            if self.item_click_type is not None:
                try:
                    validated_items.add(self.item_click_type.convert(item_str, param, ctx))
                except click.BadParameter as error:
                    raise click.BadParameter(
                        f"Item '{item_str}' failed validation: {error}"
                    ) from error
            else:
                validated_items.add(item_str)

        return validated_items
