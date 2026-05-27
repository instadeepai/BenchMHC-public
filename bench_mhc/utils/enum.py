"""Module to define enums."""

from enum import StrEnum


class AutoNameEnum(StrEnum):
    """Return the lower-case member name as enum values.

    Source: https://docs.python.org/3/library/enum.html#enum.Enum._generate_next_value_
    """

    @staticmethod
    def _generate_next_value_(
        name: str,
        start: int,  # noqa: ARG004
        count: int,  # noqa: ARG004
        last_values: list[str],  # noqa: ARG004
    ) -> str:
        """Generate the next value."""
        return name.lower()

    @classmethod
    def valid_values(cls) -> list[str]:
        """Return valid values for the enum."""
        return [str(member) for member in cls]
