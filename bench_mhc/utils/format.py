"""Module to define functions to format python data structure for better display."""

from collections.abc import Iterable
from typing import Any


def format_iterable(iterable_: Iterable[Any], limit: int | None = None) -> str:
    """Format an iterable with string for better display.

    Args:
        iterable_: Elements to format.
        limit: Optional maximal number of elements to include in the formatted string.

    Returns:
        The formatted string.

    >>> format_iterable(["key1", "key2"])
    "'key1' | 'key2'"

    >>> format_iterable(["key1"])
    "'key1'"

    >>> format_iterable(["key1", "key2", "key3"], 10)
    "'key1' | 'key2' | 'key3'"

    >>> format_iterable(["key1", "key2", "key3"], 2)
    "'key1' | 'key2' | ..."
    """
    formatted_elements = [f"'{item}'" for item in sorted(iterable_)]
    if limit is not None and len(formatted_elements) > limit:
        formatted_elements = formatted_elements[:limit]
        formatted_elements.append("...")

    return " | ".join(formatted_elements)


def format_dict(dict_: dict[str, Any], limit: int | None = None) -> str:
    """Format a dictionary with string for better display.

    Args:
        dict_: Elements to format.
        limit: Optional maximal number of elements to include in the formatted string.

    Returns:
        The formatted string.

    >>> format_dict({"A":1,"B":2,"C":3})
    "('A', 1) | ('B', 2) | ('C', 3)"

    >>> format_dict({"A":1,"B":2,"C":3}, 2)
    "('A', 1) | ('B', 2) | ..."

    >>> format_dict({"A":1,"B":2,"C":3}, 10)
    "('A', 1) | ('B', 2) | ('C', 3)"
    """
    formatted_elements = [f"{item}" for item in sorted(dict_.items())]
    if limit is not None and len(formatted_elements) > limit:
        formatted_elements = formatted_elements[:limit]
        formatted_elements.append("...")

    return " | ".join(formatted_elements)
