"""Module to define custom errors."""


class NotFittedError(Exception):
    """Exception raised when a variable is used to transform before fitting."""
