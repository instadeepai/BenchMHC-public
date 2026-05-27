"""Module to define callbacks for click arguments."""

import rich_click as click


def validate_strict_positive(
    ctx: click.Context,  # noqa: ARG001
    param: click.Option | click.Parameter,  # noqa: ARG001
    value: float | int | None,
) -> float | int | None:
    """Function to ensure the parameter value is strictly positive.

    >>> validate_strict_positive(None, None, 2)
    2

    >>> validate_strict_positive(None, None, 0)
    Traceback (most recent call last):
    click.exceptions.BadParameter: '0' provided, this parameter should be strictly greater than 0.

    >>> validate_strict_positive(None, None, -2)
    Traceback (most recent call last):
    click.exceptions.BadParameter: '-2' provided, this parameter should be strictly greater than 0.
    """
    if value is not None and value <= 0:
        raise click.BadParameter(
            f"'{value}' provided, this parameter should be strictly greater than 0."
        )

    return value


def format_gpu_arg(
    ctx: click.Context,  # noqa: ARG001
    param: click.Option | click.Parameter,  # noqa: ARG001
    value: str | None,
) -> list[int] | None:
    """Format GPU argument to a list of int(s).

    Args:
        ctx: Click context
        param: Click parameter
        value: Input value to format. Can be:
            - None
            - Single integer as string
            - Comma-separated integers

    Returns:
        Formatted GPU configuration as either:
            - None for CPU training
            - List of integers for specific GPU indices

    Raises:
        click.BadParameter: If the input does not contain only integers.

    >>> format_gpu_arg(None, None, None)

    >>> format_gpu_arg(None, None, "0")
    [0]

    >>> format_gpu_arg(None, None, "0,1")
    [0, 1]

    >>> format_gpu_arg(None, None, "A,B")
    Traceback (most recent call last):
    click.exceptions.BadParameter: 'A' provided, it should be a single integer for GPU id.
    """
    if value is None:
        return None

    gpu_indices = []
    for gpu_index in value.split(","):
        try:
            gpu_indices.append(int(gpu_index.strip()))
        except ValueError:
            raise click.BadParameter(
                f"'{gpu_index}' provided, it should be a single integer for GPU id."
            ) from None

    return gpu_indices
