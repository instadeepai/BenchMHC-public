"""Module to define the dataset modes: train / val / test / predict."""

from enum import auto

from bench_mhc.utils.enum import AutoNameEnum


class Mode(AutoNameEnum):
    """Standard dataset modes.

    The following standard keys are defined:
    * `TRAIN`: training mode.
    * `VAL`: validation mode.
    * `TEST`: testing mode.
    * `PREDICT`: inference mode.
    """

    TRAIN = auto()
    VAL = auto()
    TEST = auto()
    PREDICT = auto()
