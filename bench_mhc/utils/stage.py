"""Module to define the model development stages: fit / test / predict."""

from enum import auto

from bench_mhc.utils.enum import AutoNameEnum


class Stage(AutoNameEnum):
    """Standard modes used by PyTorch Lightning for model development stages.

    Cf. https://lightning.ai/docs/pytorch/stable/common/trainer.html#state.

    The following standard keys are defined:
    * `FIT`: training stage.
    * `VALIDATE`: validation stage.
    * `TEST`: testing stage.
    * `PREDICT`: inference stage.
    """

    FIT = auto()
    VALIDATE = auto()
    TEST = auto()
    PREDICT = auto()
