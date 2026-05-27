"""Custom MLflow logger for Lightning compatibility."""

from collections.abc import Mapping

from lightning.pytorch.loggers import MLFlowLogger
from lightning_utilities.core import rank_zero


class IntStepMLFlowLogger(MLFlowLogger):
    """MLFlowLogger that casts metric steps to int.

    Lightning passes epoch-level metric steps as floats, which breaks
    MLflow's FileStore (it writes ``1.0`` but expects to parse ``int("1.0")``).
    """

    @rank_zero.rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: The metrics to log.
            step: The step at which the metrics were computed.
        """
        if step is not None:
            step = int(step)
        super().log_metrics(metrics, step=step)
