"""Unit tests related to bench_mhc/utils/logging/mlflow.py."""

from pathlib import Path
from unittest.mock import patch

import pytest
from lightning.pytorch.loggers import MLFlowLogger

from bench_mhc.utils.logging.mlflow import IntStepMLFlowLogger


@pytest.mark.parametrize(
    ("step", "expected_step"),
    [(None, None), (1, 1), (1.0, 1), (2.7, 2)],
)
def test_int_step_mlflow_logger_log_metrics(
    tmp_path: Path, step: int | float | None, expected_step: int | None
) -> None:
    """Test IntStepMLFlowLogger.log_metrics casts step to int and forwards to the parent."""
    logger = IntStepMLFlowLogger(experiment_name="test-exp", save_dir=str(tmp_path))

    with patch.object(MLFlowLogger, "log_metrics") as super_log_metrics:
        # The override's signature is `step: int | None` (matching the parent's), but
        # Lightning actually passes floats at runtime, which is the bug this subclass
        # exists to work around. Intentionally violate the declared type here.
        logger.log_metrics({"loss": 0.5}, step=step)  # type: ignore[arg-type]

    super_log_metrics.assert_called_once_with({"loss": 0.5}, step=expected_step)
