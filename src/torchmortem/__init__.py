"""torchmortem -- diagnostic autopsy for PyTorch training runs."""

import sys
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from torchmortem.core import Autopsy
from torchmortem.types import (
    CollectorState,
    CorrelationRule,
    Finding,
    HealthScore,
    Insight,
    Reference,
    Report,
    RunMetadata,
    SamplingConfig,
    Severity,
)

# Ensure all built-in collectors, detectors, and renderers are registered
# on import by importing their packages.
import torchmortem.collectors  # noqa: F401
import torchmortem.detectors  # noqa: F401
import torchmortem.renderers  # noqa: F401

__all__ = [
    "Autopsy",
    "CollectorState",
    "CorrelationRule",
    "Finding",
    "HealthScore",
    "Insight",
    "Reference",
    "Report",
    "RunMetadata",
    "SamplingConfig",
    "Severity",
]


def default_logging(level: int = logging.INFO, propagate=False) -> None:
    """Helper to configure torchmortem's recommended log format."""
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Attach handler to root logger
    logger = logging.getLogger("torchmortem")
    logger.setLevel(level)
    logger.addHandler(handler)

    # Whether or not to propagate to root logger
    # Disable if you're getting duplicate logs
    logger.propagate = propagate
