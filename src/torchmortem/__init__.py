"""torchmortem -- diagnostic autopsy for PyTorch training runs."""

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
