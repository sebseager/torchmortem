"""Collectors package -- data collection via PyTorch hooks."""

# Import all collectors so they self-register via @register_collector.
from torchmortem.collectors.activation import ActivationCollector
from torchmortem.collectors.curvature import CurvatureCollector
from torchmortem.collectors.gradient import GradientCollector
from torchmortem.collectors.loss import LossCollector
from torchmortem.collectors.rank import RankCollector
from torchmortem.collectors.weight import WeightCollector

__all__ = [
    "ActivationCollector",
    "CurvatureCollector",
    "GradientCollector",
    "LossCollector",
    "RankCollector",
    "WeightCollector",
]
