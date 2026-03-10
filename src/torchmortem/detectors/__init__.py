"""Detectors package -- diagnostic analysis of collected data."""

from torchmortem.detectors.dead_units import DeadUnitDetector
from torchmortem.detectors.gradient_flow import GradientFlowDetector
from torchmortem.detectors.gradient_noise import GradientNoiseDetector
from torchmortem.detectors.loss_dynamics import LossDynamicsDetector
from torchmortem.detectors.rank_collapse import RankCollapseDetector
from torchmortem.detectors.saturation import SaturationDetector
from torchmortem.detectors.update_ratio import UpdateRatioDetector
from torchmortem.detectors.weight_norm import WeightNormDetector

__all__ = [
    "DeadUnitDetector",
    "GradientFlowDetector",
    "GradientNoiseDetector",
    "LossDynamicsDetector",
    "RankCollapseDetector",
    "SaturationDetector",
    "UpdateRatioDetector",
    "WeightNormDetector",
]
