"""Correlation rule: Curvature Traps.

Detects when edge-of-stability dynamics co-occur with loss plateaus,
indicating that training is trapped by loss landscape curvature.
"""

from __future__ import annotations

from torchmortem.registry import register_correlation_rule
from torchmortem.types import (
    CorrelationRule,
    Finding,
    Insight,
    Reference,
    Severity,
)


def _condition(findings: list[Finding]) -> bool:
    """Check for both edge-of-stability / curvature issues and loss plateau."""
    has_curvature = any(
        f.category == "loss_dynamics"
        and ("edge of stability" in f.title.lower() or "plateau" in f.title.lower())
        and f.severity >= Severity.WARNING
        for f in findings
    )
    # Accept if we have at least one loss_dynamics finding at WARNING+ level
    # and the combination suggests curvature issues
    has_plateau = any(
        f.category == "loss_dynamics" and "plateau" in f.title.lower() for f in findings
    )
    has_eos = any(
        f.category == "loss_dynamics" and "edge of stability" in f.title.lower() for f in findings
    )
    return has_plateau and has_eos


def _synthesize(findings: list[Finding]) -> Insight:
    """Produce a curvature trap insight."""
    relevant = [f for f in findings if f.category == "loss_dynamics"]

    return Insight(
        rule_name="curvature_traps",
        title="Training trapped by loss landscape curvature",
        summary=(
            "Edge-of-stability dynamics combined with a loss plateau suggest "
            "the optimizer is oscillating along high-curvature directions "
            "without making progress."
        ),
        detail=(
            "The loss dynamics detector flagged both edge-of-stability behavior "
            "(Hessian top eigenvalue near 2/lr) and a loss plateau. Together, "
            "these indicate the optimizer is spending energy bouncing along "
            "sharp directions rather than descending. The learning rate may be "
            "at the maximum stable value, preventing further progress without "
            "curvature-aware adjustments."
        ),
        contributing_findings=relevant,
        remediation=[
            "Try a learning rate warmup/cooldown schedule.",
            "Switch to a second-order or curvature-aware optimizer (e.g. SAM, "
            "Sharpness-Aware Minimization).",
            "Reduce the learning rate to exit the edge-of-stability regime.",
            "Add gradient clipping to dampen oscillations.",
        ],
        references=[
            Reference(
                title="Gradient Descent on Neural Networks Typically Occurs "
                "at the Edge of Stability",
                authors="Cohen et al.",
                year=2021,
            ),
            Reference(
                title="Sharpness-Aware Minimization for Efficiently Improving Generalization",
                authors="Foret et al.",
                year=2021,
            ),
        ],
    )


register_correlation_rule(
    CorrelationRule(
        name="curvature_traps",
        description=(
            "Edge-of-stability plus loss plateau -- training is trapped by "
            "high curvature in the loss landscape."
        ),
        required_categories=["loss_dynamics"],
        priority=80,
        condition=_condition,
        synthesize=_synthesize,
    )
)
