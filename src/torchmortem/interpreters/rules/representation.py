"""Correlation rule: Representation Bottleneck.

Detects when rank collapse co-occurs with loss degradation, indicating
that the network is losing representational capacity.
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
    """Check that both rank collapse and some loss issue are present."""
    has_rank = any(
        f.category == "rank_collapse" and f.severity >= Severity.WARNING for f in findings
    )
    has_loss = any(
        f.category == "loss_dynamics" and f.severity >= Severity.WARNING for f in findings
    )
    return has_rank and has_loss


def _synthesize(findings: list[Finding]) -> Insight:
    """Produce a representation bottleneck insight."""
    rank_findings = [f for f in findings if f.category == "rank_collapse"]
    loss_findings = [f for f in findings if f.category == "loss_dynamics"]

    collapsed_layers = []
    for f in rank_findings:
        collapsed_layers.extend(f.affected_layers)

    return Insight(
        rule_name="representation_bottleneck",
        title="Representation bottleneck -- rank collapse limiting capacity",
        summary=(
            "Layer representations are collapsing to a low-rank subspace "
            "while the loss fails to improve, indicating a capacity bottleneck."
        ),
        detail=(
            "The rank collapse detector found effective rank dropping "
            "significantly in one or more layers, while the loss dynamics "
            "detector flagged degradation or plateau. This suggests the "
            "network is losing the ability to represent the data diversity "
            "needed for further learning. "
            f"Layers with rank collapse: {', '.join(collapsed_layers) or 'multiple'}."
        ),
        contributing_findings=rank_findings + loss_findings,
        remediation=[
            "Increase the width (hidden dimension) of collapsed layers.",
            "Add regularization that promotes representation diversity "
            "(e.g. spectral normalization, VICReg-style variance term).",
            "Check for overly aggressive weight decay.",
            "Consider adding skip connections to preserve information flow.",
        ],
        references=[
            Reference(
                title="Understanding Dimensional Collapse in Contrastive Self-Supervised Learning",
                authors="Jing et al.",
                year=2022,
            ),
            Reference(
                title="VICReg: Variance-Invariance-Covariance Regularization "
                "for Self-Supervised Learning",
                authors="Bardes et al.",
                year=2022,
            ),
        ],
    )


register_correlation_rule(
    CorrelationRule(
        name="representation_bottleneck",
        description=(
            "Rank collapse with loss degradation -- the network has lost "
            "representational capacity, creating a bottleneck."
        ),
        required_categories=["rank_collapse", "loss_dynamics"],
        priority=85,
        condition=_condition,
        synthesize=_synthesize,
    )
)
