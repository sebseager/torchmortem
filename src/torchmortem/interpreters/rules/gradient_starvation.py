"""Correlation rule: Gradient Starvation.

Detects when vanishing/low gradients co-occur with dead units,
indicating that gradient starvation is the root cause of unit death.
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
    """Check that both gradient flow and dead unit findings are present."""
    has_gradient = any(
        f.category == "gradient_flow" and f.severity >= Severity.WARNING for f in findings
    )
    has_dead = any(f.category == "dead_units" and f.severity >= Severity.WARNING for f in findings)
    return has_gradient and has_dead


def _synthesize(findings: list[Finding]) -> Insight:
    """Produce a gradient starvation insight."""
    gradient_findings = [f for f in findings if f.category == "gradient_flow"]
    dead_findings = [f for f in findings if f.category == "dead_units"]

    dead_layers = []
    for f in dead_findings:
        dead_layers.extend(f.affected_layers)

    return Insight(
        rule_name="gradient_starvation",
        title="Gradient starvation causing unit death",
        summary=(
            "Vanishing gradients are starving downstream layers, leading to widespread dead units."
        ),
        detail=(
            "The gradient flow detector and dead unit detector both fired, "
            "suggesting a causal relationship: inadequate gradient signal is "
            "preventing neurons from ever activating, effectively killing them. "
            f"Affected layers: {', '.join(dead_layers) or 'multiple'}. "
            "This pattern is common in deep networks without skip connections "
            "or with poor initialization."
        ),
        contributing_findings=gradient_findings + dead_findings,
        remediation=[
            "Add skip/residual connections to improve gradient flow.",
            "Switch to an initialization scheme matched to your activation "
            "function (e.g. He init for ReLU).",
            "Consider using batch normalization or layer normalization.",
            "Reduce network depth or increase the learning rate slightly.",
        ],
        references=[
            Reference(
                title="Deep Residual Learning for Image Recognition",
                authors="He et al.",
                year=2016,
            ),
            Reference(
                title="Understanding the difficulty of training deep feedforward neural networks",
                authors="Glorot & Bengio",
                year=2010,
            ),
        ],
    )


register_correlation_rule(
    CorrelationRule(
        name="gradient_starvation",
        description=(
            "Vanishing gradients co-occurring with dead units -- gradient "
            "starvation is the likely root cause of unit death."
        ),
        required_categories=["gradient_flow", "dead_units"],
        priority=90,
        condition=_condition,
        synthesize=_synthesize,
    )
)
