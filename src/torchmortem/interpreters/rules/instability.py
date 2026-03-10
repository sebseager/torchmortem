"""Correlation rule: Instability Feedback Loop.

Detects when exploding gradients co-occur with weight norm explosion,
indicating a positive feedback loop driving training instability.
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
    """Check for both exploding gradients and weight norm issues."""
    has_exploding_grads = any(
        f.category == "gradient_flow"
        and "exploding" in f.title.lower()
        and f.severity >= Severity.WARNING
        for f in findings
    )
    has_weight_explosion = any(
        f.category == "weight_norm"
        and "explosion" in f.title.lower()
        and f.severity >= Severity.WARNING
        for f in findings
    )
    return has_exploding_grads and has_weight_explosion


def _synthesize(findings: list[Finding]) -> Insight:
    """Produce an instability feedback loop insight."""
    grad_findings = [
        f for f in findings if f.category == "gradient_flow" and "exploding" in f.title.lower()
    ]
    weight_findings = [
        f for f in findings if f.category == "weight_norm" and "explosion" in f.title.lower()
    ]

    return Insight(
        rule_name="instability_feedback_loop",
        title="Instability feedback loop -- gradients and weights exploding",
        summary=(
            "Exploding gradients and weight norm explosion are reinforcing "
            "each other in a positive feedback loop."
        ),
        detail=(
            "The gradient flow detector found exploding gradients while the "
            "weight norm detector found rapid weight growth. These two issues "
            "form a feedback loop: large weights produce large activations and "
            "gradients, which in turn produce even larger weight updates. "
            "Without intervention, this will lead to NaN values and training "
            "collapse."
        ),
        contributing_findings=grad_findings + weight_findings,
        remediation=[
            "Apply gradient clipping (e.g. clip_grad_norm_ with max_norm=1.0).",
            "Reduce the learning rate significantly.",
            "Add weight decay to counteract weight growth.",
            "Check for missing normalization layers (BatchNorm, LayerNorm).",
            "Verify weight initialization is appropriate for the architecture.",
        ],
        references=[
            Reference(
                title="On the difficulty of training Recurrent Neural Networks",
                authors="Pascanu et al.",
                year=2013,
            ),
            Reference(
                title="Decoupled Weight Decay Regularization",
                authors="Loshchilov & Hutter",
                year=2019,
            ),
        ],
    )


register_correlation_rule(
    CorrelationRule(
        name="instability_feedback_loop",
        description=(
            "Exploding gradients plus weight norm explosion -- positive "
            "feedback loop driving training instability."
        ),
        required_categories=["gradient_flow", "weight_norm"],
        priority=95,
        condition=_condition,
        synthesize=_synthesize,
    )
)
