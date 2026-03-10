"""RankCollapseDetector -- detects dimensional collapse in representations."""

from __future__ import annotations

import numpy as np

from torchmortem.registry import register_detector
from torchmortem.types import (
    CollectorState,
    Finding,
    Reference,
    RunMetadata,
    Severity,
)


@register_detector
class RankCollapseDetector:
    """Detects layers whose effective rank decreases significantly over training.

    Rank collapse indicates that the learned representations are collapsing
    to a lower-dimensional subspace, which wastes model capacity and can
    indicate problems like mode collapse or feature redundancy.

    Signals used: ``rank.effective_rank`` (per-layer effective rank over time).
    """

    name: str = "rank_collapse"
    required_collectors: list[str] = ["rank"]

    def __init__(
        self,
        collapse_ratio: float = 0.5,
        min_measurements: int = 3,
    ) -> None:
        self._collapse_ratio = collapse_ratio
        self._min_measurements = min_measurements

    def analyze(
        self,
        collector_states: dict[str, CollectorState],
        metadata: RunMetadata,
    ) -> list[Finding]:
        rank_state = collector_states["rank"]
        eff_rank = rank_state.series.get("effective_rank")
        steps = rank_state.steps
        layers = rank_state.layers

        if eff_rank is None or len(steps) < self._min_measurements or len(layers) == 0:
            return []

        findings: list[Finding] = []

        for layer_idx, layer_name in enumerate(layers):
            layer_rank = eff_rank[:, layer_idx]
            # Filter out zeros (failed SVD)
            valid = layer_rank > 0
            if valid.sum() < self._min_measurements:
                continue

            valid_ranks = layer_rank[valid]
            # Compare early vs late
            early = valid_ranks[: len(valid_ranks) // 3]
            late = valid_ranks[-len(valid_ranks) // 3 :]

            if len(early) == 0 or len(late) == 0:
                continue

            early_mean = float(early.mean())
            late_mean = float(late.mean())

            if early_mean < 1e-6:
                continue

            ratio = late_mean / early_mean
            if ratio > self._collapse_ratio:
                continue  # Not collapsed enough

            findings.append(
                Finding(
                    detector=self.name,
                    severity=Severity.WARNING if ratio > 0.3 else Severity.CRITICAL,
                    category="rank_collapse",
                    title=f"Rank collapse in {layer_name}",
                    summary=(
                        f"Effective rank in {layer_name} dropped from {early_mean:.1f} "
                        f"to {late_mean:.1f} ({ratio:.0%} of initial) -- representations "
                        f"are collapsing."
                    ),
                    detail=(
                        f"The effective rank (entropy-based) of layer '{layer_name}' "
                        f"decreased from {early_mean:.1f} in early training to "
                        f"{late_mean:.1f} late in training -- a {1 - ratio:.0%} "
                        f"reduction.\n\n"
                        f"Effective rank measures the 'dimensionality' of the layer's "
                        f"representations. A dropping rank means the layer is learning "
                        f"to map all inputs into a lower-dimensional subspace, wasting "
                        f"the remaining dimensions. This can happen due to:\n"
                        f"- Excessive weight decay or regularization squeezing out dimensions.\n"
                        f"- Learning rate too high causing features to align.\n"
                        f"- Architectural bottleneck forcing representations through a "
                        f"narrow subspace."
                    ),
                    evidence={
                        "early_rank": early_mean,
                        "late_rank": late_mean,
                        "collapse_ratio": ratio,
                    },
                    affected_layers=[layer_name],
                    step_range=(int(steps[0]), int(steps[-1])),
                    remediation=[
                        "Reduce regularization strength (weight decay, dropout) if representations are being over-constrained.",
                        "Add a rank regularization term or use spectral normalization to encourage full-rank representations.",
                        "Check if the layer width is appropriate -- a very wide layer may naturally have low effective rank.",
                        "Try a lower learning rate or different optimizer.",
                    ],
                    references=[
                        Reference(
                            title="Understanding Dimensional Collapse in Contrastive Self-Supervised Learning",
                            authors="Jing et al.",
                            year=2022,
                        ),
                    ],
                )
            )

        return findings
