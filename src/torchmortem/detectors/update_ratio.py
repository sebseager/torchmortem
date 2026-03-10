"""UpdateRatioDetector -- flags unhealthy update/weight ratios."""

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
class UpdateRatioDetector:
    """Detects layers where the update/weight ratio deviates from the
    healthy ~1e-3 range.

    A well-known heuristic (Karpathy): the ratio ||update|| / ||weight||
    should be roughly 1e-3. Ratios much larger indicate the learning rate
    is too high (weights change too fast); much smaller means the layer is
    barely learning.

    Signals used: ``weight.update_ratio`` (per-layer update/weight ratio).
    """

    name: str = "update_ratio"
    required_collectors: list[str] = ["weight"]

    def __init__(
        self,
        low_threshold: float = 1e-5,
        high_threshold: float = 1e-1,
    ) -> None:
        self._low_threshold = low_threshold
        self._high_threshold = high_threshold

    def analyze(
        self,
        collector_states: dict[str, CollectorState],
        metadata: RunMetadata,
    ) -> list[Finding]:
        weight_state = collector_states["weight"]
        ratios = weight_state.series.get("update_ratio")
        steps = weight_state.steps
        layers = weight_state.layers

        if ratios is None or len(steps) < 2 or len(layers) == 0:
            return []

        # Skip step 0 (no meaningful update yet)
        ratios = ratios[1:]
        analysis_steps = steps[1:]

        if len(analysis_steps) == 0:
            return []

        findings: list[Finding] = []

        # Use second half
        start_idx = len(analysis_steps) // 2
        if start_idx >= len(analysis_steps):
            start_idx = 0

        late_ratios = ratios[start_idx:]

        for layer_idx, layer_name in enumerate(layers):
            layer_ratios = late_ratios[:, layer_idx]
            # Filter out exact zeros (could be from initialization)
            nonzero = layer_ratios[layer_ratios > 0]
            if len(nonzero) == 0:
                continue

            median_ratio = float(np.median(nonzero))

            if median_ratio < self._low_threshold:
                findings.append(
                    Finding(
                        detector=self.name,
                        severity=Severity.WARNING,
                        category="update_ratio",
                        title=f"Update ratio too low in {layer_name}",
                        summary=(
                            f"Median update/weight ratio in {layer_name} is "
                            f"{median_ratio:.1e} -- this layer is barely learning."
                        ),
                        detail=(
                            f"The ratio ||weight_update|| / ||weight|| for layer "
                            f"'{layer_name}' has a median of {median_ratio:.2e} in the "
                            f"second half of training. The healthy range is roughly "
                            f"1e-4 to 1e-2 (ideally ~1e-3).\n\n"
                            f"A very low ratio means the optimizer's updates are "
                            f"negligible compared to the weight magnitudes -- the layer "
                            f"is effectively frozen. This can happen due to vanishing "
                            f"gradients, an overly small learning rate, or the layer "
                            f"being overshadowed by other layers in a shared optimizer."
                        ),
                        evidence={
                            "median_ratio": median_ratio,
                            "low_threshold": self._low_threshold,
                            "high_threshold": self._high_threshold,
                        },
                        affected_layers=[layer_name],
                        step_range=(
                            int(analysis_steps[start_idx]),
                            int(analysis_steps[-1]),
                        ),
                        remediation=[
                            "Increase the learning rate, or use per-parameter learning rate groups to boost this layer.",
                            "Check for vanishing gradients -- the low update ratio may be a symptom, not the root cause.",
                            "Consider using an adaptive optimizer (Adam, AdamW) if using SGD.",
                        ],
                        references=[
                            Reference(
                                title="A Recipe for Training Neural Networks (blog post)",
                                authors="Andrej Karpathy",
                                year=2019,
                            ),
                        ],
                    )
                )
            elif median_ratio > self._high_threshold:
                findings.append(
                    Finding(
                        detector=self.name,
                        severity=Severity.CRITICAL,
                        category="update_ratio",
                        title=f"Update ratio too high in {layer_name}",
                        summary=(
                            f"Median update/weight ratio in {layer_name} is "
                            f"{median_ratio:.1e} -- weights are changing too aggressively."
                        ),
                        detail=(
                            f"The ratio ||weight_update|| / ||weight|| for layer "
                            f"'{layer_name}' has a median of {median_ratio:.2e}. "
                            f"This is far above the healthy ~1e-3 target.\n\n"
                            f"Large update ratios mean the optimizer is making dramatic "
                            f"changes to the weights each step. This can cause training "
                            f"instability, oscillation, or divergence. The weights may "
                            f"be overshooting good solutions."
                        ),
                        evidence={
                            "median_ratio": median_ratio,
                            "low_threshold": self._low_threshold,
                            "high_threshold": self._high_threshold,
                        },
                        affected_layers=[layer_name],
                        step_range=(
                            int(analysis_steps[start_idx]),
                            int(analysis_steps[-1]),
                        ),
                        remediation=[
                            "Reduce the learning rate -- this is the most common fix.",
                            "Add gradient clipping (e.g. torch.nn.utils.clip_grad_norm_) to limit update magnitude.",
                            "Use learning rate warmup to start with smaller updates.",
                            "Switch to an adaptive optimizer (Adam/AdamW) which auto-scales per-parameter update magnitudes.",
                        ],
                        references=[
                            Reference(
                                title="A Recipe for Training Neural Networks (blog post)",
                                authors="Andrej Karpathy",
                                year=2019,
                            ),
                        ],
                    )
                )

        return findings
