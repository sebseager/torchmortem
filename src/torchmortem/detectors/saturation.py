"""SaturationDetector -- detects layers with saturated activation functions."""

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
class SaturationDetector:
    """Detects layers where sigmoid/tanh-like activations are saturated.

    Saturated activations (values near ±1 for tanh, or near 0/1 for sigmoid)
    have near-zero gradients, causing vanishing gradient flow to earlier layers.

    Signals used: ``activation.act_saturated_frac`` (per-layer fraction of
    activations near ±1).
    """

    name: str = "saturation"
    required_collectors: list[str] = ["activation"]

    def __init__(
        self,
        saturated_frac_threshold: float = 0.5,
        sustained_fraction: float = 0.5,
    ) -> None:
        self._saturated_frac_threshold = saturated_frac_threshold
        self._sustained_fraction = sustained_fraction

    def analyze(
        self,
        collector_states: dict[str, CollectorState],
        metadata: RunMetadata,
    ) -> list[Finding]:
        act_state = collector_states["activation"]
        sat_frac = act_state.series.get("act_saturated_frac")
        steps = act_state.steps
        layers = act_state.layers

        if sat_frac is None or len(steps) == 0 or len(layers) == 0:
            return []

        findings: list[Finding] = []
        n_steps = len(steps)
        start_idx = n_steps // 2
        if start_idx >= n_steps:
            return []

        late_sat = sat_frac[start_idx:]

        for layer_idx, layer_name in enumerate(layers):
            layer_sat = late_sat[:, layer_idx]
            bad_steps = (layer_sat > self._saturated_frac_threshold).sum()
            bad_ratio = bad_steps / len(layer_sat)

            if bad_ratio < self._sustained_fraction:
                continue

            mean_sat = float(layer_sat.mean())
            max_sat = float(layer_sat.max())

            severity = Severity.CRITICAL if mean_sat > 0.8 else Severity.WARNING

            findings.append(
                Finding(
                    detector=self.name,
                    severity=severity,
                    category="saturation",
                    title=f"Activation saturation in {layer_name}",
                    summary=(
                        f"{mean_sat:.0%} of activations in {layer_name} are saturated "
                        f"(near extreme values) -- gradients through this layer are "
                        f"nearly zero."
                    ),
                    detail=(
                        f"Layer '{layer_name}' has an average of {mean_sat:.1%} "
                        f"saturated activations (peak: {max_sat:.1%}) in the second "
                        f"half of training (steps {int(steps[start_idx])}-"
                        f"{int(steps[-1])}).\n\n"
                        f"When sigmoid or tanh activations saturate (output near 0/1 "
                        f"or ±1), their derivatives approach zero. This effectively "
                        f"blocks gradient flow through the saturated layer, starving "
                        f"all preceding layers of learning signal. This is a major "
                        f"cause of vanishing gradients in networks using bounded "
                        f"activation functions.\n\n"
                        f"The saturation fraction exceeded {self._saturated_frac_threshold:.0%} "
                        f"in {bad_ratio:.0%} of the analyzed steps."
                    ),
                    evidence={
                        "mean_saturated_frac": mean_sat,
                        "max_saturated_frac": max_sat,
                        "bad_step_ratio": float(bad_ratio),
                    },
                    affected_layers=[layer_name],
                    step_range=(int(steps[start_idx]), int(steps[-1])),
                    remediation=[
                        "Replace sigmoid/tanh with ReLU, GELU, or SiLU -- unbounded activations that don't saturate for positive inputs.",
                        "If sigmoid/tanh is required (e.g. output layer), add batch normalization or layer normalization before the activation to keep inputs centered.",
                        "Reduce the learning rate -- large weight updates can push activations deeper into saturation.",
                        "Use Xavier/Glorot initialization, which is designed to keep activations in the linear regime of sigmoid/tanh.",
                    ],
                    references=[
                        Reference(
                            title="Understanding the difficulty of training deep feedforward neural networks",
                            authors="Glorot & Bengio",
                            year=2010,
                        ),
                    ],
                )
            )

        return findings
