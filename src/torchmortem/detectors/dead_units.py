"""DeadUnitDetector -- detects layers with persistently inactive (dead) units."""

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
class DeadUnitDetector:
    """Detects layers where a large fraction of units are persistently dead.

    A "dead unit" has near-zero activation across all inputs. This commonly
    occurs with ReLU activations when weights push pre-activations into the
    negative regime (the "dying ReLU" problem).

    Signals used: ``activation.act_dead_frac`` (per-layer fraction of dead units).
    """

    name: str = "dead_units"
    required_collectors: list[str] = ["activation"]

    def __init__(
        self,
        dead_frac_threshold: float = 0.5,
        sustained_fraction: float = 0.5,
    ) -> None:
        self._dead_frac_threshold = dead_frac_threshold
        self._sustained_fraction = sustained_fraction

    def analyze(
        self,
        collector_states: dict[str, CollectorState],
        metadata: RunMetadata,
    ) -> list[Finding]:
        act_state = collector_states["activation"]
        dead_frac = act_state.series.get("act_dead_frac")
        steps = act_state.steps
        layers = act_state.layers

        if dead_frac is None or len(steps) == 0 or len(layers) == 0:
            return []

        findings: list[Finding] = []
        n_steps = len(steps)

        # Use second half of training to avoid flagging initial transients
        start_idx = n_steps // 2
        if start_idx >= n_steps:
            return []

        late_dead = dead_frac[start_idx:]  # (remaining, n_layers)

        for layer_idx, layer_name in enumerate(layers):
            layer_dead = late_dead[:, layer_idx]
            # Fraction of steps where dead frac exceeds threshold
            bad_steps = (layer_dead > self._dead_frac_threshold).sum()
            bad_ratio = bad_steps / len(layer_dead)

            if bad_ratio < self._sustained_fraction:
                continue

            mean_dead = float(layer_dead.mean())
            max_dead = float(layer_dead.max())

            severity = Severity.CRITICAL if mean_dead > 0.8 else Severity.WARNING

            findings.append(
                Finding(
                    detector=self.name,
                    severity=severity,
                    category="dead_units",
                    title=f"Dead units in {layer_name}",
                    summary=(
                        f"{mean_dead:.0%} of units in {layer_name} are persistently "
                        f"dead (near-zero activation) in the second half of training."
                    ),
                    detail=(
                        f"Layer '{layer_name}' has an average of {mean_dead:.1%} dead "
                        f"units (peak: {max_dead:.1%}) across the second half of "
                        f"training (steps {int(steps[start_idx])}-{int(steps[-1])}).\n\n"
                        f"Dead units produce zero or near-zero activations regardless "
                        f"of the input. In ReLU networks, this happens when a unit's "
                        f"pre-activation is always negative. These units contribute no "
                        f"useful features and waste model capacity.\n\n"
                        f"The dead unit fraction exceeded {self._dead_frac_threshold:.0%} "
                        f"in {bad_ratio:.0%} of the analyzed steps, indicating this is "
                        f"a persistent problem rather than a transient fluctuation."
                    ),
                    evidence={
                        "mean_dead_frac": mean_dead,
                        "max_dead_frac": max_dead,
                        "bad_step_ratio": float(bad_ratio),
                    },
                    affected_layers=[layer_name],
                    step_range=(int(steps[start_idx]), int(steps[-1])),
                    remediation=[
                        "Switch from ReLU to LeakyReLU, ELU, or GELU -- these allow small negative gradients that prevent units from dying permanently.",
                        "Reduce the learning rate -- large updates can push weights into the dead regime.",
                        "Use careful weight initialization (He/Kaiming init for ReLU networks).",
                        "Add batch normalization before activation layers to keep pre-activations centered.",
                    ],
                    references=[
                        Reference(
                            title="Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification",
                            authors="He et al.",
                            year=2015,
                        ),
                        Reference(
                            title="Rectified Linear Units Improve Restricted Boltzmann Machines",
                            authors="Nair & Hinton",
                            year=2010,
                        ),
                    ],
                )
            )

        return findings
