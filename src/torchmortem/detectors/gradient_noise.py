"""GradientNoiseDetector -- estimates gradient signal-to-noise ratio."""

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
class GradientNoiseDetector:
    """Estimates gradient signal-to-noise ratio from inter-step gradient variance.

    Compares the magnitude of gradient changes between consecutive steps
    (noise) to the average gradient magnitude (signal). A low SNR suggests
    the batch size may be too small, resulting in noisy gradient estimates
    that slow convergence.

    Signals used: ``gradient.grad_norm`` (per-layer gradient norms over time).
    """

    name: str = "gradient_noise"
    required_collectors: list[str] = ["gradient"]

    def __init__(
        self,
        low_snr_threshold: float = 1.0,
        min_steps: int = 20,
    ) -> None:
        self._low_snr_threshold = low_snr_threshold
        self._min_steps = min_steps

    def analyze(
        self,
        collector_states: dict[str, CollectorState],
        metadata: RunMetadata,
    ) -> list[Finding]:
        grad_state = collector_states["gradient"]
        grad_norms = grad_state.series.get("grad_norm")
        steps = grad_state.steps
        layers = grad_state.layers

        if grad_norms is None or len(steps) < self._min_steps or len(layers) == 0:
            return []

        # Compute per-layer SNR: mean(norm) / std(norm)
        # Use second half to avoid init transients
        start_idx = len(steps) // 2
        if start_idx < 5:
            start_idx = 0

        late_norms = grad_norms[start_idx:]
        if len(late_norms) < 5:
            return []

        # Average SNR across layers
        snrs_per_layer = []
        low_snr_layers = []
        for layer_idx, layer_name in enumerate(layers):
            layer_norms = late_norms[:, layer_idx]
            mean_norm = layer_norms.mean()
            std_norm = layer_norms.std()

            if mean_norm < 1e-30 or std_norm < 1e-30:
                continue

            snr = mean_norm / std_norm
            snrs_per_layer.append(snr)
            if snr < self._low_snr_threshold:
                low_snr_layers.append((layer_name, float(snr)))

        if not snrs_per_layer:
            return []

        overall_snr = float(np.mean(snrs_per_layer))

        findings: list[Finding] = []

        if overall_snr < self._low_snr_threshold:
            worst = sorted(low_snr_layers, key=lambda x: x[1])[:3]
            affected = [name for name, _ in worst]

            findings.append(
                Finding(
                    detector=self.name,
                    severity=Severity.WARNING,
                    category="gradient_noise",
                    title="High gradient noise (low SNR)",
                    summary=(
                        f"Average gradient signal-to-noise ratio is {overall_snr:.2f} "
                        f"(threshold: {self._low_snr_threshold:.1f}) -- gradient estimates "
                        f"are very noisy."
                    ),
                    detail=(
                        f"The gradient signal-to-noise ratio (mean_norm / std_norm) "
                        f"averaged across layers is {overall_snr:.2f}, below the "
                        f"threshold of {self._low_snr_threshold:.1f}.\n\n"
                        f"Worst layers:\n"
                        + "\n".join(f"  - {n}: SNR = {s:.2f}" for n, s in worst)
                        + f"\n\nA low SNR means most of the gradient signal is noise "
                        f"from mini-batch sampling. The optimizer is spending most of "
                        f"its updates fighting noise rather than making progress. This "
                        f"is common with very small batch sizes or high-variance loss "
                        f"functions."
                    ),
                    evidence={
                        "overall_snr": overall_snr,
                        "per_layer_snr": {name: snr for name, snr in low_snr_layers},
                    },
                    affected_layers=affected,
                    step_range=(int(steps[start_idx]), int(steps[-1])),
                    remediation=[
                        "Increase the batch size -- the most direct way to reduce gradient noise.",
                        "Use gradient accumulation if GPU memory is limited.",
                        "Add gradient smoothing (e.g., increase optimizer momentum).",
                        "Consider using a lower learning rate to compensate for noisy gradients.",
                    ],
                    references=[
                        Reference(
                            title="An Empirical Model of Large-Batch Training",
                            authors="McCandlish et al.",
                            year=2018,
                        ),
                    ],
                )
            )

        return findings
