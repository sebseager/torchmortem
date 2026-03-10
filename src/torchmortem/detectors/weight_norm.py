"""WeightNormDetector -- detects weight norm explosion, stagnation, and layer imbalance."""

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
class WeightNormDetector:
    """Detects unhealthy weight norm dynamics: explosion, stagnation,
    and inter-layer imbalance.

    Signals used: ``weight.weight_norm`` (per-layer weight L2 norms over time).
    """

    name: str = "weight_norm"
    required_collectors: list[str] = ["weight"]

    def __init__(
        self,
        explosion_factor: float = 10.0,
        stagnation_threshold: float = 1e-4,
        imbalance_factor: float = 100.0,
    ) -> None:
        self._explosion_factor = explosion_factor
        self._stagnation_threshold = stagnation_threshold
        self._imbalance_factor = imbalance_factor

    def analyze(
        self,
        collector_states: dict[str, CollectorState],
        metadata: RunMetadata,
    ) -> list[Finding]:
        weight_state = collector_states["weight"]
        w_norms = weight_state.series.get("weight_norm")
        steps = weight_state.steps
        layers = weight_state.layers

        if w_norms is None or len(steps) < 2 or len(layers) == 0:
            return []

        findings: list[Finding] = []
        findings.extend(self._check_explosion(w_norms, steps, layers))
        findings.extend(self._check_stagnation(w_norms, steps, layers))
        findings.extend(self._check_imbalance(w_norms, steps, layers))
        return findings

    def _check_explosion(
        self,
        w_norms: np.ndarray,
        steps: np.ndarray,
        layers: list[str],
    ) -> list[Finding]:
        """Check if any layer's weight norm grew by more than explosion_factor."""
        initial = w_norms[0]  # (n_layers,)
        final = w_norms[-1]

        affected = []
        ratios = []
        for i, layer in enumerate(layers):
            if initial[i] < 1e-30:
                continue
            ratio = final[i] / initial[i]
            if ratio > self._explosion_factor:
                affected.append(layer)
                ratios.append(ratio)

        if not affected:
            return []

        max_ratio = max(ratios)
        return [
            Finding(
                detector=self.name,
                severity=Severity.CRITICAL if max_ratio > 100 else Severity.WARNING,
                category="weight_norm",
                title="Weight norm explosion",
                summary=(
                    f"Weight norms grew by up to {max_ratio:.0f}x in {len(affected)} "
                    f"layer(s) during training."
                ),
                detail=(
                    f"The following layers saw their weight L2 norm grow by more than "
                    f"{self._explosion_factor:.0f}x from initialization to the final "
                    f"step:\n\n"
                    + "\n".join(
                        f"  - {layer}: {ratios[i]:.1f}x" for i, layer in enumerate(affected)
                    )
                    + "\n\nRapid weight norm growth can indicate training instability. "
                    "Weights are being pushed to ever-larger magnitudes, which can "
                    "cause numerical overflow, saturated activations, and eventual "
                    "divergence."
                ),
                evidence={
                    "affected_layers": affected,
                    "growth_ratios": ratios,
                    "explosion_factor": self._explosion_factor,
                },
                affected_layers=affected,
                step_range=(int(steps[0]), int(steps[-1])),
                remediation=[
                    "Add weight decay (L2 regularization) to the optimizer to penalize large weights.",
                    "Reduce the learning rate.",
                    "Use gradient clipping to prevent large updates.",
                    "Check for exploding gradients -- weight explosion is often a downstream effect.",
                ],
                references=[
                    Reference(
                        title="Decoupled Weight Decay Regularization",
                        authors="Loshchilov & Hutter",
                        year=2019,
                    ),
                ],
            )
        ]

    def _check_stagnation(
        self,
        w_norms: np.ndarray,
        steps: np.ndarray,
        layers: list[str],
    ) -> list[Finding]:
        """Check if any layer's weights barely changed during training."""
        if len(steps) < 4:
            return []

        initial = w_norms[0]
        final = w_norms[-1]

        affected = []
        for i, layer in enumerate(layers):
            norm_i = initial[i]
            if norm_i < 1e-30:
                continue
            relative_change = abs(final[i] - norm_i) / norm_i
            if relative_change < self._stagnation_threshold:
                affected.append(layer)

        if not affected:
            return []

        return [
            Finding(
                detector=self.name,
                severity=Severity.WARNING,
                category="weight_norm",
                title="Stagnant weight norms",
                summary=(
                    f"{len(affected)} layer(s) had weight norms that barely changed "
                    f"during training -- these layers may not be learning."
                ),
                detail=(
                    f"The following layers had less than "
                    f"{self._stagnation_threshold:.0%} relative change in weight norm "
                    f"over the entire training run:\n\n"
                    + "\n".join(f"  - {layer}" for layer in affected)
                    + "\n\nWhile some layers (especially early ones in well-initialized "
                    "networks) may not need large weight changes, complete stagnation "
                    "can indicate that the layer is not receiving sufficient gradient "
                    "signal."
                ),
                evidence={
                    "affected_layers": affected,
                    "stagnation_threshold": self._stagnation_threshold,
                },
                affected_layers=affected,
                step_range=(int(steps[0]), int(steps[-1])),
                remediation=[
                    "Check for vanishing gradients in earlier layers.",
                    "Increase the learning rate or use layer-wise learning rate scaling.",
                    "Verify that the layer is actually connected in the computation graph.",
                ],
            )
        ]

    def _check_imbalance(
        self,
        w_norms: np.ndarray,
        steps: np.ndarray,
        layers: list[str],
    ) -> list[Finding]:
        """Check for large inter-layer weight norm disparity."""
        if len(layers) < 2:
            return []

        # Use final step norms
        final_norms = w_norms[-1]
        nonzero = final_norms[final_norms > 1e-30]
        if len(nonzero) < 2:
            return []

        ratio = nonzero.max() / nonzero.min()
        if ratio < self._imbalance_factor:
            return []

        # Identify the extremes
        max_idx = int(np.argmax(final_norms))
        min_idx = int(np.argmin(final_norms[final_norms > 1e-30]))
        # Remap min_idx to the actual index in the layers list
        nonzero_indices = np.where(final_norms > 1e-30)[0]
        min_idx = int(nonzero_indices[np.argmin(final_norms[nonzero_indices])])

        return [
            Finding(
                detector=self.name,
                severity=Severity.WARNING,
                category="weight_norm",
                title="Inter-layer weight norm imbalance",
                summary=(
                    f"Weight norms vary by {ratio:.0f}x between the largest "
                    f"({layers[max_idx]}) and smallest ({layers[min_idx]}) layers."
                ),
                detail=(
                    f"At the end of training, the weight norm of '{layers[max_idx]}' "
                    f"({final_norms[max_idx]:.2e}) is {ratio:.0f}x larger than "
                    f"'{layers[min_idx]}' ({final_norms[min_idx]:.2e}).\n\n"
                    f"Large inter-layer norm disparity can cause optimization "
                    f"difficulties -- the optimizer's single learning rate may be "
                    f"too large for high-norm layers and too small for low-norm layers."
                ),
                evidence={
                    "norm_ratio": float(ratio),
                    "max_layer": layers[max_idx],
                    "max_norm": float(final_norms[max_idx]),
                    "min_layer": layers[min_idx],
                    "min_norm": float(final_norms[min_idx]),
                },
                affected_layers=[layers[max_idx], layers[min_idx]],
                step_range=(int(steps[0]), int(steps[-1])),
                remediation=[
                    "Use per-layer learning rate scaling or an adaptive optimizer (Adam/AdamW) which normalizes by gradient magnitude.",
                    "Apply weight normalization or spectral normalization to constrain weight norms.",
                    "Check weight initialization -- different layers may need different init scales.",
                ],
            )
        ]
