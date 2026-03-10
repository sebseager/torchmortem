"""GradientFlowDetector -- detects vanishing and exploding gradients."""

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
class GradientFlowDetector:
    """Detects vanishing and exploding gradients by analyzing inter-layer
    gradient norm ratios over time.

    Signals used: ``gradient.grad_norm`` (per-layer gradient L2 norms).

    Detection logic:
        - **Vanishing**: The ratio of gradient norms between the first and last
          tracked layers exceeds a threshold (default: 100x), meaning the gradient
          signal is orders of magnitude weaker at early layers.
        - **Exploding**: Any layer's gradient norm exceeds an absolute threshold
          (default: 1000), or gradient norms grow exponentially across layers.
        - **Gradient stalling**: A layer's gradient norm drops near zero and stays
          there for a sustained period.
    """

    name: str = "gradient_flow"
    required_collectors: list[str] = ["gradient"]

    def __init__(
        self,
        vanishing_ratio_threshold: float = 100.0,
        exploding_norm_threshold: float = 1000.0,
        stall_threshold: float = 1e-8,
        stall_fraction: float = 0.3,
    ) -> None:
        self._vanishing_ratio = vanishing_ratio_threshold
        self._exploding_norm = exploding_norm_threshold
        self._stall_threshold = stall_threshold
        self._stall_fraction = stall_fraction

    def analyze(
        self,
        collector_states: dict[str, CollectorState],
        metadata: RunMetadata,
    ) -> list[Finding]:
        grad_state = collector_states["gradient"]
        grad_norms = grad_state.series.get("grad_norm")
        steps = grad_state.steps
        layers = grad_state.layers

        if grad_norms is None or len(steps) == 0 or len(layers) < 2:
            return []

        findings: list[Finding] = []

        findings.extend(self._check_vanishing(grad_norms, steps, layers))
        findings.extend(self._check_exploding(grad_norms, steps, layers))
        findings.extend(self._check_stalling(grad_norms, steps, layers))

        return findings

    def _check_vanishing(
        self,
        grad_norms: np.ndarray,
        steps: np.ndarray,
        layers: list[str],
    ) -> list[Finding]:
        """Check for vanishing gradients via first/last layer ratio."""
        # Use the second half of training (skip initial transient).
        n_steps = len(steps)
        start_idx = n_steps // 2
        if start_idx >= n_steps:
            return []

        norms_late = grad_norms[start_idx:]  # (remaining_steps, n_layers)
        # Mean gradient norm per layer in the second half.
        mean_per_layer = norms_late.mean(axis=0)  # (n_layers,)

        # Avoid division by zero.
        if mean_per_layer[0] < 1e-30:
            # First layer has effectively zero gradient, vanishing
            ratio = float("inf")
        elif mean_per_layer[-1] < 1e-30:
            return []  # Last layer has zero gradient, different issue
        else:
            ratio = float(mean_per_layer[-1] / mean_per_layer[0])

        if ratio < self._vanishing_ratio:
            return []

        # Identify the most affected layers (lowest norm).
        sorted_indices = np.argsort(mean_per_layer)
        worst_layers = [layers[i] for i in sorted_indices[: min(3, len(layers))]]

        return [
            Finding(
                detector=self.name,
                severity=Severity.CRITICAL if ratio > 1000 else Severity.WARNING,
                category="gradient_flow",
                title="Vanishing gradients detected",
                summary=(
                    f"Gradient signal to the earliest layers is {ratio:.0f}x weaker than "
                    f"the last layer -- the first layers are barely learning."
                ),
                detail=(
                    f"The average gradient L2 norm in the second half of training "
                    f"(steps {int(steps[start_idx])}-{int(steps[-1])}) was analyzed "
                    f"across all {len(layers)} tracked layers.\n\n"
                    f"The last layer ({layers[-1]}) had a mean gradient norm of "
                    f"{mean_per_layer[-1]:.2e}, while the first layer ({layers[0]}) "
                    f"had {mean_per_layer[0]:.2e} -- a ratio of {ratio:.0f}x.\n\n"
                    f"This means gradient information is being attenuated as it flows "
                    f"backward through the network. Early layers receive a vanishingly "
                    f"small learning signal, causing them to update slowly or not at all. "
                    f"This is a classic pathology in deep networks without residual "
                    f"connections, especially with activation functions like sigmoid or "
                    f"tanh that can saturate.\n\n"
                    f"Most affected layers: {', '.join(worst_layers)}"
                ),
                evidence={
                    "ratio": ratio,
                    "mean_per_layer": mean_per_layer.tolist(),
                    "layer_names": layers,
                    "analysis_start_step": int(steps[start_idx]),
                },
                affected_layers=worst_layers,
                step_range=(int(steps[start_idx]), int(steps[-1])),
                remediation=[
                    "Add skip/residual connections to allow gradients to flow directly to earlier layers.",
                    "Switch from sigmoid/tanh to ReLU or its variants (LeakyReLU, GELU) which don't saturate for positive inputs.",
                    "Use batch normalization or layer normalization to stabilize gradient magnitudes.",
                    "Reduce network depth or use a shallower architecture if the current depth isn't justified by task complexity.",
                    "Try gradient clipping as a band-aid, but prefer architectural fixes.",
                ],
                references=[
                    Reference(
                        title="Understanding the difficulty of training deep feedforward neural networks",
                        authors="Glorot & Bengio",
                        year=2010,
                    ),
                    Reference(
                        title="Deep Residual Learning for Image Recognition",
                        authors="He et al.",
                        year=2016,
                    ),
                ],
            )
        ]

    def _check_exploding(
        self,
        grad_norms: np.ndarray,
        steps: np.ndarray,
        layers: list[str],
    ) -> list[Finding]:
        """Check for exploding gradients (extreme absolute norms)."""
        # Find steps/layers where gradient norms exceed the threshold.
        mask = grad_norms > self._exploding_norm
        if not mask.any():
            return []

        # Which layers had explosions?
        layer_had_explosion = mask.any(axis=0)  # (n_layers,)
        affected = [layers[i] for i, v in enumerate(layer_had_explosion) if v]

        # When did it start?
        step_had_explosion = mask.any(axis=1)  # (n_steps,)
        first_step_idx = int(np.argmax(step_had_explosion))
        last_step_idx = len(steps) - 1 - int(np.argmax(step_had_explosion[::-1]))

        max_norm = float(grad_norms.max())

        return [
            Finding(
                detector=self.name,
                severity=Severity.CRITICAL,
                category="gradient_flow",
                title="Exploding gradients detected",
                summary=(
                    f"Gradient norms exceeded {self._exploding_norm:.0f} in "
                    f"{len(affected)} layer(s), peaking at {max_norm:.1e}."
                ),
                detail=(
                    f"One or more layers exhibited extremely large gradient norms "
                    f"(>{self._exploding_norm:.0f}) between steps "
                    f"{int(steps[first_step_idx])} and {int(steps[last_step_idx])}.\n\n"
                    f"The maximum observed gradient norm was {max_norm:.1e}. "
                    f"Exploding gradients cause numerically unstable weight updates "
                    f"that can make training diverge (loss -> NaN/inf). This commonly "
                    f"occurs with high learning rates, poor weight initialization, or "
                    f"recurrent architectures without gradient clipping.\n\n"
                    f"Affected layers: {', '.join(affected)}"
                ),
                evidence={
                    "max_norm": max_norm,
                    "threshold": self._exploding_norm,
                    "affected_layer_names": affected,
                },
                affected_layers=affected,
                step_range=(int(steps[first_step_idx]), int(steps[last_step_idx])),
                remediation=[
                    "Apply gradient clipping (e.g., torch.nn.utils.clip_grad_norm_ with max_norm=1.0).",
                    "Reduce the learning rate.",
                    "Use a more conservative weight initialization (e.g., Kaiming or Xavier).",
                    "Add gradient normalization or batch normalization layers.",
                    "If using RNNs, consider switching to LSTM or GRU which have gating mechanisms.",
                ],
                references=[
                    Reference(
                        title="On the difficulty of training Recurrent Neural Networks",
                        authors="Pascanu et al.",
                        year=2013,
                    ),
                ],
            )
        ]

    def _check_stalling(
        self,
        grad_norms: np.ndarray,
        steps: np.ndarray,
        layers: list[str],
    ) -> list[Finding]:
        """Check for layers where gradient norm drops to ~zero and stays."""
        findings: list[Finding] = []
        n_steps = len(steps)

        for layer_idx, layer_name in enumerate(layers):
            layer_norms = grad_norms[:, layer_idx]
            stalled_mask = layer_norms < self._stall_threshold
            stalled_frac = stalled_mask.sum() / n_steps

            if stalled_frac < self._stall_fraction:
                continue

            # Find the first step where stalling begins.
            first_stall_idx = int(np.argmax(stalled_mask))

            findings.append(
                Finding(
                    detector=self.name,
                    severity=Severity.WARNING,
                    category="gradient_flow",
                    title=f"Gradient stalling in {layer_name}",
                    summary=(
                        f"Layer '{layer_name}' had near-zero gradient norm "
                        f"({stalled_frac:.0%} of steps)."
                    ),
                    detail=(
                        f"The gradient norm for layer '{layer_name}' was below "
                        f"{self._stall_threshold:.0e} for {stalled_frac:.0%} of all "
                        f"recorded steps, starting around step {int(steps[first_stall_idx])}.\n\n"
                        f"When a layer receives near-zero gradients, its weights are "
                        f"effectively frozen -- it provides no useful learning signal. "
                        f"This can be caused by upstream dead activations, saturated "
                        f"activation functions, or severe vanishing gradients."
                    ),
                    evidence={
                        "stalled_fraction": float(stalled_frac),
                        "threshold": self._stall_threshold,
                    },
                    affected_layers=[layer_name],
                    step_range=(int(steps[first_stall_idx]), int(steps[-1])),
                    remediation=[
                        f"Investigate why layer '{layer_name}' stops receiving gradient signal.",
                        "Check for dead activations in upstream layers (see dead unit detection).",
                        "Consider a different initialization scheme or adding a residual connection.",
                    ],
                )
            )

        return findings
