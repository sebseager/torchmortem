"""LossDynamicsDetector -- detects catapult phase, edge-of-stability, plateaus, and divergence."""

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
class LossDynamicsDetector:
    """Detects anomalous loss trajectory patterns.

    - **Catapult phase**: Sharp loss spike in early training followed by rapid recovery.
    - **Edge of stability**: Sharpness (top Hessian eigenvalue) hovering near 2/lr.
    - **Loss plateau**: Prolonged period of near-zero loss improvement.
    - **Divergence**: Loss increasing monotonically.

    Signals used: ``loss.loss``, ``loss.loss_smoothed``, ``curvature.top_eigenvalue``.
    """

    name: str = "loss_dynamics"
    required_collectors: list[str] = ["loss"]

    def __init__(
        self,
        plateau_window: int = 20,
        plateau_threshold: float = 1e-5,
        catapult_spike_factor: float = 2.0,
        divergence_window: int = 20,
    ) -> None:
        self._plateau_window = plateau_window
        self._plateau_threshold = plateau_threshold
        self._catapult_spike_factor = catapult_spike_factor
        self._divergence_window = divergence_window

    def analyze(
        self,
        collector_states: dict[str, CollectorState],
        metadata: RunMetadata,
    ) -> list[Finding]:
        loss_state = collector_states["loss"]
        loss_vals = loss_state.series.get("loss")
        smoothed = loss_state.series.get("loss_smoothed")
        steps = loss_state.steps

        if loss_vals is None or len(steps) < 10:
            return []

        findings: list[Finding] = []
        findings.extend(self._check_divergence(loss_vals, smoothed, steps))
        findings.extend(
            self._check_plateau(smoothed if smoothed is not None else loss_vals, steps)
        )
        findings.extend(self._check_catapult(loss_vals, smoothed, steps))
        findings.extend(self._check_edge_of_stability(collector_states, metadata))
        return findings

    def _check_divergence(
        self,
        loss_vals: np.ndarray,
        smoothed: np.ndarray | None,
        steps: np.ndarray,
    ) -> list[Finding]:
        """Check if loss is monotonically increasing in the tail."""
        series = smoothed if smoothed is not None else loss_vals
        n = len(series)
        window = min(self._divergence_window, n // 2)
        if window < 5:
            return []

        tail = series[-window:]
        # Check if the tail is predominantly increasing
        diffs = np.diff(tail)
        increasing_frac = (diffs > 0).sum() / len(diffs)

        if increasing_frac < 0.8:
            return []

        # Check magnitude of increase
        start_val = float(tail[0])
        end_val = float(tail[-1])
        if start_val > 1e-30 and end_val / start_val < 1.5:
            return []

        return [
            Finding(
                detector=self.name,
                severity=Severity.CRITICAL,
                category="loss_dynamics",
                title="Training divergence detected",
                summary=(
                    f"Loss increased from {start_val:.3e} to {end_val:.3e} "
                    f"over the last {window} steps -- training is diverging."
                ),
                detail=(
                    f"The smoothed loss was increasing in {increasing_frac:.0%} of the "
                    f"last {window} steps. The loss grew from {start_val:.3e} to "
                    f"{end_val:.3e}, a {end_val / start_val:.1f}x increase.\n\n"
                    f"Divergence means the model is getting worse, not better. This "
                    f"typically indicates the learning rate is too high, causing the "
                    f"optimizer to overshoot minima repeatedly. It can also result from "
                    f"numerical instability (NaN/inf in weights or gradients)."
                ),
                evidence={
                    "start_loss": start_val,
                    "end_loss": end_val,
                    "increasing_fraction": float(increasing_frac),
                },
                step_range=(int(steps[-window]), int(steps[-1])),
                remediation=[
                    "Reduce the learning rate -- this is the most common fix.",
                    "Add gradient clipping to prevent explosive updates.",
                    "Check for NaN/inf in loss values.",
                    "Use learning rate warmup to start training gently.",
                ],
            )
        ]

    def _check_plateau(
        self,
        series: np.ndarray,
        steps: np.ndarray,
    ) -> list[Finding]:
        """Check for prolonged loss plateaus."""
        n = len(series)
        window = self._plateau_window
        if n < window * 2:
            return []

        # Compute rolling relative change
        findings: list[Finding] = []
        # Check the second half for plateaus
        start = n // 2
        for i in range(start, n - window, window):
            segment = series[i : i + window]
            mean_val = segment.mean()
            if mean_val < 1e-30:
                continue
            relative_range = (segment.max() - segment.min()) / abs(mean_val)
            if relative_range < self._plateau_threshold:
                return [
                    Finding(
                        detector=self.name,
                        severity=Severity.WARNING,
                        category="loss_dynamics",
                        title="Loss plateau detected",
                        summary=(
                            f"Loss barely changed (relative range: {relative_range:.2e}) "
                            f"over {window} steps starting at step {int(steps[i])}."
                        ),
                        detail=(
                            f"The loss remained nearly constant (mean: {mean_val:.4e}, "
                            f"relative range: {relative_range:.2e}) for {window} "
                            f"consecutive steps (steps {int(steps[i])}-"
                            f"{int(steps[min(i + window - 1, n - 1)])}).\n\n"
                            f"Loss plateaus can indicate:\n"
                            f"- The model is stuck in a saddle point or flat region.\n"
                            f"- The learning rate is too small for the current loss landscape.\n"
                            f"- The model has converged (if loss is acceptably low)."
                        ),
                        evidence={
                            "relative_range": float(relative_range),
                            "mean_loss": float(mean_val),
                        },
                        step_range=(
                            int(steps[i]),
                            int(steps[min(i + window - 1, n - 1)]),
                        ),
                        remediation=[
                            "Try increasing the learning rate temporarily (learning rate restart/warmup cycle).",
                            "Use a learning rate scheduler with periodic restarts (cosine annealing with warm restarts).",
                            "Check if the model has actually converged -- the plateau may be the optimal loss.",
                            "Add noise to the optimizer (e.g., switch to SGD with higher momentum) to escape saddle points.",
                        ],
                        references=[
                            Reference(
                                title="Identifying and attacking the saddle point problem in high-dimensional non-convex optimization",
                                authors="Dauphin et al.",
                                year=2014,
                            ),
                        ],
                    )
                ]

        return findings

    def _check_catapult(
        self,
        loss_vals: np.ndarray,
        smoothed: np.ndarray | None,
        steps: np.ndarray,
    ) -> list[Finding]:
        """Check for catapult phase: sharp spike then rapid recovery in early training."""
        n = len(loss_vals)
        early_cutoff = max(n // 10, 5)
        if early_cutoff > n:
            return []

        early_loss = loss_vals[:early_cutoff]
        if len(early_loss) < 3:
            return []

        # Look for spike: loss > factor * running average
        running_avg = np.cumsum(early_loss) / np.arange(1, len(early_loss) + 1)
        for i in range(2, len(early_loss)):
            if early_loss[i] > self._catapult_spike_factor * running_avg[i - 1]:
                # Found spike, check for recovery
                post_spike = loss_vals[i:]
                if len(post_spike) < 3:
                    continue
                pre_spike_level = running_avg[i - 1]
                # Check if loss recovers to below pre-spike level
                recovery_idx = None
                for j in range(1, min(len(post_spike), early_cutoff)):
                    if post_spike[j] < pre_spike_level:
                        recovery_idx = j
                        break

                if recovery_idx is not None:
                    spike_val = float(early_loss[i])
                    return [
                        Finding(
                            detector=self.name,
                            severity=Severity.INFO,
                            category="loss_dynamics",
                            title="Catapult phase detected",
                            summary=(
                                f"Loss spiked to {spike_val:.3e} at step {int(steps[i])} "
                                f"({spike_val / pre_spike_level:.1f}x above running average) "
                                f"then recovered within {recovery_idx} steps."
                            ),
                            detail=(
                                f"A 'catapult phase' was detected early in training: the "
                                f"loss spiked sharply at step {int(steps[i])} to {spike_val:.3e} "
                                f"(running average was {pre_spike_level:.3e}), then recovered "
                                f"to below the pre-spike level within {recovery_idx} steps.\n\n"
                                f"This pattern is well-documented in deep learning: with "
                                f"large learning rates, networks can temporarily spike in loss "
                                f"before finding a flatter, more generalizable minimum. This "
                                f"is typically NOT harmful -- it often leads to better generalization."
                            ),
                            evidence={
                                "spike_step": int(steps[i]),
                                "spike_value": spike_val,
                                "pre_spike_avg": float(pre_spike_level),
                                "recovery_steps": recovery_idx,
                            },
                            step_range=(
                                int(steps[max(0, i - 1)]),
                                int(steps[min(i + recovery_idx, n - 1)]),
                            ),
                            references=[
                                Reference(
                                    title="The Large Learning Rate Phase of Neural Network Training",
                                    authors="Lewkowycz et al.",
                                    year=2020,
                                ),
                            ],
                        )
                    ]

        return []

    def _check_edge_of_stability(
        self,
        collector_states: dict[str, CollectorState],
        metadata: RunMetadata,
    ) -> list[Finding]:
        """Check for edge-of-stability: sharpness hovering near 2/lr."""
        curvature = collector_states.get("curvature")
        if curvature is None:
            return []

        eigenvalues = curvature.series.get("top_eigenvalue")
        if eigenvalues is None or len(eigenvalues) < 5:
            return []

        lr = metadata.learning_rate
        if lr is None or lr <= 0:
            return []

        threshold = 2.0 / lr
        tolerance = 0.3  # 30% tolerance

        # Count how many eigenvalue measurements are near 2/lr
        near_threshold = np.abs(eigenvalues - threshold) / threshold < tolerance
        near_frac = near_threshold.sum() / len(eigenvalues)

        if near_frac < 0.3:
            return []

        mean_sharpness = float(eigenvalues.mean())
        return [
            Finding(
                detector=self.name,
                severity=Severity.INFO,
                category="loss_dynamics",
                title="Edge of stability regime detected",
                summary=(
                    f"Sharpness (top Hessian eigenvalue) is hovering near 2/lr = "
                    f"{threshold:.1f} for {near_frac:.0%} of measured steps -- "
                    f"training is at the edge of stability."
                ),
                detail=(
                    f"The estimated top Hessian eigenvalue (mean: {mean_sharpness:.2f}) "
                    f"is near the stability threshold 2/lr = {threshold:.2f} for "
                    f"{near_frac:.0%} of the steps where curvature was measured.\n\n"
                    f"This is the 'edge of stability' phenomenon: gradient descent "
                    f"with a fixed learning rate naturally drives the sharpness toward "
                    f"2/lr, where the quadratic approximation becomes unstable. The "
                    f"optimizer effectively clips the learning rate to the local "
                    f"curvature.\n\n"
                    f"This is typically not a problem -- it's a normal regime for "
                    f"full-batch or large-batch training. The loss may oscillate but "
                    f"still decrease on average."
                ),
                evidence={
                    "mean_sharpness": mean_sharpness,
                    "stability_threshold": threshold,
                    "near_threshold_fraction": float(near_frac),
                },
                step_range=(int(curvature.steps[0]), int(curvature.steps[-1])),
                references=[
                    Reference(
                        title="Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability",
                        authors="Cohen et al.",
                        year=2021,
                    ),
                ],
            )
        ]
