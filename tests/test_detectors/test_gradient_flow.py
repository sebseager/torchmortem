"""Tests for GradientFlowDetector."""

from __future__ import annotations

import numpy as np

from torchmortem.detectors.gradient_flow import GradientFlowDetector
from torchmortem.types import CollectorState, RunMetadata, Severity


def _make_gradient_state(
    grad_norms: np.ndarray,
    num_steps: int | None = None,
    layer_names: list[str] | None = None,
) -> dict[str, CollectorState]:
    """Create a synthetic gradient CollectorState.

    Args:
        grad_norms: Shape (num_steps, num_layers).
        num_steps: If None, inferred from grad_norms.
        layer_names: If None, generated as "layer_0", "layer_1", etc.
    """
    n_steps, n_layers = grad_norms.shape
    if num_steps is None:
        num_steps = n_steps
    if layer_names is None:
        layer_names = [f"layer_{i}" for i in range(n_layers)]

    return {
        "gradient": CollectorState(
            name="gradient",
            steps=np.arange(num_steps, dtype=np.int64),
            layers=layer_names,
            series={
                "grad_norm": grad_norms,
                "grad_mean": np.zeros_like(grad_norms),
                "grad_max": grad_norms,
                "grad_zero_frac": np.zeros_like(grad_norms),
            },
        )
    }


def _default_metadata(total_steps: int = 100) -> RunMetadata:
    return RunMetadata(total_steps=total_steps)


class TestGradientFlowDetector:
    """Tests for GradientFlowDetector."""

    def test_detects_vanishing_gradients(self) -> None:
        """Classic vanishing gradient: first layer norm << last layer norm."""
        n_steps = 100
        n_layers = 5
        # Gradient norms decay exponentially from last layer to first.
        # Last layer: ~1.0, First layer: ~0.001
        norms = np.ones((n_steps, n_layers))
        for i in range(n_layers):
            norms[:, i] = 10 ** (-3 + i)  # 0.001, 0.01, 0.1, 1, 10

        states = _make_gradient_state(norms)
        detector = GradientFlowDetector(vanishing_ratio_threshold=100)

        findings = detector.analyze(states, _default_metadata(n_steps))

        assert len(findings) >= 1
        vanishing = [f for f in findings if "Vanishing" in f.title]
        assert len(vanishing) == 1
        assert vanishing[0].severity in (Severity.WARNING, Severity.CRITICAL)
        assert vanishing[0].category == "gradient_flow"
        assert len(vanishing[0].remediation) > 0
        assert len(vanishing[0].affected_layers) > 0

    def test_no_finding_for_healthy_gradients(self) -> None:
        """Uniform gradient norms should produce no vanishing/exploding finding."""
        n_steps = 100
        n_layers = 4
        norms = np.ones((n_steps, n_layers)) * 0.5  # All layers have norm ~0.5

        states = _make_gradient_state(norms)
        detector = GradientFlowDetector()

        findings = detector.analyze(states, _default_metadata(n_steps))

        vanishing = [f for f in findings if "Vanishing" in f.title]
        exploding = [f for f in findings if "Exploding" in f.title]
        assert len(vanishing) == 0
        assert len(exploding) == 0

    def test_detects_exploding_gradients(self) -> None:
        """Gradient norms exceeding threshold should trigger exploding."""
        n_steps = 100
        n_layers = 3
        norms = np.ones((n_steps, n_layers)) * 0.5
        # Inject explosion at step 50 in layer 1
        norms[50, 1] = 5000.0
        norms[51, 1] = 3000.0

        states = _make_gradient_state(norms)
        detector = GradientFlowDetector(exploding_norm_threshold=1000)

        findings = detector.analyze(states, _default_metadata(n_steps))

        exploding = [f for f in findings if "Exploding" in f.title]
        assert len(exploding) == 1
        assert exploding[0].severity == Severity.CRITICAL
        assert "layer_1" in exploding[0].affected_layers

    def test_detects_gradient_stalling(self) -> None:
        """A layer with near-zero gradient for >30% of steps should be flagged."""
        n_steps = 100
        n_layers = 3
        norms = np.ones((n_steps, n_layers)) * 0.5
        # Layer 0 has zero gradient for 50% of steps
        norms[:50, 0] = 1e-12

        states = _make_gradient_state(norms)
        detector = GradientFlowDetector(stall_threshold=1e-8, stall_fraction=0.3)

        findings = detector.analyze(states, _default_metadata(n_steps))

        stalling = [f for f in findings if "stalling" in f.title.lower()]
        assert len(stalling) == 1
        assert "layer_0" in stalling[0].affected_layers

    def test_empty_data_returns_no_findings(self) -> None:
        """Empty collector state should return empty findings list."""
        states = _make_gradient_state(np.zeros((0, 0)).reshape(0, 0))
        # Need at least 2 layers
        states["gradient"] = CollectorState(
            name="gradient",
            steps=np.array([], dtype=np.int64),
            layers=[],
            series={"grad_norm": np.zeros((0, 0))},
        )
        detector = GradientFlowDetector()
        findings = detector.analyze(states, _default_metadata(0))
        assert findings == []

    def test_single_layer_returns_no_vanishing(self) -> None:
        """Vanishing detection needs at least 2 layers."""
        norms = np.ones((50, 1)) * 0.001
        states = _make_gradient_state(norms, layer_names=["only_layer"])
        detector = GradientFlowDetector()
        findings = detector.analyze(states, _default_metadata(50))
        vanishing = [f for f in findings if "Vanishing" in f.title]
        assert len(vanishing) == 0

    def test_vanishing_severity_scales_with_ratio(self) -> None:
        """Extreme ratio -> CRITICAL, moderate ratio -> WARNING."""
        n_steps = 100
        n_layers = 3

        # Moderate vanishing (ratio ~200)
        norms_moderate = np.ones((n_steps, n_layers))
        norms_moderate[:, 0] = 0.005
        norms_moderate[:, 2] = 1.0

        states = _make_gradient_state(norms_moderate)
        detector = GradientFlowDetector(vanishing_ratio_threshold=100)
        findings = detector.analyze(states, _default_metadata(n_steps))
        vanishing = [f for f in findings if "Vanishing" in f.title]
        assert len(vanishing) == 1
        assert vanishing[0].severity == Severity.WARNING

        # Extreme vanishing (ratio ~10000)
        norms_extreme = np.ones((n_steps, n_layers))
        norms_extreme[:, 0] = 0.0001
        norms_extreme[:, 2] = 1.0

        states = _make_gradient_state(norms_extreme)
        findings = detector.analyze(states, _default_metadata(n_steps))
        vanishing = [f for f in findings if "Vanishing" in f.title]
        assert len(vanishing) == 1
        assert vanishing[0].severity == Severity.CRITICAL

    def test_finding_has_references(self) -> None:
        """Vanishing gradient finding should include paper references."""
        n_steps = 100
        n_layers = 3
        norms = np.ones((n_steps, n_layers))
        norms[:, 0] = 0.001

        states = _make_gradient_state(norms)
        detector = GradientFlowDetector(vanishing_ratio_threshold=100)
        findings = detector.analyze(states, _default_metadata(n_steps))

        vanishing = [f for f in findings if "Vanishing" in f.title]
        assert len(vanishing) == 1
        assert len(vanishing[0].references) > 0
        ref_titles = [r.title for r in vanishing[0].references]
        assert any("deep" in t.lower() for t in ref_titles)
