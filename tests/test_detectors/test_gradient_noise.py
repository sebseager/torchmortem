"""Tests for GradientNoiseDetector."""

import numpy as np

from torchmortem.detectors.gradient_noise import GradientNoiseDetector
from torchmortem.types import CollectorState, RunMetadata, Severity


def _make_metadata(total_steps: int = 100) -> RunMetadata:
    return RunMetadata(model_name="TestModel", total_steps=total_steps)


class TestGradientNoiseDetector:
    def test_detects_high_noise(self):
        rng = np.random.default_rng(42)
        steps = np.arange(100)
        # Very noisy gradients: std >> mean
        grad_norms = rng.exponential(0.1, (100, 3))  # High variance, low mean

        state = CollectorState(
            name="gradient",
            steps=steps,
            layers=["l0", "l1", "l2"],
            series={"grad_norm": grad_norms},
        )

        detector = GradientNoiseDetector(low_snr_threshold=2.0)
        findings = detector.analyze({"gradient": state}, _make_metadata())
        assert len(findings) == 1
        assert findings[0].category == "gradient_noise"
        assert findings[0].severity == Severity.WARNING

    def test_healthy_no_findings(self):
        steps = np.arange(100)
        # Very stable gradients: mean >> std
        grad_norms = np.ones((100, 3)) * 5.0  # Constant norms
        grad_norms += np.random.default_rng(42).uniform(-0.01, 0.01, (100, 3))

        state = CollectorState(
            name="gradient",
            steps=steps,
            layers=["l0", "l1", "l2"],
            series={"grad_norm": grad_norms},
        )

        detector = GradientNoiseDetector()
        findings = detector.analyze({"gradient": state}, _make_metadata())
        assert len(findings) == 0

    def test_insufficient_steps(self):
        steps = np.arange(5)
        grad_norms = np.ones((5, 2))

        state = CollectorState(
            name="gradient",
            steps=steps,
            layers=["l0", "l1"],
            series={"grad_norm": grad_norms},
        )

        detector = GradientNoiseDetector(min_steps=20)
        findings = detector.analyze({"gradient": state}, _make_metadata(5))
        assert len(findings) == 0

    def test_has_references(self):
        rng = np.random.default_rng(42)
        steps = np.arange(100)
        grad_norms = rng.exponential(0.1, (100, 2))

        state = CollectorState(
            name="gradient",
            steps=steps,
            layers=["l0", "l1"],
            series={"grad_norm": grad_norms},
        )

        detector = GradientNoiseDetector(low_snr_threshold=2.0)
        findings = detector.analyze({"gradient": state}, _make_metadata())
        if findings:
            assert len(findings[0].references) > 0
