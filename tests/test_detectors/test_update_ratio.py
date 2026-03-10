"""Tests for UpdateRatioDetector."""

import numpy as np

from torchmortem.detectors.update_ratio import UpdateRatioDetector
from torchmortem.types import CollectorState, RunMetadata, Severity


def _make_metadata(total_steps: int = 100) -> RunMetadata:
    return RunMetadata(model_name="TestModel", total_steps=total_steps)


class TestUpdateRatioDetector:
    def test_detects_too_low(self):
        steps = np.arange(100)
        ratios = np.full((100, 2), 1e-3)  # Healthy
        ratios[:, 0] = 1e-7  # Too low

        state = CollectorState(
            name="weight",
            steps=steps,
            layers=["frozen_layer", "healthy_layer"],
            series={"update_ratio": ratios},
        )

        detector = UpdateRatioDetector()
        findings = detector.analyze({"weight": state}, _make_metadata())
        assert len(findings) == 1
        assert findings[0].severity == Severity.WARNING
        assert "frozen_layer" in findings[0].affected_layers
        assert "too low" in findings[0].title.lower()

    def test_detects_too_high(self):
        steps = np.arange(100)
        ratios = np.full((100, 1), 0.5)  # Way too high

        state = CollectorState(
            name="weight",
            steps=steps,
            layers=["unstable_layer"],
            series={"update_ratio": ratios},
        )

        detector = UpdateRatioDetector()
        findings = detector.analyze({"weight": state}, _make_metadata())
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL
        assert "too high" in findings[0].title.lower()

    def test_healthy_no_findings(self):
        steps = np.arange(100)
        ratios = np.full((100, 3), 1e-3)  # All healthy

        state = CollectorState(
            name="weight",
            steps=steps,
            layers=["l0", "l1", "l2"],
            series={"update_ratio": ratios},
        )

        detector = UpdateRatioDetector()
        findings = detector.analyze({"weight": state}, _make_metadata())
        assert len(findings) == 0

    def test_empty_data(self):
        state = CollectorState(
            name="weight",
            steps=np.array([0], dtype=np.int64),
            layers=["l0"],
            series={"update_ratio": np.array([[1e-3]])},
        )

        detector = UpdateRatioDetector()
        # Only 1 step should skip (need at least 2)
        findings = detector.analyze({"weight": state}, _make_metadata(1))
        assert len(findings) == 0

    def test_has_references(self):
        steps = np.arange(100)
        ratios = np.full((100, 1), 0.5)

        state = CollectorState(
            name="weight",
            steps=steps,
            layers=["layer0"],
            series={"update_ratio": ratios},
        )

        detector = UpdateRatioDetector()
        findings = detector.analyze({"weight": state}, _make_metadata())
        assert len(findings[0].references) > 0
