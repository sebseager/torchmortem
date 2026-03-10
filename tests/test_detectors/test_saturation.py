"""Tests for SaturationDetector."""

import numpy as np

from torchmortem.detectors.saturation import SaturationDetector
from torchmortem.types import CollectorState, RunMetadata, Severity


def _make_metadata(total_steps: int = 100) -> RunMetadata:
    return RunMetadata(model_name="TestModel", total_steps=total_steps)


class TestSaturationDetector:
    def test_detects_saturated_layer(self):
        steps = np.arange(100)
        sat_frac = np.zeros((100, 2))
        sat_frac[:, 1] = 0.85  # Layer 1 is highly saturated

        state = CollectorState(
            name="activation",
            steps=steps,
            layers=["layer0", "layer1"],
            series={"act_saturated_frac": sat_frac},
        )

        detector = SaturationDetector()
        findings = detector.analyze({"activation": state}, _make_metadata())
        assert len(findings) == 1
        assert findings[0].category == "saturation"
        assert findings[0].severity == Severity.CRITICAL
        assert "layer1" in findings[0].affected_layers

    def test_healthy_no_findings(self):
        steps = np.arange(100)
        sat_frac = np.full((100, 2), 0.1)  # Low saturation

        state = CollectorState(
            name="activation",
            steps=steps,
            layers=["l0", "l1"],
            series={"act_saturated_frac": sat_frac},
        )

        detector = SaturationDetector()
        findings = detector.analyze({"activation": state}, _make_metadata())
        assert len(findings) == 0

    def test_warning_severity(self):
        steps = np.arange(100)
        sat_frac = np.full((100, 1), 0.65)

        state = CollectorState(
            name="activation",
            steps=steps,
            layers=["layer0"],
            series={"act_saturated_frac": sat_frac},
        )

        detector = SaturationDetector()
        findings = detector.analyze({"activation": state}, _make_metadata())
        assert len(findings) == 1
        assert findings[0].severity == Severity.WARNING

    def test_empty_data(self):
        state = CollectorState(
            name="activation",
            steps=np.array([], dtype=np.int64),
            layers=[],
            series={"act_saturated_frac": np.array([])},
        )

        detector = SaturationDetector()
        findings = detector.analyze({"activation": state}, _make_metadata(0))
        assert len(findings) == 0

    def test_has_remediation_and_references(self):
        steps = np.arange(100)
        sat_frac = np.full((100, 1), 0.9)
        state = CollectorState(
            name="activation",
            steps=steps,
            layers=["layer0"],
            series={"act_saturated_frac": sat_frac},
        )

        detector = SaturationDetector()
        findings = detector.analyze({"activation": state}, _make_metadata())
        assert len(findings[0].remediation) > 0
        assert len(findings[0].references) > 0
