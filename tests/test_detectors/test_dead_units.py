"""Tests for DeadUnitDetector."""

import numpy as np
import pytest

from torchmortem.detectors.dead_units import DeadUnitDetector
from torchmortem.types import CollectorState, RunMetadata, Severity


def _make_metadata(total_steps: int = 100) -> RunMetadata:
    return RunMetadata(model_name="TestModel", total_steps=total_steps)


class TestDeadUnitDetector:
    def test_detects_dead_layer(self):
        """A layer with >50% dead units should be flagged."""
        steps = np.arange(100)
        # Layer 0 is healthy, layer 1 has 90% dead units throughout
        dead_frac = np.zeros((100, 2))
        dead_frac[:, 1] = 0.9

        state = CollectorState(
            name="activation",
            steps=steps,
            layers=["layer0", "layer1"],
            series={"act_dead_frac": dead_frac},
        )

        detector = DeadUnitDetector()
        findings = detector.analyze({"activation": state}, _make_metadata())

        assert len(findings) == 1
        assert findings[0].category == "dead_units"
        assert findings[0].severity == Severity.CRITICAL  # 90% > 80%
        assert "layer1" in findings[0].affected_layers

    def test_healthy_no_findings(self):
        """Layers with low dead fraction should not be flagged."""
        steps = np.arange(100)
        dead_frac = np.full((100, 3), 0.05)  # Only 5% dead

        state = CollectorState(
            name="activation",
            steps=steps,
            layers=["l0", "l1", "l2"],
            series={"act_dead_frac": dead_frac},
        )

        detector = DeadUnitDetector()
        findings = detector.analyze({"activation": state}, _make_metadata())
        assert len(findings) == 0

    def test_warning_severity(self):
        """Dead fraction between 50-80% should be WARNING."""
        steps = np.arange(100)
        dead_frac = np.zeros((100, 1))
        dead_frac[:, 0] = 0.65  # Between thresholds

        state = CollectorState(
            name="activation",
            steps=steps,
            layers=["layer0"],
            series={"act_dead_frac": dead_frac},
        )

        detector = DeadUnitDetector()
        findings = detector.analyze({"activation": state}, _make_metadata())
        assert len(findings) == 1
        assert findings[0].severity == Severity.WARNING

    def test_transient_not_flagged(self):
        """Dead units only in the first half should not be flagged."""
        steps = np.arange(100)
        dead_frac = np.zeros((100, 1))
        dead_frac[:50, 0] = 0.9  # Only dead in first half
        dead_frac[50:, 0] = 0.1  # Healthy in second half

        state = CollectorState(
            name="activation",
            steps=steps,
            layers=["layer0"],
            series={"act_dead_frac": dead_frac},
        )

        detector = DeadUnitDetector()
        findings = detector.analyze({"activation": state}, _make_metadata())
        assert len(findings) == 0

    def test_empty_data(self):
        state = CollectorState(
            name="activation",
            steps=np.array([], dtype=np.int64),
            layers=[],
            series={"act_dead_frac": np.array([])},
        )

        detector = DeadUnitDetector()
        findings = detector.analyze({"activation": state}, _make_metadata(0))
        assert len(findings) == 0

    def test_has_remediation(self):
        steps = np.arange(100)
        dead_frac = np.full((100, 1), 0.9)
        state = CollectorState(
            name="activation",
            steps=steps,
            layers=["layer0"],
            series={"act_dead_frac": dead_frac},
        )

        detector = DeadUnitDetector()
        findings = detector.analyze({"activation": state}, _make_metadata())
        assert len(findings) == 1
        assert len(findings[0].remediation) > 0
        assert len(findings[0].references) > 0
