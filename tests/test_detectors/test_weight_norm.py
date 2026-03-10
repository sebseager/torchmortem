"""Tests for WeightNormDetector."""

import numpy as np

from torchmortem.detectors.weight_norm import WeightNormDetector
from torchmortem.types import CollectorState, RunMetadata, Severity


def _make_metadata(total_steps: int = 100) -> RunMetadata:
    return RunMetadata(model_name="TestModel", total_steps=total_steps)


class TestWeightNormDetector:
    def test_detects_explosion(self):
        steps = np.arange(100)
        w_norms = np.ones((100, 2))
        # Layer 1 grows 50x
        w_norms[:, 1] = np.linspace(1, 50, 100)

        state = CollectorState(
            name="weight",
            steps=steps,
            layers=["stable", "exploding"],
            series={"weight_norm": w_norms},
        )

        detector = WeightNormDetector()
        findings = detector.analyze({"weight": state}, _make_metadata())
        assert any(f.title == "Weight norm explosion" for f in findings)
        explosion_finding = [f for f in findings if "explosion" in f.title.lower()][0]
        assert "exploding" in explosion_finding.affected_layers

    def test_detects_stagnation(self):
        steps = np.arange(100)
        w_norms = np.ones((100, 2))
        # Layer 0 barely changes (< 0.01% relative change)
        w_norms[:, 0] = 5.0 + np.random.default_rng(42).uniform(-1e-6, 1e-6, 100)
        # Layer 1 changes significantly
        w_norms[:, 1] = np.linspace(1.0, 2.0, 100)

        state = CollectorState(
            name="weight",
            steps=steps,
            layers=["frozen", "learning"],
            series={"weight_norm": w_norms},
        )

        detector = WeightNormDetector()
        findings = detector.analyze({"weight": state}, _make_metadata())
        stagnation = [f for f in findings if "stagnant" in f.title.lower()]
        assert len(stagnation) == 1
        assert "frozen" in stagnation[0].affected_layers

    def test_detects_imbalance(self):
        steps = np.arange(10)
        w_norms = np.ones((10, 3))
        w_norms[:, 0] = 0.01  # Tiny
        w_norms[:, 2] = 100.0  # Huge

        state = CollectorState(
            name="weight",
            steps=steps,
            layers=["tiny", "normal", "huge"],
            series={"weight_norm": w_norms},
        )

        detector = WeightNormDetector(imbalance_factor=100.0)
        findings = detector.analyze({"weight": state}, _make_metadata(10))
        imbalance = [f for f in findings if "imbalance" in f.title.lower()]
        assert len(imbalance) == 1

    def test_healthy_no_findings(self):
        steps = np.arange(100)
        w_norms = np.ones((100, 3))
        # All layers grow modestly (1.0 -> 1.5)
        for i in range(3):
            w_norms[:, i] = np.linspace(1.0, 1.5, 100)

        state = CollectorState(
            name="weight",
            steps=steps,
            layers=["l0", "l1", "l2"],
            series={"weight_norm": w_norms},
        )

        detector = WeightNormDetector()
        findings = detector.analyze({"weight": state}, _make_metadata())
        assert len(findings) == 0

    def test_empty_data(self):
        state = CollectorState(
            name="weight",
            steps=np.array([0], dtype=np.int64),
            layers=["l0"],
            series={"weight_norm": np.array([[1.0]])},
        )

        detector = WeightNormDetector()
        findings = detector.analyze({"weight": state}, _make_metadata(1))
        assert len(findings) == 0
