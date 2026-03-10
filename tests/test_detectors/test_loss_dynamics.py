"""Tests for LossDynamicsDetector."""

import numpy as np

from torchmortem.detectors.loss_dynamics import LossDynamicsDetector
from torchmortem.types import CollectorState, RunMetadata, Severity


def _make_metadata(total_steps: int = 100, lr: float | None = 0.01) -> RunMetadata:
    return RunMetadata(model_name="TestModel", total_steps=total_steps, learning_rate=lr)


class TestLossDynamicsDetector:
    def test_detects_divergence(self):
        steps = np.arange(100)
        # Loss increases monotonically in the tail
        loss = np.concatenate([np.ones(60), np.linspace(1, 10, 40)])
        smoothed = loss.copy()

        state = CollectorState(
            name="loss",
            steps=steps,
            series={"loss": loss, "loss_smoothed": smoothed},
        )

        detector = LossDynamicsDetector()
        findings = detector.analyze({"loss": state}, _make_metadata())
        divergence = [f for f in findings if "divergence" in f.title.lower()]
        assert len(divergence) == 1
        assert divergence[0].severity == Severity.CRITICAL

    def test_detects_plateau(self):
        steps = np.arange(200)
        # Loss starts high then plateaus
        loss = np.concatenate([np.linspace(10, 1, 100), np.ones(100) * 1.0])
        smoothed = loss.copy()

        state = CollectorState(
            name="loss",
            steps=steps,
            series={"loss": loss, "loss_smoothed": smoothed},
        )

        detector = LossDynamicsDetector(plateau_window=20, plateau_threshold=1e-3)
        findings = detector.analyze({"loss": state}, _make_metadata(200))
        plateau = [f for f in findings if "plateau" in f.title.lower()]
        assert len(plateau) == 1
        assert plateau[0].severity == Severity.WARNING

    def test_detects_catapult(self):
        steps = np.arange(100)
        # Normal start, then big spike, then recovery
        loss = np.ones(100) * 0.5
        loss[5] = 5.0  # Big spike
        loss[6:] = np.linspace(0.4, 0.2, 94)  # Recovery below pre-spike

        state = CollectorState(
            name="loss",
            steps=steps,
            series={"loss": loss, "loss_smoothed": loss.copy()},
        )

        detector = LossDynamicsDetector()
        findings = detector.analyze({"loss": state}, _make_metadata())
        catapult = [f for f in findings if "catapult" in f.title.lower()]
        assert len(catapult) == 1
        assert catapult[0].severity == Severity.INFO

    def test_detects_edge_of_stability(self):
        steps = np.arange(100)
        loss = np.linspace(1, 0.1, 100)

        loss_state = CollectorState(
            name="loss",
            steps=steps,
            series={"loss": loss, "loss_smoothed": loss},
        )

        # Curvature near 2/lr = 200 (lr=0.01)
        curv_steps = np.arange(0, 100, 10)
        eigenvalues = np.ones(10) * 195 + np.random.default_rng(42).uniform(-10, 10, 10)

        curv_state = CollectorState(
            name="curvature",
            steps=curv_steps,
            series={"top_eigenvalue": eigenvalues},
        )

        detector = LossDynamicsDetector()
        findings = detector.analyze(
            {"loss": loss_state, "curvature": curv_state},
            _make_metadata(lr=0.01),
        )
        eos = [f for f in findings if "edge of stability" in f.title.lower()]
        assert len(eos) == 1
        assert eos[0].severity == Severity.INFO

    def test_healthy_no_findings(self):
        steps = np.arange(100)
        loss = np.linspace(2.0, 0.1, 100)  # Healthy decreasing loss

        state = CollectorState(
            name="loss",
            steps=steps,
            series={"loss": loss, "loss_smoothed": loss},
        )

        detector = LossDynamicsDetector()
        findings = detector.analyze({"loss": state}, _make_metadata())
        # Healthy decreasing loss should have no critical/warning findings
        critical_or_warning = [
            f for f in findings if f.severity in (Severity.CRITICAL, Severity.WARNING)
        ]
        assert len(critical_or_warning) == 0

    def test_empty_data(self):
        state = CollectorState(
            name="loss",
            steps=np.array([], dtype=np.int64),
            series={"loss": np.array([])},
        )

        detector = LossDynamicsDetector()
        findings = detector.analyze({"loss": state}, _make_metadata(0))
        assert len(findings) == 0
