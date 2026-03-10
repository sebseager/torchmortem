"""Tests for RankCollapseDetector."""

import numpy as np

from torchmortem.detectors.rank_collapse import RankCollapseDetector
from torchmortem.types import CollectorState, RunMetadata, Severity


def _make_metadata(total_steps: int = 100) -> RunMetadata:
    return RunMetadata(model_name="TestModel", total_steps=total_steps)


class TestRankCollapseDetector:
    def test_detects_collapse(self):
        steps = np.arange(30)
        # Effective rank drops from 10 to 2
        ranks = np.zeros((30, 1))
        ranks[:10, 0] = 10.0  # High early
        ranks[10:20, 0] = 6.0
        ranks[20:, 0] = 2.0  # Low late

        state = CollectorState(
            name="rank",
            steps=steps,
            layers=["layer0"],
            series={"effective_rank": ranks},
        )

        detector = RankCollapseDetector()
        findings = detector.analyze({"rank": state}, _make_metadata(30))
        assert len(findings) == 1
        assert findings[0].category == "rank_collapse"
        assert "layer0" in findings[0].affected_layers

    def test_healthy_no_findings(self):
        steps = np.arange(30)
        # Stable rank
        ranks = np.full((30, 2), 8.0)

        state = CollectorState(
            name="rank",
            steps=steps,
            layers=["l0", "l1"],
            series={"effective_rank": ranks},
        )

        detector = RankCollapseDetector()
        findings = detector.analyze({"rank": state}, _make_metadata(30))
        assert len(findings) == 0

    def test_severity_scales_with_collapse(self):
        steps = np.arange(30)
        # Severe collapse, 10 to 1
        ranks = np.zeros((30, 1))
        ranks[:10, 0] = 10.0
        ranks[20:, 0] = 1.0

        state = CollectorState(
            name="rank",
            steps=steps,
            layers=["layer0"],
            series={"effective_rank": ranks},
        )

        detector = RankCollapseDetector()
        findings = detector.analyze({"rank": state}, _make_metadata(30))
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL  # < 30% remaining

    def test_insufficient_data(self):
        steps = np.array([0, 1])
        ranks = np.array([[5.0], [4.0]])

        state = CollectorState(
            name="rank",
            steps=steps,
            layers=["l0"],
            series={"effective_rank": ranks},
        )

        detector = RankCollapseDetector()
        findings = detector.analyze({"rank": state}, _make_metadata(2))
        assert len(findings) == 0

    def test_has_references(self):
        steps = np.arange(30)
        ranks = np.zeros((30, 1))
        ranks[:10, 0] = 10.0
        ranks[20:, 0] = 2.0

        state = CollectorState(
            name="rank",
            steps=steps,
            layers=["layer0"],
            series={"effective_rank": ranks},
        )

        detector = RankCollapseDetector()
        findings = detector.analyze({"rank": state}, _make_metadata(30))
        assert len(findings) == 1
        assert len(findings[0].references) > 0
