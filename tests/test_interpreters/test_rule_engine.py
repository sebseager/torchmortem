"""Tests for the DefaultInterpreter rule engine (Phase 4 enhancements)."""

from __future__ import annotations

from torchmortem.interpreters.default import DefaultInterpreter
from torchmortem.types import (
    Finding,
    HealthScore,
    RunMetadata,
    Severity,
)


def _make_finding(
    severity: Severity = Severity.WARNING,
    title: str = "Test finding",
    category: str = "test",
    detector: str = "test_detector",
    affected_layers: list[str] | None = None,
    remediation: list[str] | None = None,
) -> Finding:
    return Finding(
        detector=detector,
        severity=severity,
        category=category,
        title=title,
        summary="Summary",
        detail="Detail",
        affected_layers=affected_layers or [],
        remediation=remediation or [],
    )


class TestHealthScores:
    """Tests for per-layer health score computation."""

    def test_healthy_model_all_ones(self) -> None:
        """No findings, all layers get score 1.0."""
        metadata = RunMetadata(
            total_steps=100,
            layer_names=["layer0", "layer1", "layer2"],
        )
        interpreter = DefaultInterpreter()
        report = interpreter.interpret([], metadata, {})

        assert len(report.health_scores) == 3
        for hs in report.health_scores:
            assert hs.score == 1.0
            assert hs.issues == []

    def test_critical_finding_reduces_score(self) -> None:
        """A CRITICAL finding on a specific layer penalizes it by 0.5."""
        findings = [
            _make_finding(
                Severity.CRITICAL,
                title="Dead units",
                category="dead_units",
                affected_layers=["layer1"],
            ),
        ]
        metadata = RunMetadata(
            total_steps=100,
            layer_names=["layer0", "layer1", "layer2"],
        )
        interpreter = DefaultInterpreter()
        report = interpreter.interpret(findings, metadata, {})

        scores_by_name = {hs.layer_name: hs for hs in report.health_scores}
        assert scores_by_name["layer0"].score == 1.0
        assert scores_by_name["layer1"].score == 0.5
        assert scores_by_name["layer2"].score == 1.0

    def test_warning_finding_reduces_score(self) -> None:
        """A WARNING finding penalizes by 0.2."""
        findings = [
            _make_finding(
                Severity.WARNING,
                title="Saturation",
                category="saturation",
                affected_layers=["layer0"],
            ),
        ]
        metadata = RunMetadata(
            total_steps=100,
            layer_names=["layer0", "layer1"],
        )
        interpreter = DefaultInterpreter()
        report = interpreter.interpret(findings, metadata, {})

        scores_by_name = {hs.layer_name: hs for hs in report.health_scores}
        assert scores_by_name["layer0"].score == 0.8
        assert scores_by_name["layer1"].score == 1.0

    def test_multiple_findings_stack(self) -> None:
        """Multiple findings on the same layer stack their penalties."""
        findings = [
            _make_finding(
                Severity.CRITICAL,
                title="Dead units",
                category="dead_units",
                affected_layers=["layer0"],
            ),
            _make_finding(
                Severity.WARNING,
                title="Saturation",
                category="saturation",
                affected_layers=["layer0"],
            ),
        ]
        metadata = RunMetadata(
            total_steps=100,
            layer_names=["layer0"],
        )
        interpreter = DefaultInterpreter()
        report = interpreter.interpret(findings, metadata, {})

        # 0.5 + 0.2 = 0.7 penalty; score = 0.3
        assert report.health_scores[0].score == 0.3

    def test_score_floors_at_zero(self) -> None:
        """Health score cannot go below 0."""
        findings = [
            _make_finding(
                Severity.CRITICAL,
                category="dead_units",
                affected_layers=["layer0"],
            ),
            _make_finding(
                Severity.CRITICAL,
                category="gradient_flow",
                affected_layers=["layer0"],
            ),
            _make_finding(
                Severity.CRITICAL,
                category="weight_norm",
                affected_layers=["layer0"],
            ),
        ]
        metadata = RunMetadata(
            total_steps=100,
            layer_names=["layer0"],
        )
        interpreter = DefaultInterpreter()
        report = interpreter.interpret(findings, metadata, {})

        # 3 * 0.5 = 1.5 penalty; floor at 0.0
        assert report.health_scores[0].score == 0.0

    def test_no_layer_names_no_scores(self) -> None:
        """If metadata has no layer names, health scores are empty."""
        metadata = RunMetadata(total_steps=100)
        interpreter = DefaultInterpreter()
        report = interpreter.interpret([], metadata, {})

        assert report.health_scores == []

    def test_issues_tracked(self) -> None:
        """Issues list on health scores contains finding titles."""
        findings = [
            _make_finding(
                Severity.WARNING,
                title="Saturated activations",
                category="saturation",
                affected_layers=["fc1"],
            ),
        ]
        metadata = RunMetadata(
            total_steps=100,
            layer_names=["fc1"],
        )
        interpreter = DefaultInterpreter()
        report = interpreter.interpret(findings, metadata, {})

        assert "Saturated activations" in report.health_scores[0].issues


class TestRuleEngine:
    """Tests for the correlation rule engine."""

    def test_gradient_starvation_rule_fires(self) -> None:
        """Gradient starvation fires when both gradient_flow + dead_units
        findings are present."""
        findings = [
            _make_finding(
                Severity.CRITICAL,
                title="Vanishing gradients detected",
                category="gradient_flow",
                affected_layers=["layer0", "layer1"],
            ),
            _make_finding(
                Severity.WARNING,
                title="Dead units in layer1",
                category="dead_units",
                affected_layers=["layer1"],
            ),
        ]
        metadata = RunMetadata(total_steps=100, layer_names=["layer0", "layer1"])
        interpreter = DefaultInterpreter()
        report = interpreter.interpret(findings, metadata, {})

        assert len(report.insights) >= 1
        names = [i.rule_name for i in report.insights]
        assert "gradient_starvation" in names

    def test_instability_rule_fires(self) -> None:
        """Instability feedback loop fires with exploding grads + weight
        explosion."""
        findings = [
            _make_finding(
                Severity.CRITICAL,
                title="Exploding gradients detected",
                category="gradient_flow",
            ),
            _make_finding(
                Severity.CRITICAL,
                title="Weight norm explosion",
                category="weight_norm",
            ),
        ]
        metadata = RunMetadata(total_steps=100)
        interpreter = DefaultInterpreter()
        report = interpreter.interpret(findings, metadata, {})

        names = [i.rule_name for i in report.insights]
        assert "instability_feedback_loop" in names

    def test_representation_bottleneck_fires(self) -> None:
        """Representation bottleneck fires with rank_collapse + loss_dynamics."""
        findings = [
            _make_finding(
                Severity.CRITICAL,
                title="Rank collapse in layer0",
                category="rank_collapse",
                affected_layers=["layer0"],
            ),
            _make_finding(
                Severity.WARNING,
                title="Loss plateau detected",
                category="loss_dynamics",
            ),
        ]
        metadata = RunMetadata(total_steps=100)
        interpreter = DefaultInterpreter()
        report = interpreter.interpret(findings, metadata, {})

        names = [i.rule_name for i in report.insights]
        assert "representation_bottleneck" in names

    def test_curvature_traps_fires(self) -> None:
        """Curvature traps fires when both plateau + edge-of-stability present."""
        findings = [
            _make_finding(
                Severity.WARNING,
                title="Loss plateau detected",
                category="loss_dynamics",
            ),
            _make_finding(
                Severity.INFO,
                title="Edge of stability regime detected",
                category="loss_dynamics",
            ),
        ]
        metadata = RunMetadata(total_steps=100)
        interpreter = DefaultInterpreter()
        report = interpreter.interpret(findings, metadata, {})

        names = [i.rule_name for i in report.insights]
        assert "curvature_traps" in names

    def test_no_rules_fire_without_matching_categories(self) -> None:
        """Rules don't fire if required categories are missing."""
        findings = [
            _make_finding(
                Severity.CRITICAL,
                title="Vanishing gradients detected",
                category="gradient_flow",
            ),
            # No dead_units, gradient_starvation shouldn't fire
        ]
        metadata = RunMetadata(total_steps=100)
        interpreter = DefaultInterpreter()
        report = interpreter.interpret(findings, metadata, {})

        starvation_insights = [i for i in report.insights if i.rule_name == "gradient_starvation"]
        assert len(starvation_insights) == 0

    def test_condition_prevents_firing(self) -> None:
        """Rules don't fire if condition returns False."""
        # Instability requires "exploding" in gradient_flow title and
        # "explosion" in weight_norm title.
        findings = [
            _make_finding(
                Severity.WARNING,
                title="Vanishing gradients detected",  # Not "exploding"
                category="gradient_flow",
            ),
            _make_finding(
                Severity.WARNING,
                title="Stagnant weight norms",  # Not "explosion"
                category="weight_norm",
            ),
        ]
        metadata = RunMetadata(total_steps=100)
        interpreter = DefaultInterpreter()
        report = interpreter.interpret(findings, metadata, {})

        instability_insights = [
            i for i in report.insights if i.rule_name == "instability_feedback_loop"
        ]
        assert len(instability_insights) == 0

    def test_insights_in_summary(self) -> None:
        """Cross-signal insights appear in the executive summary."""
        findings = [
            _make_finding(
                Severity.CRITICAL,
                title="Exploding gradients detected",
                category="gradient_flow",
            ),
            _make_finding(
                Severity.CRITICAL,
                title="Weight norm explosion",
                category="weight_norm",
            ),
        ]
        metadata = RunMetadata(
            model_name="TestNet",
            total_steps=100,
        )
        interpreter = DefaultInterpreter()
        report = interpreter.interpret(findings, metadata, {})

        assert "Cross-signal analysis" in report.executive_summary

    def test_insight_has_contributing_findings(self) -> None:
        """Each insight references its contributing findings."""
        findings = [
            _make_finding(
                Severity.CRITICAL,
                title="Exploding gradients detected",
                category="gradient_flow",
            ),
            _make_finding(
                Severity.CRITICAL,
                title="Weight norm explosion",
                category="weight_norm",
            ),
        ]
        metadata = RunMetadata(total_steps=100)
        interpreter = DefaultInterpreter()
        report = interpreter.interpret(findings, metadata, {})

        instability = [i for i in report.insights if i.rule_name == "instability_feedback_loop"]
        assert len(instability) == 1
        assert len(instability[0].contributing_findings) >= 2

    def test_insight_has_remediation(self) -> None:
        """Insights include actionable remediation steps."""
        findings = [
            _make_finding(
                Severity.CRITICAL,
                title="Exploding gradients detected",
                category="gradient_flow",
            ),
            _make_finding(
                Severity.CRITICAL,
                title="Weight norm explosion",
                category="weight_norm",
            ),
        ]
        metadata = RunMetadata(total_steps=100)
        interpreter = DefaultInterpreter()
        report = interpreter.interpret(findings, metadata, {})

        instability = [i for i in report.insights if i.rule_name == "instability_feedback_loop"]
        assert len(instability[0].remediation) >= 1
