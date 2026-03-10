"""Tests for DefaultInterpreter."""

from __future__ import annotations

from torchmortem.interpreters.default import DefaultInterpreter
from torchmortem.types import (
    Finding,
    RunMetadata,
    Severity,
)


def _make_finding(
    severity: Severity = Severity.WARNING,
    title: str = "Test finding",
    remediation: list[str] | None = None,
) -> Finding:
    return Finding(
        detector="test_detector",
        severity=severity,
        category="test",
        title=title,
        summary="Summary",
        detail="Detail",
        remediation=remediation or [],
    )


class TestDefaultInterpreter:
    """Tests for DefaultInterpreter."""

    def test_sorts_by_severity(self) -> None:
        """Findings are sorted highest severity first."""
        findings = [
            _make_finding(Severity.INFO, "Info finding"),
            _make_finding(Severity.CRITICAL, "Critical finding"),
            _make_finding(Severity.WARNING, "Warning finding"),
        ]
        metadata = RunMetadata(total_steps=100)
        interpreter = DefaultInterpreter()

        report = interpreter.interpret(findings, metadata, {})

        assert report.findings[0].severity == Severity.CRITICAL
        assert report.findings[1].severity == Severity.WARNING
        assert report.findings[2].severity == Severity.INFO

    def test_empty_findings_summary(self) -> None:
        """No findings, healthy summary."""
        metadata = RunMetadata(total_steps=100)
        interpreter = DefaultInterpreter()

        report = interpreter.interpret([], metadata, {})

        assert "No significant issues" in report.executive_summary

    def test_critical_findings_summary(self) -> None:
        """Critical findings appear in summary."""
        findings = [
            _make_finding(Severity.CRITICAL, "Vanishing gradients"),
        ]
        metadata = RunMetadata(model_name="MyModel", total_steps=200)
        interpreter = DefaultInterpreter()

        report = interpreter.interpret(findings, metadata, {})

        assert "CRITICAL" in report.executive_summary
        assert "MyModel" in report.executive_summary
        assert "200" in report.executive_summary

    def test_top_recommendation_in_summary(self) -> None:
        """Top remediation appears in summary."""
        findings = [
            _make_finding(
                Severity.CRITICAL,
                "Bad thing",
                remediation=["Fix the bad thing first"],
            ),
        ]
        metadata = RunMetadata(total_steps=100)
        interpreter = DefaultInterpreter()

        report = interpreter.interpret(findings, metadata, {})

        assert "Fix the bad thing first" in report.executive_summary

    def test_metadata_preserved(self) -> None:
        """Report metadata matches input."""
        metadata = RunMetadata(
            model_name="TestNet",
            total_steps=500,
            optimizer_name="Adam",
        )
        interpreter = DefaultInterpreter()

        report = interpreter.interpret([], metadata, {})

        assert report.metadata.model_name == "TestNet"
        assert report.metadata.total_steps == 500
