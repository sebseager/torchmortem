"""Tests for HTMLRenderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from torchmortem.renderers.html.renderer import HTMLRenderer
from torchmortem.types import (
    CollectorState,
    Finding,
    Reference,
    Report,
    RunMetadata,
    Severity,
)


def _make_report_with_findings() -> Report:
    """Create a Report with findings and collector data."""
    metadata = RunMetadata(
        model_name="TestNet",
        total_steps=200,
        total_parameters=5000,
        optimizer_name="Adam",
        learning_rate=0.001,
        device="cpu",
    )
    findings = [
        Finding(
            detector="gradient_flow",
            severity=Severity.CRITICAL,
            category="gradient_flow",
            title="Vanishing gradients detected",
            summary="Gradient signal is 500x weaker at early layers.",
            detail="Detailed explanation about vanishing gradients.",
            affected_layers=["fc1", "fc2"],
            step_range=(50, 200),
            remediation=["Add residual connections", "Use ReLU activations"],
            references=[Reference(title="Deep Residual Learning", authors="He et al.", year=2016)],
        ),
        Finding(
            detector="gradient_flow",
            severity=Severity.WARNING,
            category="gradient_flow",
            title="Gradient stalling in fc1",
            summary="Layer fc1 had near-zero gradients for 60% of steps.",
            detail="Detailed explanation about gradient stalling.",
            affected_layers=["fc1"],
            step_range=(100, 200),
            remediation=["Investigate upstream activations"],
        ),
    ]
    collector_states = {
        "gradient": CollectorState(
            name="gradient",
            steps=np.arange(200, dtype=np.int64),
            layers=["fc1", "fc2", "fc3"],
            series={"grad_norm": np.random.rand(200, 3)},
        ),
        "loss": CollectorState(
            name="loss",
            steps=np.arange(200, dtype=np.int64),
            series={
                "loss": np.linspace(5.0, 0.5, 200),
                "loss_smoothed": np.linspace(5.0, 0.5, 200),
            },
        ),
    }
    return Report(
        metadata=metadata,
        executive_summary="Training had critical gradient flow issues.",
        findings=findings,
        collector_states=collector_states,
    )


class TestHTMLRenderer:
    """Tests for HTMLRenderer."""

    def test_renders_html_file(self, tmp_path: Path) -> None:
        """Output file is created and contains HTML."""
        renderer = HTMLRenderer()
        report = _make_report_with_findings()
        output = tmp_path / "report.html"

        renderer.render(report, output)

        assert output.exists()
        html = output.read_text()
        assert "<!DOCTYPE html>" in html
        assert "torchmortem" in html

    def test_contains_executive_summary(self, tmp_path: Path) -> None:
        """Executive summary appears in the HTML."""
        renderer = HTMLRenderer()
        report = _make_report_with_findings()
        output = tmp_path / "report.html"
        renderer.render(report, output)

        html = output.read_text()
        assert "critical gradient flow issues" in html

    def test_contains_findings(self, tmp_path: Path) -> None:
        """Findings appear in the HTML."""
        renderer = HTMLRenderer()
        report = _make_report_with_findings()
        output = tmp_path / "report.html"
        renderer.render(report, output)

        html = output.read_text()
        assert "Vanishing gradients detected" in html
        assert "Gradient stalling" in html

    def test_contains_severity_badges(self, tmp_path: Path) -> None:
        """Severity levels appear in the HTML."""
        renderer = HTMLRenderer()
        report = _make_report_with_findings()
        output = tmp_path / "report.html"
        renderer.render(report, output)

        html = output.read_text()
        assert "CRITICAL" in html
        assert "WARNING" in html

    def test_contains_metadata(self, tmp_path: Path) -> None:
        """Run metadata appears in the HTML."""
        renderer = HTMLRenderer()
        report = _make_report_with_findings()
        output = tmp_path / "report.html"
        renderer.render(report, output)

        html = output.read_text()
        assert "TestNet" in html
        assert "200" in html  # total steps
        assert "Adam" in html

    def test_contains_plotly_charts(self, tmp_path: Path) -> None:
        """Plotly chart initialization code appears in the HTML."""
        renderer = HTMLRenderer()
        report = _make_report_with_findings()
        output = tmp_path / "report.html"
        renderer.render(report, output)

        html = output.read_text()
        assert "Plotly.newPlot" in html
        assert "gradient-chart" in html
        assert "loss-chart" in html

    def test_contains_remediation(self, tmp_path: Path) -> None:
        """Remediation suggestions appear in the HTML."""
        renderer = HTMLRenderer()
        report = _make_report_with_findings()
        output = tmp_path / "report.html"
        renderer.render(report, output)

        html = output.read_text()
        assert "Add residual connections" in html

    def test_empty_findings_shows_healthy(self, tmp_path: Path) -> None:
        """Report with no findings shows healthy message."""
        renderer = HTMLRenderer()
        report = Report(
            metadata=RunMetadata(model_name="GoodModel", total_steps=100),
            executive_summary="No issues detected.",
            findings=[],
            collector_states={},
        )
        output = tmp_path / "report.html"
        renderer.render(report, output)

        html = output.read_text()
        assert "healthy" in html.lower()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        renderer = HTMLRenderer()
        report = _make_report_with_findings()
        output = tmp_path / "sub" / "dir" / "report.html"
        renderer.render(report, output)
        assert output.exists()
