"""Tests for JSONRenderer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from torchmortem.renderers.json_renderer import JSONRenderer
from torchmortem.types import (
    CollectorState,
    Finding,
    Report,
    RunMetadata,
    Severity,
)


def _make_sample_report() -> Report:
    """Create a minimal Report for testing."""
    metadata = RunMetadata(
        model_name="TestModel",
        total_steps=100,
        total_parameters=1234,
        optimizer_name="SGD",
        learning_rate=0.01,
    )
    findings = [
        Finding(
            detector="gradient_flow",
            severity=Severity.WARNING,
            category="gradient_flow",
            title="Test finding",
            summary="A test finding for the renderer.",
            detail="Detailed explanation.",
            affected_layers=["layer_0"],
            step_range=(10, 90),
            remediation=["Do something", "Do another thing"],
        ),
    ]
    collector_states = {
        "gradient": CollectorState(
            name="gradient",
            steps=np.arange(5, dtype=np.int64),
            layers=["layer_0", "layer_1"],
            series={"grad_norm": np.random.rand(5, 2)},
        ),
    }
    return Report(
        metadata=metadata,
        executive_summary="All is well.",
        findings=findings,
        collector_states=collector_states,
    )


class TestJSONRenderer:
    """Tests for JSONRenderer."""

    def test_renders_valid_json(self, tmp_path: Path) -> None:
        """Output file is valid JSON."""
        renderer = JSONRenderer()
        report = _make_sample_report()
        output = tmp_path / "report.json"

        renderer.render(report, output)

        assert output.exists()
        data = json.loads(output.read_text())
        assert isinstance(data, dict)

    def test_contains_expected_keys(self, tmp_path: Path) -> None:
        """JSON has the expected top-level structure."""
        renderer = JSONRenderer()
        report = _make_sample_report()
        output = tmp_path / "report.json"
        renderer.render(report, output)

        data = json.loads(output.read_text())
        assert "metadata" in data
        assert "executive_summary" in data
        assert "findings" in data
        assert "collector_data" in data

    def test_metadata_fields(self, tmp_path: Path) -> None:
        """Metadata fields are preserved."""
        renderer = JSONRenderer()
        report = _make_sample_report()
        output = tmp_path / "report.json"
        renderer.render(report, output)

        data = json.loads(output.read_text())
        assert data["metadata"]["model_name"] == "TestModel"
        assert data["metadata"]["total_steps"] == 100
        assert data["metadata"]["optimizer_name"] == "SGD"

    def test_findings_have_severity_as_string(self, tmp_path: Path) -> None:
        """Severity should be serialized as a lowercase string."""
        renderer = JSONRenderer()
        report = _make_sample_report()
        output = tmp_path / "report.json"
        renderer.render(report, output)

        data = json.loads(output.read_text())
        assert data["findings"][0]["severity"] == "warning"

    def test_numpy_arrays_serialized(self, tmp_path: Path) -> None:
        """Numpy arrays in collector data should be serialized to lists."""
        renderer = JSONRenderer()
        report = _make_sample_report()
        output = tmp_path / "report.json"
        renderer.render(report, output)

        data = json.loads(output.read_text())
        grad_data = data["collector_data"]["gradient"]
        assert isinstance(grad_data["steps"], list)
        assert isinstance(grad_data["series"]["grad_norm"], list)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Output path's parent directory is created if it doesn't exist."""
        renderer = JSONRenderer()
        report = _make_sample_report()
        output = tmp_path / "subdir" / "nested" / "report.json"
        renderer.render(report, output)
        assert output.exists()
