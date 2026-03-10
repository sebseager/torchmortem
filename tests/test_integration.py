"""Integration test -- full end-to-end pipeline with a pathological model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from torchmortem import Autopsy, Severity


class _DeepSigmoidNet(nn.Module):
    """Deep MLP with sigmoid activations, will exhibit vanishing gradients."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        layers.append(nn.Linear(10, 32))
        layers.append(nn.Sigmoid())
        for _ in range(5):
            layers.append(nn.Linear(32, 32))
            layers.append(nn.Sigmoid())
        layers.append(nn.Linear(32, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TestEndToEnd:
    """Full pipeline integration tests."""

    def test_pathological_mlp_detects_vanishing(self, tmp_path: Path) -> None:
        """Train a deep sigmoid MLP and expect vanishing gradient findings."""
        model = _DeepSigmoidNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        with Autopsy(model, optimizer=optimizer, sampling="thorough") as autopsy:
            for _ in range(80):
                x = torch.randn(16, 10)
                y = torch.randn(16, 1)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                autopsy.step(loss=loss.item())

        # Generate both report formats.
        html_path = tmp_path / "report.html"
        json_path = tmp_path / "report.json"
        report = autopsy.report(html_path)
        autopsy.report(json_path)

        # Verify files exist and are non-trivial.
        assert html_path.exists()
        assert html_path.stat().st_size > 1000  # Should be substantial HTML
        assert json_path.exists()

        # Verify JSON is valid and structured.
        data = json.loads(json_path.read_text())
        assert data["metadata"]["model_name"] == "_DeepSigmoidNet"
        assert data["metadata"]["total_steps"] == 80
        assert data["metadata"]["optimizer_name"] == "SGD"
        assert len(data["findings"]) > 0

        # Verify report object.
        assert report.metadata.total_steps == 80
        assert len(report.findings) > 0
        assert report.executive_summary  # Non-empty

        # We expect vanishing gradients in a deep sigmoid network.
        categories = {f.category for f in report.findings}
        assert "gradient_flow" in categories

        # At least one finding should be WARNING or CRITICAL.
        severities = {f.severity for f in report.findings}
        assert Severity.WARNING in severities or Severity.CRITICAL in severities

    def test_healthy_model_minimal_findings(self, tmp_path: Path) -> None:
        """A well-configured shallow model should produce few/no findings."""
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        with Autopsy(model, optimizer=optimizer) as autopsy:
            for _ in range(50):
                x = torch.randn(16, 10)
                y = torch.randn(16, 1)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                autopsy.step(loss=loss.item())

        report = autopsy.report(tmp_path / "healthy.html")

        # Healthy model: either no findings or only INFO-level.
        critical = [f for f in report.findings if f.severity == Severity.CRITICAL]
        assert len(critical) == 0, f"Unexpected critical findings: {[f.title for f in critical]}"

    def test_collector_states_in_report(self, tmp_path: Path) -> None:
        """Collector states are present and have the right shape."""
        model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        num_steps = 20
        with Autopsy(model, optimizer=optimizer, sampling="thorough") as autopsy:
            for _ in range(num_steps):
                x = torch.randn(8, 5)
                optimizer.zero_grad()
                pred = model(x)
                loss = pred.sum()
                loss.backward()
                optimizer.step()
                autopsy.step(loss=loss.item())

        report = autopsy.get_report()
        assert report is not None

        # Gradient collector state
        assert "gradient" in report.collector_states
        grad_state = report.collector_states["gradient"]
        assert len(grad_state.steps) == num_steps
        assert "grad_norm" in grad_state.series
        assert grad_state.series["grad_norm"].shape[0] == num_steps

        # Loss collector state
        assert "loss" in report.collector_states
        loss_state = report.collector_states["loss"]
        assert len(loss_state.steps) == num_steps
        assert "loss" in loss_state.series
