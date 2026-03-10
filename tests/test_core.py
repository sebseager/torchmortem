"""Tests for Autopsy orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from torchmortem import Autopsy, SamplingConfig


class TestAutopsyContextManager:
    """Tests for Autopsy as a context manager."""

    def test_basic_context_manager(self, dummy_mlp: nn.Module) -> None:
        """Context manager attaches and detaches properly."""
        with Autopsy(dummy_mlp) as autopsy:
            assert autopsy._attached
            x = torch.randn(4, 10)
            y = dummy_mlp(x)
            loss = y.sum()
            loss.backward()
            autopsy.step(loss=loss.item())

        assert not autopsy._attached
        assert autopsy._report is not None

    def test_generates_html_report(self, dummy_mlp: nn.Module, tmp_path: Path) -> None:
        """Context manager -> train -> HTML report."""
        optimizer = torch.optim.SGD(dummy_mlp.parameters(), lr=0.01)

        with Autopsy(dummy_mlp, optimizer=optimizer) as autopsy:
            for _ in range(10):
                x = torch.randn(4, 10)
                optimizer.zero_grad()
                y = dummy_mlp(x)
                loss = y.sum()
                loss.backward()
                optimizer.step()
                autopsy.step(loss=loss.item())

        output = tmp_path / "test_report.html"
        report = autopsy.report(output)

        assert output.exists()
        assert report.metadata.model_name == "DummyMLP"
        assert report.metadata.total_steps == 10

    def test_generates_json_report(self, dummy_mlp: nn.Module, tmp_path: Path) -> None:
        """Context manager -> train -> JSON report."""
        with Autopsy(dummy_mlp) as autopsy:
            for _ in range(5):
                x = torch.randn(4, 10)
                y = dummy_mlp(x)
                loss = y.sum()
                loss.backward()
                autopsy.step(loss=loss.item())
                dummy_mlp.zero_grad()

        output = tmp_path / "test_report.json"
        autopsy.report(output)

        data = json.loads(output.read_text())
        assert data["metadata"]["model_name"] == "DummyMLP"
        assert data["metadata"]["total_steps"] == 5

    def test_infers_format_from_extension(self, dummy_mlp: nn.Module, tmp_path: Path) -> None:
        """Format is inferred from file extension."""
        with Autopsy(dummy_mlp) as autopsy:
            x = torch.randn(4, 10)
            y = dummy_mlp(x)
            y.sum().backward()
            autopsy.step(loss=1.0)
            dummy_mlp.zero_grad()

        json_path = tmp_path / "report.json"
        autopsy.report(json_path)
        data = json.loads(json_path.read_text())
        assert "metadata" in data

    def test_sampling_preset(self, dummy_mlp: nn.Module) -> None:
        """Sampling preset is passed through to collectors."""
        with Autopsy(dummy_mlp, sampling="fast") as autopsy:
            # Run 10 steps with fast sampling (interval=5), only 2 recorded
            for step in range(10):
                x = torch.randn(4, 10)
                y = dummy_mlp(x)
                y.sum().backward()
                autopsy.step(loss=1.0)
                dummy_mlp.zero_grad()

        report = autopsy.get_report()
        assert report is not None
        # Loss collector with interval=5 should record steps 0 and 5
        loss_state = report.collector_states.get("loss")
        assert loss_state is not None
        assert len(loss_state.steps) == 2

    def test_sampling_config_object(self, dummy_mlp: nn.Module) -> None:
        """SamplingConfig object is accepted."""
        config = SamplingConfig(default_interval=2, expensive_interval=10)
        with Autopsy(dummy_mlp, sampling=config) as autopsy:
            for step in range(6):
                x = torch.randn(4, 10)
                y = dummy_mlp(x)
                y.sum().backward()
                autopsy.step(loss=1.0)
                dummy_mlp.zero_grad()

        report = autopsy.get_report()
        assert report is not None
        loss_state = report.collector_states["loss"]
        assert len(loss_state.steps) == 3  # steps 0, 2, 4


class TestAutopsyExplicit:
    """Tests for explicit attach/detach usage."""

    def test_explicit_lifecycle(self, dummy_mlp: nn.Module) -> None:
        """Explicit attach -> step -> detach works."""
        autopsy = Autopsy(dummy_mlp)
        autopsy.attach()

        x = torch.randn(4, 10)
        y = dummy_mlp(x)
        y.sum().backward()
        autopsy.step(loss=1.0)
        dummy_mlp.zero_grad()

        autopsy.detach()
        assert autopsy.get_report() is not None

    def test_step_before_attach_is_noop(self, dummy_mlp: nn.Module) -> None:
        """Calling step() before attach() should silently do nothing."""
        autopsy = Autopsy(dummy_mlp)
        autopsy.step(loss=1.0)  # Should not raise

    def test_double_attach_warns(self, dummy_mlp: nn.Module) -> None:
        """Calling attach() twice should warn but not crash."""
        autopsy = Autopsy(dummy_mlp)
        autopsy.attach()
        autopsy.attach()  # Should log warning, not crash
        autopsy.detach()

    def test_report_before_detach_auto_detaches(
        self, dummy_mlp: nn.Module, tmp_path: Path
    ) -> None:
        """Calling report() while still attached should auto-detach."""
        autopsy = Autopsy(dummy_mlp)
        autopsy.attach()

        x = torch.randn(4, 10)
        y = dummy_mlp(x)
        y.sum().backward()
        autopsy.step(loss=1.0)
        dummy_mlp.zero_grad()

        output = tmp_path / "report.html"
        autopsy.report(output)
        assert output.exists()
        assert not autopsy._attached


class TestAutopsyMetadata:
    """Tests for metadata extraction."""

    def test_extracts_model_name(self, dummy_mlp: nn.Module) -> None:
        """Model class name is extracted."""
        with Autopsy(dummy_mlp) as autopsy:
            x = torch.randn(4, 10)
            dummy_mlp(x).sum().backward()
            autopsy.step()
            dummy_mlp.zero_grad()

        report = autopsy.get_report()
        assert report is not None
        assert report.metadata.model_name == "DummyMLP"

    def test_extracts_optimizer_info(self, dummy_mlp: nn.Module) -> None:
        """Optimizer name and LR are extracted."""
        optimizer = torch.optim.Adam(dummy_mlp.parameters(), lr=0.005)
        with Autopsy(dummy_mlp, optimizer=optimizer) as autopsy:
            x = torch.randn(4, 10)
            optimizer.zero_grad()
            dummy_mlp(x).sum().backward()
            optimizer.step()
            autopsy.step()

        report = autopsy.get_report()
        assert report is not None
        assert report.metadata.optimizer_name == "Adam"
        assert report.metadata.learning_rate == 0.005

    def test_total_parameters_counted(self, dummy_mlp: nn.Module) -> None:
        """Total parameter count is correct."""
        with Autopsy(dummy_mlp) as autopsy:
            x = torch.randn(4, 10)
            dummy_mlp(x).sum().backward()
            autopsy.step()
            dummy_mlp.zero_grad()

        report = autopsy.get_report()
        assert report is not None
        expected = sum(p.numel() for p in dummy_mlp.parameters())
        assert report.metadata.total_parameters == expected
