"""Tests for GradientCollector."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from torchmortem.collectors.gradient import GradientCollector
from torchmortem.types import SamplingConfig


class TestGradientCollector:
    """Tests for GradientCollector."""

    def test_attach_detach(self, dummy_mlp: nn.Module) -> None:
        """Collector attaches hooks and detaches cleanly."""
        collector = GradientCollector()
        sampling = SamplingConfig(default_interval=1)

        collector.attach(dummy_mlp, None, sampling)
        assert len(collector._hooks) > 0

        collector.detach()
        assert len(collector._hooks) == 0

    def test_records_gradient_norms(self, dummy_mlp: nn.Module) -> None:
        """After forward+backward, gradient norms are recorded."""
        collector = GradientCollector()
        sampling = SamplingConfig(default_interval=1)
        collector.attach(dummy_mlp, None, sampling)

        # Forward + backward
        x = torch.randn(4, 10)
        y = dummy_mlp(x)
        loss = y.sum()
        loss.backward()

        collector.on_step(0)
        collector.detach()

        state = collector.state()
        assert state.name == "gradient"
        assert len(state.steps) == 1
        assert state.steps[0] == 0
        assert len(state.layers) > 0
        assert "grad_norm" in state.series
        assert "grad_mean" in state.series
        assert "grad_max" in state.series
        assert "grad_zero_frac" in state.series

        # Shape: (1 step, N layers)
        assert state.series["grad_norm"].shape == (1, len(state.layers))

        # All gradient norms should be positive (something flowed)
        assert (state.series["grad_norm"] > 0).all()

    def test_multiple_steps(self, dummy_mlp: nn.Module) -> None:
        """Records data across multiple training steps."""
        collector = GradientCollector()
        sampling = SamplingConfig(default_interval=1)
        collector.attach(dummy_mlp, None, sampling)

        num_steps = 5
        for step in range(num_steps):
            x = torch.randn(4, 10)
            y = dummy_mlp(x)
            loss = y.sum()
            loss.backward()
            collector.on_step(step)
            dummy_mlp.zero_grad()

        collector.detach()
        state = collector.state()

        assert len(state.steps) == num_steps
        assert state.series["grad_norm"].shape == (num_steps, len(state.layers))

    def test_sampling_respects_interval(self, dummy_mlp: nn.Module) -> None:
        """Collector skips steps when sampling interval > 1."""
        collector = GradientCollector()
        sampling = SamplingConfig(default_interval=3)  # Record every 3rd step
        collector.attach(dummy_mlp, None, sampling)

        for step in range(9):
            x = torch.randn(4, 10)
            y = dummy_mlp(x)
            loss = y.sum()
            loss.backward()
            collector.on_step(step)
            dummy_mlp.zero_grad()

        collector.detach()
        state = collector.state()

        # Steps 0, 3, 6 should be recorded
        assert len(state.steps) == 3
        np.testing.assert_array_equal(state.steps, [0, 3, 6])

    def test_only_hooks_parameterized_leaves(self, dummy_mlp: nn.Module) -> None:
        """Only leaf modules with parameters get hooks (not ReLU, not Sequential)."""
        collector = GradientCollector()
        sampling = SamplingConfig(default_interval=1)
        collector.attach(dummy_mlp, None, sampling)

        # DummyMLP has fc1, fc2, fc3 (Linear layers)
        assert len(collector._layer_names) == 3
        for name in collector._layer_names:
            assert "fc" in name  # Should be fc1, fc2, fc3

        collector.detach()

    def test_grad_zero_frac_reasonable(self, dummy_mlp: nn.Module) -> None:
        """Gradient zero fraction should be between 0 and 1."""
        collector = GradientCollector()
        sampling = SamplingConfig(default_interval=1)
        collector.attach(dummy_mlp, None, sampling)

        x = torch.randn(4, 10)
        y = dummy_mlp(x)
        loss = y.sum()
        loss.backward()
        collector.on_step(0)
        collector.detach()

        state = collector.state()
        zero_fracs = state.series["grad_zero_frac"]
        assert (zero_fracs >= 0).all()
        assert (zero_fracs <= 1).all()
