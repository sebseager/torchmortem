"""Tests for ActivationCollector."""

import torch
import torch.nn as nn
import numpy as np
import pytest

from torchmortem.collectors.activation import ActivationCollector
from torchmortem.types import SamplingConfig


class TestActivationCollector:
    def _make_model(self):
        """Simple model with activation function."""
        return nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Sigmoid(),
            nn.Linear(4, 2),
        )

    def test_attach_detach(self):
        model = self._make_model()
        collector = ActivationCollector()
        sampling = SamplingConfig()
        collector.attach(model, None, sampling)
        assert len(collector._hooks) > 0
        assert len(collector._layer_names) > 0
        collector.detach()
        assert len(collector._hooks) == 0

    def test_records_activations(self):
        model = self._make_model()
        collector = ActivationCollector()
        sampling = SamplingConfig()
        collector.attach(model, None, sampling)

        x = torch.randn(8, 4)
        model(x)
        collector.on_step(0)

        state = collector.state()
        assert state.name == "activation"
        assert len(state.steps) == 1
        assert "act_mean" in state.series
        assert "act_std" in state.series
        assert "act_dead_frac" in state.series
        assert "act_saturated_frac" in state.series
        collector.detach()

    def test_multi_step_shapes(self):
        model = self._make_model()
        collector = ActivationCollector()
        sampling = SamplingConfig()
        collector.attach(model, None, sampling)

        x = torch.randn(8, 4)
        for step in range(5):
            model(x)
            collector.on_step(step)

        state = collector.state()
        n_layers = len(state.layers)
        assert state.series["act_mean"].shape == (5, n_layers)
        assert state.series["act_std"].shape == (5, n_layers)
        collector.detach()

    def test_sampling_interval(self):
        model = self._make_model()
        collector = ActivationCollector()
        sampling = SamplingConfig(default_interval=3)
        collector.attach(model, None, sampling)

        x = torch.randn(8, 4)
        for step in range(6):
            model(x)
            collector.on_step(step)

        state = collector.state()
        # Steps 0 and 3 should be recorded (every 3 steps)
        assert len(state.steps) == 2
        assert list(state.steps) == [0, 3]
        collector.detach()

    def test_dead_fraction_bounds(self):
        model = self._make_model()
        collector = ActivationCollector()
        sampling = SamplingConfig()
        collector.attach(model, None, sampling)

        x = torch.randn(8, 4)
        model(x)
        collector.on_step(0)

        state = collector.state()
        dead = state.series["act_dead_frac"]
        assert (dead >= 0).all()
        assert (dead <= 1).all()
        collector.detach()

    def test_saturated_fraction_bounds(self):
        model = self._make_model()
        collector = ActivationCollector()
        sampling = SamplingConfig()
        collector.attach(model, None, sampling)

        x = torch.randn(8, 4)
        model(x)
        collector.on_step(0)

        state = collector.state()
        sat = state.series["act_saturated_frac"]
        assert (sat >= 0).all()
        assert (sat <= 1).all()
        collector.detach()

    def test_hooks_activation_layers(self):
        """Verify that activation layers (ReLU, Sigmoid) are hooked."""
        model = self._make_model()
        collector = ActivationCollector()
        sampling = SamplingConfig()
        collector.attach(model, None, sampling)

        # Should hook: Linear(4,8), ReLU, Linear(8,4), Sigmoid, Linear(4,2)
        assert len(collector._layer_names) >= 3  # at least the linears + activations
        collector.detach()
