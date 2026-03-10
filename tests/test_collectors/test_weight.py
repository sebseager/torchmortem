"""Tests for WeightCollector."""

import torch
import torch.nn as nn
import numpy as np
import pytest

from torchmortem.collectors.weight import WeightCollector
from torchmortem.types import SamplingConfig


class TestWeightCollector:
    def _make_model(self):
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def test_attach_detach(self):
        model = self._make_model()
        collector = WeightCollector()
        sampling = SamplingConfig()
        collector.attach(model, None, sampling)
        assert len(collector._layer_names) == 2  # Two Linear layers
        collector.detach()
        assert len(collector._param_refs) == 0

    def test_records_weight_norms(self):
        model = self._make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        collector = WeightCollector()
        sampling = SamplingConfig()
        collector.attach(model, optimizer, sampling)

        x = torch.randn(4, 4)
        y = model(x)
        y.sum().backward()
        optimizer.step()
        collector.on_step(0)

        state = collector.state()
        assert state.name == "weight"
        assert len(state.steps) == 1
        assert "weight_norm" in state.series
        assert "update_norm" in state.series
        assert "update_ratio" in state.series
        collector.detach()

    def test_update_norm_nonzero_after_step(self):
        model = self._make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        collector = WeightCollector()
        sampling = SamplingConfig()
        collector.attach(model, optimizer, sampling)

        x = torch.randn(4, 4)
        y = model(x)
        y.sum().backward()
        optimizer.step()
        collector.on_step(0)

        state = collector.state()
        # After an optimizer step, update norms should be > 0
        assert (state.series["update_norm"] > 0).any()
        collector.detach()

    def test_multi_step_shapes(self):
        model = self._make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        collector = WeightCollector()
        sampling = SamplingConfig()
        collector.attach(model, optimizer, sampling)

        for step in range(5):
            x = torch.randn(4, 4)
            y = model(x)
            y.sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            collector.on_step(step)

        state = collector.state()
        n_layers = len(state.layers)
        assert state.series["weight_norm"].shape == (5, n_layers)
        assert state.series["update_norm"].shape == (5, n_layers)
        assert state.series["update_ratio"].shape == (5, n_layers)
        collector.detach()

    def test_sampling_interval(self):
        model = self._make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        collector = WeightCollector()
        sampling = SamplingConfig(default_interval=2)
        collector.attach(model, optimizer, sampling)

        for step in range(6):
            x = torch.randn(4, 4)
            y = model(x)
            y.sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            collector.on_step(step)

        state = collector.state()
        assert len(state.steps) == 3  # Steps 0, 2, 4
        assert list(state.steps) == [0, 2, 4]
        collector.detach()

    def test_update_ratio_reasonable(self):
        model = self._make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        collector = WeightCollector()
        sampling = SamplingConfig()
        collector.attach(model, optimizer, sampling)

        for step in range(3):
            x = torch.randn(4, 4)
            y = model(x)
            y.sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            collector.on_step(step)

        state = collector.state()
        ratios = state.series["update_ratio"]
        # With small LR, ratios should be small but finite
        assert (ratios >= 0).all()
        assert (ratios < 1).all()  # Ratio shouldn't be > 1 with lr=0.001
        collector.detach()
