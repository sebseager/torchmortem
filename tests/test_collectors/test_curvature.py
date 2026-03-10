"""Tests for CurvatureCollector."""

import torch
import torch.nn as nn
import numpy as np
import pytest

from torchmortem.collectors.curvature import CurvatureCollector
from torchmortem.types import SamplingConfig


class TestCurvatureCollector:
    def _make_model_and_loss(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        return model

    def test_attach_detach(self):
        model = self._make_model_and_loss()
        collector = CurvatureCollector()
        sampling = SamplingConfig()
        collector.attach(model, None, sampling)
        assert collector._model is model
        collector.detach()
        assert collector._model is None

    def test_records_eigenvalue_with_loss_tensor(self):
        model = self._make_model_and_loss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        collector = CurvatureCollector(power_iterations=5)
        sampling = SamplingConfig(expensive_interval=1)
        collector.attach(model, optimizer, sampling)

        x = torch.randn(8, 4)
        y = model(x)
        loss = y.sum()
        # Pass loss_tensor (not .item()) for HVP
        collector.on_step(0, loss_tensor=loss)

        state = collector.state()
        assert state.name == "curvature"
        # Should have one measurement
        assert len(state.steps) == 1
        assert "top_eigenvalue" in state.series
        assert state.series["top_eigenvalue"][0] >= 0
        collector.detach()

    def test_skips_without_loss_tensor(self):
        model = self._make_model_and_loss()
        collector = CurvatureCollector()
        sampling = SamplingConfig(expensive_interval=1)
        collector.attach(model, None, sampling)

        # Don't pass loss_tensor
        collector.on_step(0)

        state = collector.state()
        assert len(state.steps) == 0
        collector.detach()

    def test_respects_sampling(self):
        model = self._make_model_and_loss()
        collector = CurvatureCollector(power_iterations=3)
        sampling = SamplingConfig(expensive_interval=5)
        collector.attach(model, None, sampling)

        for step in range(10):
            x = torch.randn(4, 4)
            y = model(x)
            loss = y.sum()
            collector.on_step(step, loss_tensor=loss)

        state = collector.state()
        # Should record at steps 0 and 5
        assert len(state.steps) == 2
        assert list(state.steps) == [0, 5]
        collector.detach()
