"""Tests for RankCollector."""

import torch
import torch.nn as nn
import numpy as np

from torchmortem.collectors.rank import RankCollector
from torchmortem.types import SamplingConfig


class TestRankCollector:
    def _make_model(self):
        return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    def test_attach_detach(self):
        model = self._make_model()
        collector = RankCollector()
        sampling = SamplingConfig(expensive_interval=1)
        collector.attach(model, None, sampling)
        assert len(collector._hooks) > 0
        collector.detach()
        assert len(collector._hooks) == 0

    def test_records_effective_rank(self):
        model = self._make_model()
        collector = RankCollector()
        sampling = SamplingConfig(expensive_interval=1)
        collector.attach(model, None, sampling)

        x = torch.randn(16, 8)  # Need batch > 1 for SVD
        model(x)
        collector.on_step(0)

        state = collector.state()
        assert state.name == "rank"
        assert len(state.steps) == 1
        assert "effective_rank" in state.series
        # Effective rank should be positive
        assert (state.series["effective_rank"] >= 0).all()
        collector.detach()

    def test_multi_step(self):
        model = self._make_model()
        collector = RankCollector()
        sampling = SamplingConfig(expensive_interval=1)
        collector.attach(model, None, sampling)

        for step in range(3):
            x = torch.randn(16, 8)
            model(x)
            collector.on_step(step)

        state = collector.state()
        n_layers = len(state.layers)
        assert state.series["effective_rank"].shape == (3, n_layers)
        collector.detach()

    def test_sampling_interval(self):
        model = self._make_model()
        collector = RankCollector()
        sampling = SamplingConfig(expensive_interval=3)
        collector.attach(model, None, sampling)

        for step in range(6):
            x = torch.randn(16, 8)
            model(x)
            collector.on_step(step)

        state = collector.state()
        assert len(state.steps) == 2  # Steps 0 and 3
        assert list(state.steps) == [0, 3]
        collector.detach()
