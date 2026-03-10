"""Tests for LossCollector."""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from torchmortem.collectors.loss import LossCollector
from torchmortem.types import SamplingConfig


class TestLossCollector:
    """Tests for LossCollector."""

    def test_records_loss_values(self) -> None:
        """Loss values passed via on_step are recorded."""
        collector = LossCollector()
        sampling = SamplingConfig(default_interval=1)
        collector.attach(nn.Linear(1, 1), None, sampling)

        losses = [2.5, 2.3, 2.1, 1.9, 1.7]
        for step, loss_val in enumerate(losses):
            collector.on_step(step, loss=loss_val)

        state = collector.state()
        assert state.name == "loss"
        assert len(state.steps) == 5
        np.testing.assert_array_almost_equal(state.series["loss"], losses)

    def test_smoothed_loss(self) -> None:
        """Smoothed loss follows EMA formula."""
        collector = LossCollector(smoothing_alpha=0.5)
        sampling = SamplingConfig(default_interval=1)
        collector.attach(nn.Linear(1, 1), None, sampling)

        collector.on_step(0, loss=10.0)
        collector.on_step(1, loss=0.0)

        state = collector.state()
        # Step 0: EMA = 10.0
        # Step 1: EMA = 0.5 * 0.0 + 0.5 * 10.0 = 5.0
        np.testing.assert_almost_equal(state.series["loss_smoothed"][0], 10.0)
        np.testing.assert_almost_equal(state.series["loss_smoothed"][1], 5.0)

    def test_loss_delta(self) -> None:
        """Loss delta is the difference from the previous step."""
        collector = LossCollector()
        sampling = SamplingConfig(default_interval=1)
        collector.attach(nn.Linear(1, 1), None, sampling)

        collector.on_step(0, loss=3.0)
        collector.on_step(1, loss=2.5)
        collector.on_step(2, loss=2.7)

        state = collector.state()
        np.testing.assert_almost_equal(state.series["loss_delta"][0], 0.0)  # no previous
        np.testing.assert_almost_equal(state.series["loss_delta"][1], -0.5)  # 2.5 - 3.0
        np.testing.assert_almost_equal(state.series["loss_delta"][2], 0.2)  # 2.7 - 2.5

    def test_no_loss_kwarg_skips(self) -> None:
        """Steps without loss= kwarg are silently skipped."""
        collector = LossCollector()
        sampling = SamplingConfig(default_interval=1)
        collector.attach(nn.Linear(1, 1), None, sampling)

        collector.on_step(0)  # No loss
        collector.on_step(1, loss=1.0)

        state = collector.state()
        assert len(state.steps) == 1
        assert state.steps[0] == 1

    def test_sampling_interval(self) -> None:
        """Respects sampling interval while maintaining accurate EMA."""
        collector = LossCollector(smoothing_alpha=0.5)
        sampling = SamplingConfig(default_interval=2)
        collector.attach(nn.Linear(1, 1), None, sampling)

        # Steps 0,1,2,3,4 but only 0,2,4 should be recorded
        for i in range(5):
            collector.on_step(i, loss=float(i))

        state = collector.state()
        assert len(state.steps) == 3
        np.testing.assert_array_equal(state.steps, [0, 2, 4])

        # But EMA should still account for ALL steps (not just recorded ones)
        # Step 0: EMA = 0.0
        # Step 1: EMA = 0.5*1 + 0.5*0 = 0.5 (not recorded)
        # Step 2: EMA = 0.5*2 + 0.5*0.5 = 1.25 (recorded)
        np.testing.assert_almost_equal(state.series["loss_smoothed"][0], 0.0)
        np.testing.assert_almost_equal(state.series["loss_smoothed"][1], 1.25)

    def test_detach_is_noop(self) -> None:
        """Detach is a no-op (no hooks to remove)."""
        collector = LossCollector()
        sampling = SamplingConfig(default_interval=1)
        collector.attach(nn.Linear(1, 1), None, sampling)
        collector.detach()  # Should not raise
