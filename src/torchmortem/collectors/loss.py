"""LossCollector -- records loss values passed via step()."""

from __future__ import annotations

import numpy as np
import torch.nn as nn
import torch.optim as optim

from torchmortem.registry import register_collector
from torchmortem.types import CollectorCost, CollectorState, SamplingConfig


@register_collector
class LossCollector:
    """Collects loss values provided by the user via ``autopsy.step(loss=...)``.

    Recorded metrics (per step):
        - ``loss``: Raw loss value.
        - ``loss_smoothed``: Exponential moving average (α=0.1).
        - ``loss_delta``: Difference from previous step.
    """

    name: str = "loss"
    cost: CollectorCost = CollectorCost.TRIVIAL

    def __init__(self, smoothing_alpha: float = 0.1) -> None:
        self._alpha = smoothing_alpha
        self._sampling: SamplingConfig | None = None

        self._step_buf: list[int] = []
        self._loss_buf: list[float] = []
        self._smoothed_buf: list[float] = []
        self._delta_buf: list[float] = []

        self._ema: float | None = None
        self._prev_loss: float | None = None

    def attach(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None,
        sampling: SamplingConfig,
    ) -> None:
        self._sampling = sampling

    def detach(self) -> None:
        pass  # No hooks to remove.

    def on_step(self, step: int, **kwargs: object) -> None:
        loss = kwargs.get("loss")
        if loss is None:
            return

        assert self._sampling is not None
        loss_val = float(loss)  # type: ignore[arg-type]

        # Always update EMA/delta even if we don't record this step,
        # so smoothed values remain accurate.
        if self._ema is None:
            self._ema = loss_val
        else:
            self._ema = self._alpha * loss_val + (1 - self._alpha) * self._ema

        delta = loss_val - self._prev_loss if self._prev_loss is not None else 0.0
        self._prev_loss = loss_val

        if not self._sampling.should_collect(self.name, self.cost, step):
            return

        self._step_buf.append(step)
        self._loss_buf.append(loss_val)
        self._smoothed_buf.append(self._ema)
        self._delta_buf.append(delta)

    def state(self) -> CollectorState:
        return CollectorState(
            name=self.name,
            steps=np.array(self._step_buf, dtype=np.int64),
            series={
                "loss": np.array(self._loss_buf, dtype=np.float64),
                "loss_smoothed": np.array(self._smoothed_buf, dtype=np.float64),
                "loss_delta": np.array(self._delta_buf, dtype=np.float64),
            },
        )
