"""WeightCollector -- records per-layer weight norms and update norms."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchmortem.registry import register_collector
from torchmortem.types import CollectorCost, CollectorState, SamplingConfig


@register_collector
class WeightCollector:
    """Collects per-layer weight statistics after each optimizer step.

    Recorded metrics (per layer, per step):
        - ``weight_norm``: L2 norm of the weight tensor.
        - ``update_norm``: L2 norm of the weight change from previous step.
        - ``update_ratio``: ``update_norm / weight_norm`` -- ~1e-3 for healthy training.
    """

    name: str = "weight"
    cost: CollectorCost = CollectorCost.CHEAP

    def __init__(self) -> None:
        self._sampling: SamplingConfig | None = None
        self._layer_names: list[str] = []
        self._layer_index: dict[str, int] = {}
        self._param_refs: dict[str, nn.Parameter] = {}

        # Previous weight snapshots for computing updates
        self._prev_weights: dict[str, torch.Tensor] = {}

        # Buffers
        self._step_buf: list[int] = []
        self._weight_norm_buf: list[list[float]] = []
        self._update_norm_buf: list[list[float]] = []
        self._update_ratio_buf: list[list[float]] = []

    def attach(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None,
        sampling: SamplingConfig,
    ) -> None:
        self._sampling = sampling
        self._layer_names.clear()
        self._layer_index.clear()
        self._param_refs.clear()
        self._prev_weights.clear()

        for name, module in model.named_modules():
            # Track leaf modules with weight parameters
            if not list(module.children()):
                weight = getattr(module, "weight", None)
                if weight is not None and isinstance(weight, nn.Parameter):
                    self._layer_index[name] = len(self._layer_names)
                    self._layer_names.append(name)
                    self._param_refs[name] = weight
                    # Snapshot initial weights
                    self._prev_weights[name] = weight.data.detach().clone()

    def detach(self) -> None:
        self._param_refs.clear()
        self._prev_weights.clear()

    def on_step(self, step: int, **kwargs: object) -> None:
        assert self._sampling is not None
        if not self._sampling.should_collect(self.name, self.cost, step):
            # Still update prev_weights so update_norm is accurate next time
            for lname, param in self._param_refs.items():
                self._prev_weights[lname] = param.data.detach().clone()
            return

        n_layers = len(self._layer_names)
        w_norms = [0.0] * n_layers
        u_norms = [0.0] * n_layers
        u_ratios = [0.0] * n_layers

        with torch.no_grad():
            for lname, idx in self._layer_index.items():
                param = self._param_refs[lname]
                w_norm = torch.linalg.vector_norm(param.data).item()
                w_norms[idx] = w_norm

                prev = self._prev_weights.get(lname)
                if prev is not None:
                    update = param.data - prev
                    u_norm = torch.linalg.vector_norm(update).item()
                    u_norms[idx] = u_norm
                    u_ratios[idx] = u_norm / w_norm if w_norm > 1e-30 else 0.0
                else:
                    u_norms[idx] = 0.0
                    u_ratios[idx] = 0.0

                # Snapshot current weight for next step
                self._prev_weights[lname] = param.data.detach().clone()

        self._step_buf.append(step)
        self._weight_norm_buf.append(w_norms)
        self._update_norm_buf.append(u_norms)
        self._update_ratio_buf.append(u_ratios)

    def state(self) -> CollectorState:
        return CollectorState(
            name=self.name,
            steps=np.array(self._step_buf, dtype=np.int64),
            layers=list(self._layer_names),
            series={
                "weight_norm": np.array(self._weight_norm_buf, dtype=np.float64),
                "update_norm": np.array(self._update_norm_buf, dtype=np.float64),
                "update_ratio": np.array(self._update_ratio_buf, dtype=np.float64),
            },
        )
