"""GradientCollector -- records per-layer gradient norms using backward hooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchmortem.registry import register_collector
from torchmortem.types import CollectorCost, CollectorState, SamplingConfig


@register_collector
class GradientCollector:
    """Collects per-layer gradient L2 norms via ``register_full_backward_hook``.

    Recorded metrics (per layer, per step):
        - ``grad_norm``: L2 norm of the gradient tensor.
        - ``grad_mean``: Mean of the gradient tensor.
        - ``grad_max``: Max absolute value of the gradient tensor.
        - ``grad_zero_frac``: Fraction of gradient elements that are exactly zero.
    """

    name: str = "gradient"
    cost: CollectorCost = CollectorCost.CHEAP

    def __init__(self) -> None:
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._sampling: SamplingConfig | None = None
        self._layer_names: list[str] = []
        self._layer_index: dict[str, int] = {}

        # Buffers: accumulated per-step, flushed to numpy at end.
        self._step_buf: list[int] = []
        self._grad_norm_buf: list[list[float]] = []
        self._grad_mean_buf: list[list[float]] = []
        self._grad_max_buf: list[list[float]] = []
        self._grad_zero_frac_buf: list[list[float]] = []

        # Per-step staging area (filled by hooks, flushed by on_step).
        self._current_norms: dict[str, float] = {}
        self._current_means: dict[str, float] = {}
        self._current_maxs: dict[str, float] = {}
        self._current_zero_fracs: dict[str, float] = {}

    def attach(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None,
        sampling: SamplingConfig,
    ) -> None:
        self._sampling = sampling
        self._hooks.clear()
        self._layer_names.clear()
        self._layer_index.clear()

        for name, module in model.named_modules():
            # Only hook leaf modules that have parameters.
            if not list(module.children()) and any(
                p.requires_grad for p in module.parameters(recurse=False)
            ):
                self._layer_index[name] = len(self._layer_names)
                self._layer_names.append(name)

                handle = module.register_full_backward_hook(self._make_hook(name))
                self._hooks.append(handle)

    def _make_hook(self, layer_name: str) -> Any:
        """Create a backward hook closure for a specific layer."""

        def hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor | None, ...],
            grad_output: tuple[torch.Tensor | None, ...],
        ) -> None:
            # Use grad_output (gradient w.r.t. output of this layer).
            grad = grad_output[0]
            if grad is None:
                return
            with torch.no_grad():
                self._current_norms[layer_name] = torch.linalg.vector_norm(grad).item()
                self._current_means[layer_name] = grad.mean().item()
                self._current_maxs[layer_name] = grad.abs().max().item()
                numel = grad.numel()
                zero_count = (grad == 0).sum().item()
                self._current_zero_fracs[layer_name] = zero_count / numel if numel > 0 else 0.0

        return hook

    def on_step(self, step: int, **kwargs: object) -> None:
        assert self._sampling is not None
        if not self._sampling.should_collect(self.name, self.cost, step):
            self._current_norms.clear()
            self._current_means.clear()
            self._current_maxs.clear()
            self._current_zero_fracs.clear()
            return

        n_layers = len(self._layer_names)
        norms = [0.0] * n_layers
        means = [0.0] * n_layers
        maxs = [0.0] * n_layers
        zero_fracs = [0.0] * n_layers

        for lname, idx in self._layer_index.items():
            norms[idx] = self._current_norms.get(lname, 0.0)
            means[idx] = self._current_means.get(lname, 0.0)
            maxs[idx] = self._current_maxs.get(lname, 0.0)
            zero_fracs[idx] = self._current_zero_fracs.get(lname, 0.0)

        self._step_buf.append(step)
        self._grad_norm_buf.append(norms)
        self._grad_mean_buf.append(means)
        self._grad_max_buf.append(maxs)
        self._grad_zero_frac_buf.append(zero_fracs)

        self._current_norms.clear()
        self._current_means.clear()
        self._current_maxs.clear()
        self._current_zero_fracs.clear()

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def state(self) -> CollectorState:
        return CollectorState(
            name=self.name,
            steps=np.array(self._step_buf, dtype=np.int64),
            layers=list(self._layer_names),
            series={
                "grad_norm": np.array(self._grad_norm_buf, dtype=np.float64),
                "grad_mean": np.array(self._grad_mean_buf, dtype=np.float64),
                "grad_max": np.array(self._grad_max_buf, dtype=np.float64),
                "grad_zero_frac": np.array(self._grad_zero_frac_buf, dtype=np.float64),
            },
        )
