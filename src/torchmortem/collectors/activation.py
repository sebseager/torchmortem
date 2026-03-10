"""ActivationCollector -- records per-layer activation statistics via forward hooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchmortem.registry import register_collector
from torchmortem.types import CollectorCost, CollectorState, SamplingConfig


@register_collector
class ActivationCollector:
    """Collects per-layer activation statistics via ``register_forward_hook``.

    Recorded metrics (per layer, per step):
        - ``act_mean``: Mean activation value.
        - ``act_std``: Standard deviation of activations.
        - ``act_dead_frac``: Fraction of activations that are near zero (|x| < threshold).
        - ``act_saturated_frac``: Fraction of activations near ±1 (|x| > 1 - threshold).
    """

    name: str = "activation"
    cost: CollectorCost = CollectorCost.CHEAP

    def __init__(
        self,
        dead_threshold: float = 1e-6,
        saturation_threshold: float = 0.05,
    ) -> None:
        self._dead_threshold = dead_threshold
        self._saturation_threshold = saturation_threshold

        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._sampling: SamplingConfig | None = None
        self._layer_names: list[str] = []
        self._layer_index: dict[str, int] = {}

        # Buffers
        self._step_buf: list[int] = []
        self._act_mean_buf: list[list[float]] = []
        self._act_std_buf: list[list[float]] = []
        self._act_dead_frac_buf: list[list[float]] = []
        self._act_saturated_frac_buf: list[list[float]] = []

        # Per-step staging (filled by hooks, flushed in on_step)
        self._current_means: dict[str, float] = {}
        self._current_stds: dict[str, float] = {}
        self._current_dead_fracs: dict[str, float] = {}
        self._current_saturated_fracs: dict[str, float] = {}

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
            # Hook activation-producing leaf modules (those that transform data).
            if not list(module.children()) and self._is_hookable(module):
                self._layer_index[name] = len(self._layer_names)
                self._layer_names.append(name)
                handle = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(handle)

    @staticmethod
    def _is_hookable(module: nn.Module) -> bool:
        """Determine if a module produces meaningful activations to track."""
        # Hook any leaf module that has parameters or is an activation function.
        activation_types = (
            nn.ReLU,
            nn.LeakyReLU,
            nn.PReLU,
            nn.ELU,
            nn.SELU,
            nn.GELU,
            nn.Sigmoid,
            nn.Tanh,
            nn.Softmax,
            nn.LogSoftmax,
            nn.SiLU,
            nn.Mish,
            nn.Hardswish,
            nn.Hardsigmoid,
        )
        if isinstance(module, activation_types):
            return True
        # Also hook modules with parameters (linear, conv, etc.)
        if any(True for _ in module.parameters(recurse=False)):
            return True
        return False

    def _make_hook(self, layer_name: str) -> Any:
        """Create a forward hook closure for a specific layer."""
        dead_thresh = self._dead_threshold
        sat_thresh = self._saturation_threshold

        def hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor):
                return
            with torch.no_grad():
                flat = output.detach().float()
                self._current_means[layer_name] = flat.mean().item()
                self._current_stds[layer_name] = flat.std().item()
                numel = flat.numel()
                if numel > 0:
                    self._current_dead_fracs[layer_name] = (
                        flat.abs() < dead_thresh
                    ).sum().item() / numel
                    self._current_saturated_fracs[layer_name] = (
                        flat.abs() > (1.0 - sat_thresh)
                    ).sum().item() / numel
                else:
                    self._current_dead_fracs[layer_name] = 0.0
                    self._current_saturated_fracs[layer_name] = 0.0

        return hook

    def on_step(self, step: int, **kwargs: object) -> None:
        assert self._sampling is not None
        if not self._sampling.should_collect(self.name, self.cost, step):
            self._current_means.clear()
            self._current_stds.clear()
            self._current_dead_fracs.clear()
            self._current_saturated_fracs.clear()
            return

        n_layers = len(self._layer_names)
        means = [0.0] * n_layers
        stds = [0.0] * n_layers
        dead_fracs = [0.0] * n_layers
        saturated_fracs = [0.0] * n_layers

        for lname, idx in self._layer_index.items():
            means[idx] = self._current_means.get(lname, 0.0)
            stds[idx] = self._current_stds.get(lname, 0.0)
            dead_fracs[idx] = self._current_dead_fracs.get(lname, 0.0)
            saturated_fracs[idx] = self._current_saturated_fracs.get(lname, 0.0)

        self._step_buf.append(step)
        self._act_mean_buf.append(means)
        self._act_std_buf.append(stds)
        self._act_dead_frac_buf.append(dead_fracs)
        self._act_saturated_frac_buf.append(saturated_fracs)

        self._current_means.clear()
        self._current_stds.clear()
        self._current_dead_fracs.clear()
        self._current_saturated_fracs.clear()

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
                "act_mean": np.array(self._act_mean_buf, dtype=np.float64),
                "act_std": np.array(self._act_std_buf, dtype=np.float64),
                "act_dead_frac": np.array(self._act_dead_frac_buf, dtype=np.float64),
                "act_saturated_frac": np.array(self._act_saturated_frac_buf, dtype=np.float64),
            },
        )
