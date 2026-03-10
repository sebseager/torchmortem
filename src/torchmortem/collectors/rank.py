"""RankCollector -- estimates effective rank of activations via SVD."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchmortem.registry import register_collector
from torchmortem.types import CollectorCost, CollectorState, SamplingConfig


@register_collector
class RankCollector:
    """Estimates the effective rank of activations using SVD.

    Effective rank is based on the entropy of normalized singular values:
        eff_rank = exp(-sum p_i log(p_i))
    where p_i = sigma_i / sum sigma_j.

    A low effective rank indicates that the layer's representations are
    collapsing to a lower-dimensional subspace.

    This is an expensive collector, so runs periodically.

    Recorded metrics (per step, per layer):
        - ``effective_rank``: Effective rank of the activation matrix.
    """

    name: str = "rank"
    cost: CollectorCost = CollectorCost.EXPENSIVE

    def __init__(self) -> None:
        self._sampling: SamplingConfig | None = None
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._layer_names: list[str] = []
        self._layer_index: dict[str, int] = {}

        # Per-step staging
        self._current_ranks: dict[str, float] = {}

        # Buffers
        self._step_buf: list[int] = []
        self._rank_buf: list[list[float]] = []

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
            # Hook leaf modules with parameters
            if not list(module.children()) and any(
                p.requires_grad for p in module.parameters(recurse=False)
            ):
                self._layer_index[name] = len(self._layer_names)
                self._layer_names.append(name)
                handle = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(handle)

    def _make_hook(self, layer_name: str) -> Any:
        def hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor):
                return
            assert self._sampling is not None
            # We check sampling here since the hook fires every forward pass
            # but we only want to compute SVD periodically
            with torch.no_grad():
                # Reshape to 2D: (batch, features) for SVD
                out = output.detach().float()
                if out.dim() > 2:
                    out = out.reshape(out.shape[0], -1)
                elif out.dim() == 1:
                    out = out.unsqueeze(0)

                if out.shape[0] < 2 or out.shape[1] < 2:
                    return

                try:
                    # Compute SVD (only need singular values)
                    s = torch.linalg.svdvals(out)
                    # Compute effective rank
                    s_sum = s.sum()
                    if s_sum > 1e-30:
                        p = s / s_sum
                        # Avoid log(0)
                        p = p[p > 1e-30]
                        entropy = -(p * p.log()).sum().item()
                        eff_rank = float(np.exp(entropy))
                    else:
                        eff_rank = 0.0
                    self._current_ranks[layer_name] = eff_rank
                except Exception:
                    pass  # SVD can fail on degenerate matrices

        return hook

    def on_step(self, step: int, **kwargs: object) -> None:
        assert self._sampling is not None
        if not self._sampling.should_collect(self.name, self.cost, step):
            self._current_ranks.clear()
            return

        n_layers = len(self._layer_names)
        ranks = [0.0] * n_layers

        for lname, idx in self._layer_index.items():
            ranks[idx] = self._current_ranks.get(lname, 0.0)

        self._step_buf.append(step)
        self._rank_buf.append(ranks)
        self._current_ranks.clear()

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
                "effective_rank": np.array(self._rank_buf, dtype=np.float64),
            },
        )
