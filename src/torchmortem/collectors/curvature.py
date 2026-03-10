"""CurvatureCollector -- estimates top Hessian eigenvalue via power iteration."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchmortem.registry import register_collector
from torchmortem.types import CollectorCost, CollectorState, SamplingConfig


@register_collector
class CurvatureCollector:
    """Estimates the top Hessian eigenvalue using Hessian-vector products
    via power iteration.

    This uses ``torch.autograd.functional.hvp`` to compute Hessian-vector
    products without materializing the full Hessian. The top eigenvalue
    (sharpness) is key for detecting edge-of-stability and catapult phenomena.

    Recorded metrics (per step, scalar):
        - ``top_eigenvalue``: Estimated largest eigenvalue of the Hessian.

    This is an expensive collector, so runs periodically (default: every 50 steps).
    """

    name: str = "curvature"
    cost: CollectorCost = CollectorCost.EXPENSIVE

    def __init__(self, power_iterations: int = 10) -> None:
        self._power_iterations = power_iterations
        self._sampling: SamplingConfig | None = None
        self._model: nn.Module | None = None
        self._last_loss_fn: Any = None

        # Buffers
        self._step_buf: list[int] = []
        self._eigenvalue_buf: list[float] = []

        # Cached random vector for power iteration continuity
        self._v: torch.Tensor | None = None

    def attach(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None,
        sampling: SamplingConfig,
    ) -> None:
        self._sampling = sampling
        self._model = model

    def detach(self) -> None:
        self._model = None
        self._v = None

    def on_step(self, step: int, **kwargs: object) -> None:
        assert self._sampling is not None
        if not self._sampling.should_collect(self.name, self.cost, step):
            return

        # We need a loss tensor (not .item()) to compute Hessian-vector products.
        loss_tensor = kwargs.get("loss_tensor")
        if loss_tensor is None or self._model is None:
            return

        if not isinstance(loss_tensor, torch.Tensor):
            return

        try:
            eigenvalue = self._estimate_top_eigenvalue(loss_tensor)
            self._step_buf.append(step)
            self._eigenvalue_buf.append(eigenvalue)
        except Exception:
            pass  # Silently skip if HVP fails (e.g., no grad graph)

    def _estimate_top_eigenvalue(self, loss: torch.Tensor) -> float:
        """Estimate top Hessian eigenvalue via power iteration."""
        assert self._model is not None
        params = [p for p in self._model.parameters() if p.requires_grad]
        if not params:
            return 0.0

        # Flatten all parameters into one vector
        param_shapes = [p.shape for p in params]
        total_params = sum(p.numel() for p in params)

        # Initialize or reuse random direction
        device = params[0].device
        if self._v is None or self._v.shape[0] != total_params:
            self._v = torch.randn(total_params, device=device)
            self._v = self._v / torch.linalg.vector_norm(self._v)

        v = self._v

        # Compute gradient
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        # Replace None grads (unused params) with zeros
        flat_parts = []
        for g, p in zip(grads, params):
            if g is None:
                flat_parts.append(torch.zeros(p.numel(), device=device))
            else:
                flat_parts.append(g.reshape(-1))
        flat_grad = torch.cat(flat_parts)

        for _ in range(self._power_iterations):
            # Hessian-vector product: Hv = d(grad^T v)/d(params)
            gv = torch.dot(flat_grad, v)
            hvp_list = torch.autograd.grad(gv, params, retain_graph=True, allow_unused=True)
            hv_parts = []
            for h, p in zip(hvp_list, params):
                if h is None:
                    hv_parts.append(torch.zeros(p.numel(), device=device))
                else:
                    hv_parts.append(h.reshape(-1))
            hv = torch.cat(hv_parts)

            eigenvalue = torch.dot(v, hv).item()
            # Normalize for next iteration (detach to avoid deep graph)
            norm = torch.linalg.vector_norm(hv).item()
            if norm > 1e-30:
                v = (hv / norm).detach()
            else:
                break

        self._v = v.detach()
        return abs(eigenvalue)

    def state(self) -> CollectorState:
        return CollectorState(
            name=self.name,
            steps=np.array(self._step_buf, dtype=np.int64),
            series={
                "top_eigenvalue": np.array(self._eigenvalue_buf, dtype=np.float64),
            },
        )
