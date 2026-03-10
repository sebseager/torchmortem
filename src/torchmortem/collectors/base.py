"""Collector protocol -- the interface all collectors implement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch.nn as nn
    import torch.optim as optim

    from torchmortem.types import CollectorCost, CollectorState, SamplingConfig


@runtime_checkable
class Collector(Protocol):
    """Protocol for data collectors.

    Collectors attach PyTorch hooks to a model and record summary statistics
    each step. They never store full tensors, only scalars or small arrays
    per layer per step.
    """

    name: str
    cost: CollectorCost

    def attach(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None,
        sampling: SamplingConfig,
    ) -> None:
        """Register hooks on the model/optimizer. Called once before training."""
        ...

    def detach(self) -> None:
        """Remove all hooks. Called after training ends."""
        ...

    def on_step(self, step: int, **kwargs: object) -> None:
        """Called after each training step. kwargs may contain loss, etc."""
        ...

    def state(self) -> CollectorState:
        """Return collected data as a CollectorState."""
        ...
