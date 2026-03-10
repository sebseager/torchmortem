"""Shared test fixtures for torchmortem."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchmortem.types import SamplingConfig


class DummyMLP(nn.Module):
    """Simple MLP for testing, 3 linear layers with ReLU."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 16, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class DeepSigmoidMLP(nn.Module):
    """Deep MLP with sigmoid activations, designed to vanish gradients."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 16,
        output_dim: int = 1,
        num_hidden: int = 6,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Sigmoid())
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Sigmoid())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@pytest.fixture
def dummy_mlp() -> DummyMLP:
    return DummyMLP()


@pytest.fixture
def deep_sigmoid_mlp() -> DeepSigmoidMLP:
    return DeepSigmoidMLP()


@pytest.fixture
def sampling_every_step() -> SamplingConfig:
    return SamplingConfig(default_interval=1, expensive_interval=1)


def run_training_loop(
    model: nn.Module,
    num_steps: int = 50,
    lr: float = 0.01,
    input_dim: int = 10,
    batch_size: int = 16,
) -> tuple[torch.optim.Optimizer, list[float]]:
    """Run a simple training loop and return (optimizer, losses)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses: list[float] = []

    for _ in range(num_steps):
        x = torch.randn(batch_size, input_dim)
        y = torch.randn(batch_size, 1)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return optimizer, losses
