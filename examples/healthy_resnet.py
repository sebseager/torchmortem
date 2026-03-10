"""healthy_resnet.py -- Well-configured model demonstrating a clean report.

This trains a small residual network on synthetic data with good practices,
including skip connections (residual blocks), ReLU activations, Adam optimizer
with moderate learning rate, and batch normalization.

We expect minimal to no issues.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchmortem import Autopsy


class ResidualBlock(nn.Module):
    """Simple residual block with batch norm."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class SmallResNet(nn.Module):
    """Small residual network for regression."""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, num_blocks: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

        # He initialization for ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_proj(x))
        x = self.blocks(x)
        return self.head(x)


def make_synthetic_data(n_samples: int = 1024, input_dim: int = 20) -> TensorDataset:
    """Generate a simple regression dataset."""
    X = torch.randn(n_samples, input_dim)
    y = (X[:, 0] * X[:, 1] + torch.sin(X[:, 2])).unsqueeze(-1)
    return TensorDataset(X, y)


def main() -> None:
    model = SmallResNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    dataset = make_synthetic_data()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    num_epochs = 15

    with Autopsy(model, optimizer=optimizer, sampling="balanced") as autopsy:
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                autopsy.step(loss=loss.item())
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs} -- loss: {avg_loss:.4f}")

    # Generate reports
    report = autopsy.report("healthy_resnet_report.html")
    autopsy.report("healthy_resnet_report.json", fmt="json")

    print(f"\n{'=' * 50}")
    print("Executive Summary:")
    print(report.executive_summary)
    print(f"\nFindings: {len(report.findings)}")
    print(f"Insights: {len(report.insights)}")

    if report.health_scores:
        min_score = min(hs.score for hs in report.health_scores)
        print(f"Min layer health: {min_score:.0%}")


if __name__ == "__main__":
    main()
