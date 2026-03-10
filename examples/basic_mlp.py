"""basic_mlp.py -- Intentionally pathological MLP to demonstrate torchmortem.

This trains a deep MLP on synthetic data with bad hyperparameters:
- 8 hidden layers (too deep for a vanilla MLP without skip connections)
- Sigmoid activations (prone to saturation, vanishing gradients)
- Default initialization (not suited for deep sigmoids)
- Relatively high learning rate

We expect vanishing gradients (early layers receive ~0 gradient signal), and
possibly gradient stalling in the first few layers.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchmortem import Autopsy


def build_pathological_mlp(
    input_dim: int = 20,
    hidden_dim: int = 64,
    output_dim: int = 1,
    num_hidden: int = 8,
) -> nn.Module:
    """Build deep MLP with sigmoid activations, should exhibit vanishing gradients."""
    layers: list[nn.Module] = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.Sigmoid())
    for _ in range(num_hidden - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Sigmoid())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def make_synthetic_data(
    n_samples: int = 512,
    input_dim: int = 20,
) -> TensorDataset:
    """Generate a simple regression dataset."""
    X = torch.randn(n_samples, input_dim)
    # Target is a nonlinear function of the first 3 features.
    y = (X[:, 0] * X[:, 1] + torch.sin(X[:, 2])).unsqueeze(-1)
    return TensorDataset(X, y)


def main() -> None:
    # Setup
    model = build_pathological_mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    dataset = make_synthetic_data()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 10

    # Train with Autopsy attached
    with Autopsy(model, optimizer=optimizer, sampling="thorough") as autopsy:
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
    report = autopsy.report("basic_mlp_report.html")
    autopsy.report("basic_mlp_report.json", fmt="json")

    print(f"\n{'=' * 50}")
    print("Executive Summary:")
    print(report.executive_summary)
    print(f"\nFindings: {len(report.findings)}")
    for f in report.findings:
        print(f"  [{f.severity.name}] {f.title}")
    if report.insights:
        print(f"\nInsights: {len(report.insights)}")
        for i in report.insights:
            print(f"  💡 {i.title}")
    if report.health_scores:
        min_hs = min(report.health_scores, key=lambda hs: hs.score)
        print(f"\nUnhealthiest layer: {min_hs.layer_name} ({min_hs.score:.0%})")


if __name__ == "__main__":
    main()
