"""cnn_overfit.py -- Small CNN that overfits on a tiny synthetic image dataset.

This trains a small convolutional network on very few samples (64 training images)
with no regularization (dropout, weight decay) and a moderate learning rate.

We expect the model to memorize, so loss may plateau early or drop near zero
(overfit), weight norms may grow, and some units may become dead.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchmortem import Autopsy


class TinyCNN(nn.Module):
    """Small convolutional network for image classification."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def make_synthetic_images(
    n_samples: int = 64,
    num_classes: int = 5,
    img_size: int = 16,
) -> TensorDataset:
    """Generate a tiny synthetic image classification dataset."""
    X = torch.randn(n_samples, 3, img_size, img_size)
    y = torch.randint(0, num_classes, (n_samples,))
    return TensorDataset(X, y)


def main() -> None:
    model = TinyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = make_synthetic_images()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    num_epochs = 30

    with Autopsy(model, optimizer=optimizer, sampling="thorough") as autopsy:
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                autopsy.step(loss=loss.item())
                epoch_loss += loss.item()
                correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total += y_batch.size(0)

            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct / total
            print(
                f"Epoch {epoch + 1}/{num_epochs} -- loss: {avg_loss:.4f}, accuracy: {accuracy:.1%}"
            )

    # Generate reports
    report = autopsy.report("cnn_overfit_report.html")
    autopsy.report("cnn_overfit_report.json", fmt="json")

    print(f"\n{'=' * 50}")
    print("Executive Summary:")
    print(report.executive_summary)
    print(f"\nFindings: {len(report.findings)}")
    for f in report.findings:
        print(f"  [{f.severity.name}] {f.title}")
    if report.insights:
        print(f"\nInsights: {len(report.insights)}")
        for i in report.insights:
            print(f"  * {i.title}")
    if report.health_scores:
        min_hs = min(report.health_scores, key=lambda hs: hs.score)
        print(f"\nUnhealthiest layer: {min_hs.layer_name} ({min_hs.score:.0%})")


if __name__ == "__main__":
    main()
