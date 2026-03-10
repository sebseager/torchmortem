"""lstm_vanishing.py -- Vanilla LSTM trained on long sequences.

This trains an LSTM on a synthetic sequence task where the label depends
on the very first token, forcing the network to propagate information
across the full sequence length. We use long sequences (length 200), no
gradient clipping, and a relatively deep architecture (3 stacked LSTM layers).

We expect gradient flow issues (vanishing gradients through time), and potentially
low update ratios in early layers.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchmortem import Autopsy


class StackedLSTM(nn.Module):
    """Stacked LSTM for sequence classification."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        output, _ = self.lstm(x)
        # Use the last hidden state for classification
        return self.classifier(output[:, -1, :])


def make_long_sequence_data(
    n_samples: int = 256,
    seq_len: int = 200,
    input_dim: int = 10,
    num_classes: int = 3,
) -> TensorDataset:
    """Generate sequences where the label depends on the first time step.

    This forces the LSTM to propagate information across the full
    sequence length, which is hard for vanilla LSTMs on long sequences.
    """
    X = torch.randn(n_samples, seq_len, input_dim) * 0.1
    # Label is determined by the first time step's first feature
    first_val = X[:, 0, 0]
    y = torch.zeros(n_samples, dtype=torch.long)
    y[first_val > 0.05] = 1
    y[first_val > 0.1] = 2
    return TensorDataset(X, y)


def main() -> None:
    model = StackedLSTM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    dataset = make_long_sequence_data()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 15

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
                # No gradient clipping -- intentional
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
    report = autopsy.report("lstm_vanishing_report.html")
    autopsy.report("lstm_vanishing_report.json", fmt="json")

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
