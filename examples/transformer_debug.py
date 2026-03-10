"""transformer_debug.py -- Small Transformer with common training issues.

This trains a minimal Transformer encoder on a synthetic sequence
classification task with intentionally problematic settings. We don't use
learning rate warmup, set a high learning rate, and don't use gradient clipping.

We could see exploding gradients in early steps, weight norm growth, and
potentially edge-of-stability behavior.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchmortem import Autopsy


class TinyTransformer(nn.Module):
    """Minimal Transformer encoder for sequence classification."""

    def __init__(
        self,
        vocab_size: int = 100,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        num_classes: int = 5,
        max_len: int = 32,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.encoder(x)
        # Mean pooling over sequence
        x = x.mean(dim=1)
        return self.classifier(x)


def make_synthetic_data(
    n_samples: int = 512,
    vocab_size: int = 100,
    seq_len: int = 32,
    num_classes: int = 5,
) -> TensorDataset:
    """Generate a synthetic sequence classification dataset."""
    X = torch.randint(0, vocab_size, (n_samples, seq_len))
    # Labels based on count of token 0 in sequence (arbitrary rule)
    y = (X == 0).sum(dim=1) % num_classes
    return TensorDataset(X, y)


def main() -> None:
    model = TinyTransformer()
    # Intentionally high LR for Transformers, no warmup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    dataset = make_synthetic_data()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 10

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
                # No gradient clipping, intentional
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
    report = autopsy.report("transformer_debug_report.html")
    autopsy.report("transformer_debug_report.json", fmt="json")

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


if __name__ == "__main__":
    main()
