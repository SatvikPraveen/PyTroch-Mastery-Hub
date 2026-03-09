"""
Example: Transformer Text Classification
==========================================
Demonstrates using the TransformerClassifier from src.nlp.models for
binary sentiment classification on synthetic text data.
Run: python examples/transformer_text_classification.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.nlp.models import TransformerClassifier
from src.nlp.tokenization import SimpleTokenizer
from src.utils.metrics import accuracy, precision_recall_f1


# ── Synthetic data ────────────────────────────────────────────────
POSITIVE_SENTENCES = [
    "the model performs exceptionally well on all benchmarks",
    "excellent results with minimal training time",
    "outstanding performance across all evaluation metrics",
    "the architecture achieves state of the art accuracy",
    "impressive generalization to unseen data",
] * 20  # repeat to get enough samples

NEGATIVE_SENTENCES = [
    "the model struggles to converge during training",
    "poor performance on out of distribution samples",
    "high variance and unstable training behavior",
    "fails to generalize to the validation set",
    "the loss does not decrease after many epochs",
] * 20


def build_dataset(tokenizer, sentences, labels, max_len=20):
    """Tokenize sentences and return encoded tensors."""
    encoded = [tokenizer.encode(s) for s in sentences]
    # Pad or truncate to max_len
    padded = []
    for seq in encoded:
        if len(seq) >= max_len:
            padded.append(seq[:max_len])
        else:
            padded.append(seq + [tokenizer.vocab.get("<pad>", 0)] * (max_len - len(seq)))
    x = torch.tensor(padded, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)


def main():
    print("=" * 60)
    print("PyTorch Mastery Hub — Transformer Text Classification")
    print("=" * 60)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"\nDevice: {device}")

    # ── Tokenizer ────────────────────────────────────────────────
    print("\nBuilding vocabulary...")
    all_sentences = POSITIVE_SENTENCES + NEGATIVE_SENTENCES
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(all_sentences, min_freq=1)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # ── Dataset ──────────────────────────────────────────────────
    labels = [1] * len(POSITIVE_SENTENCES) + [0] * len(NEGATIVE_SENTENCES)
    dataset = build_dataset(tokenizer, all_sentences, labels, max_len=20)

    split = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [split, len(dataset) - split]
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    print(f"  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    # ── Model ────────────────────────────────────────────────────
    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=2,
        max_seq_len=20,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model.__class__.__name__}  |  Params: {total_params:,}")

    # ── Training ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining for 5 epochs...")
    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                logits = model(x_batch)
                preds = logits.argmax(dim=-1).cpu()
                all_preds.append(preds)
                all_targets.append(y_batch)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        val_acc = accuracy(all_preds, all_targets)
        p, r, f1 = precision_recall_f1(all_preds, all_targets, num_classes=2)

        print(f"  Epoch {epoch}/5  |  "
              f"train_loss: {total_loss / len(train_loader):.4f}  |  "
              f"val_acc: {val_acc:.2%}  |  "
              f"F1: {f1:.4f}")

    print("\n✓ Transformer classification example completed!")


if __name__ == "__main__":
    main()
