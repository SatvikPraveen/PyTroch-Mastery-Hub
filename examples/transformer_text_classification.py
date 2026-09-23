"""
Example: Transformer Text Classification
==========================================
Demonstrates using the TransformerClassifier from pytorch_mastery_hub.nlp.models for
binary sentiment classification on synthetic text data.
Run: python examples/transformer_text_classification.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_mastery_hub.nlp.models import TransformerClassifier
from pytorch_mastery_hub.nlp.tokenization import SimpleTokenizer
from pytorch_mastery_hub.utils.device_utils import get_device
from pytorch_mastery_hub.utils.metrics import accuracy, precision_recall_f1
from pytorch_mastery_hub.utils.model_utils import count_parameters
from pytorch_mastery_hub.utils.reproducibility import seed_everything

MAX_LEN = 16

# ── Synthetic data ────────────────────────────────────────────────
POSITIVE_SENTENCES = [
    "the model performs exceptionally well on all benchmarks",
    "excellent results with minimal training time",
    "outstanding performance across all evaluation metrics",
    "the architecture achieves state of the art accuracy",
    "impressive generalization to unseen data",
    "training converges quickly and the results are great",
] * 20  # repeat to get enough samples

NEGATIVE_SENTENCES = [
    "the model struggles to converge during training",
    "poor performance on out of distribution samples",
    "high variance and unstable training behavior",
    "fails to generalize to the validation set",
    "the loss does not decrease after many epochs",
    "disappointing accuracy and very slow training",
] * 20


def encode_batch(tokenizer, sentences, max_len=MAX_LEN):
    """Tokenize, add <BOS>/<EOS>, pad/truncate; return (input_ids, attention_mask)."""
    pad_id = tokenizer.word_to_idx["<PAD>"]
    ids = []
    for sentence in sentences:
        seq = tokenizer.encode(sentence)[:max_len]
        ids.append(seq + [pad_id] * (max_len - len(seq)))
    input_ids = torch.tensor(ids, dtype=torch.long)
    attention_mask = (input_ids != pad_id).long()  # 1 = real token, 0 = padding
    return input_ids, attention_mask


def main():
    print("=" * 60)
    print("PyTorch Mastery Hub — Transformer Text Classification")
    print("=" * 60)
    seed_everything(0)
    device = get_device()
    print(f"\nDevice: {device}")
    # The inference fast path of nn.TransformerEncoder needs an op MPS does not implement.
    torch.backends.mha.set_fastpath_enabled(False)

    # ── Tokenizer ────────────────────────────────────────────────
    print("\nBuilding vocabulary...")
    all_sentences = POSITIVE_SENTENCES + NEGATIVE_SENTENCES
    tokenizer = SimpleTokenizer(min_freq=1)
    tokenizer.build_vocab(all_sentences)
    print(f"  Vocabulary size: {len(tokenizer)}")
    print(f"  Example encoding: {tokenizer.encode(all_sentences[0])}")

    # ── Dataset ──────────────────────────────────────────────────
    labels = torch.tensor([1] * len(POSITIVE_SENTENCES) + [0] * len(NEGATIVE_SENTENCES))
    input_ids, attention_mask = encode_batch(tokenizer, all_sentences)
    dataset = TensorDataset(input_ids, attention_mask, labels)

    split = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    print(f"  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    # ── Model ────────────────────────────────────────────────────
    model = TransformerClassifier(
        vocab_size=len(tokenizer),
        d_model=64,
        num_heads=4,
        num_layers=2,
        num_classes=2,
        max_len=MAX_LEN,
        dropout=0.1,
    ).to(device)
    print(f"\nModel: {model.__class__.__name__}  |  Params: {count_parameters(model):,}")

    # ── Training ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining for 5 epochs...")
    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        for x_batch, mask_batch, y_batch in train_loader:
            x_batch, mask_batch, y_batch = (
                x_batch.to(device),
                mask_batch.to(device),
                y_batch.to(device),
            )
            optimizer.zero_grad()
            logits = model(x_batch, attention_mask=mask_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for x_batch, mask_batch, y_batch in val_loader:
                logits = model(x_batch.to(device), attention_mask=mask_batch.to(device))
                all_logits.append(logits.cpu())
                all_targets.append(y_batch)
        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        val_acc = accuracy(all_logits, all_targets)  # takes logits, returns a fraction
        _, _, f1 = precision_recall_f1(all_logits, all_targets, average="macro")

        print(
            f"  Epoch {epoch}/5  |  "
            f"train_loss: {total_loss / len(train_loader):.4f}  |  "
            f"val_acc: {val_acc:.2%}  |  "
            f"F1: {f1:.4f}"
        )

    # ── Inference on new sentences ───────────────────────────────
    print("\nPredictions on unseen sentences:")
    new_sentences = [
        "excellent accuracy and impressive results",
        "unstable training and poor accuracy",
    ]
    ids, mask = encode_batch(tokenizer, new_sentences)
    model.eval()
    with torch.no_grad():
        probs = model(ids.to(device), attention_mask=mask.to(device)).softmax(dim=-1).cpu()
    for sentence, p in zip(new_sentences, probs):
        label = "positive" if p[1] > 0.5 else "negative"
        print(f"  {label:<8} ({p[1]:.2f})  «{sentence}»")

    print("\n✓ Transformer classification example completed!")


if __name__ == "__main__":
    main()
