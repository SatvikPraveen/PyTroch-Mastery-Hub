"""Tests for CustomImageDataset argument validation."""

from __future__ import annotations

import pytest

from pytorch_mastery_hub.computer_vision.datasets import CustomImageDataset


def test_unknown_mode_rejected(tmp_path):
    csv = tmp_path / "labels.csv"
    csv.write_text("filename,label\na.png,cat\n")
    with pytest.raises(ValueError, match="mode"):
        CustomImageDataset(csv, tmp_path, mode="segmentation")


def test_classification_mode_builds_class_index(tmp_path):
    csv = tmp_path / "labels.csv"
    csv.write_text("filename,label\na.png,dog\nb.png,cat\n")
    ds = CustomImageDataset(csv, tmp_path)
    assert ds.class_to_idx == {"cat": 0, "dog": 1}
