"""Tests for the pytorch-hub CLI."""

from __future__ import annotations

import json

import pytest
import yaml

from pytorch_mastery_hub.utils import cli


def test_help_and_no_command(capsys):
    assert cli.main([]) == 0
    assert "pytorch-hub" in capsys.readouterr().out


def test_info(capsys):
    assert cli.main(["info"]) == 0
    out = capsys.readouterr().out
    assert "torch" in out and "device" in out


def test_config_merging(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"trainer": {"epochs": 2}, "model": {"hidden_sizes": [8]}}))
    cfg = cli.load_train_config(p, {"trainer": {"epochs": None, "precision": "bf16"}, "seed": 7})
    assert cfg["trainer"]["epochs"] == 2  # None override is ignored
    assert cfg["trainer"]["precision"] == "bf16"
    assert cfg["model"]["hidden_sizes"] == [8] and cfg["model"]["type"] == "mlp"
    assert cfg["seed"] == 7


def test_train_synthetic_end_to_end(tmp_path):
    out = tmp_path / "run"
    rc = cli.main(
        [
            "train",
            "--epochs",
            "2",
            "--device",
            "cpu",
            "--output-dir",
            str(out),
            "--quiet",
            "--ema",
            "0.9",
            "--seed",
            "1",
        ]
    )
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["epochs_run"] == 2 and (out / "best.pt").exists()
    lines = (out / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2 and "val_loss" in json.loads(lines[0])
    assert (out / "train.log").read_text()


def test_train_with_config_file(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "o"),
                "data": {"n_samples": 256, "batch_size": 32},
                "trainer": {"epochs": 1},
                "scheduler": {"type": "onecycle"},
                "optimizer": {"type": "sgd", "lr": 0.05},
            }
        )
    )
    assert cli.main(["train", "--config", str(cfg), "--device", "cpu", "--quiet"]) == 0
    assert (tmp_path / "o" / "summary.json").exists()


def test_train_bad_model(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"model": {"type": "rnn"}, "output_dir": str(tmp_path)}))
    with pytest.raises(ValueError):
        cli.main(["train", "--config", str(cfg), "--device", "cpu", "--quiet"])


def test_benchmark_small(capsys):
    assert (
        cli.main(
            ["benchmark", "--sizes", "64", "--seq-lens", "32", "--iters", "2", "--device", "cpu"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "matmul" in out and "sdpa" in out
