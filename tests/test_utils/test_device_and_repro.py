"""Tests for device_utils, reproducibility, logging_utils, model_utils, memory_utils."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from torch import nn

from pytorch_mastery_hub.utils.device_utils import (
    autocast_dtype,
    device_info,
    get_device,
    move_to_device,
    synchronize,
)
from pytorch_mastery_hub.utils.logging_utils import MetricsLogger, get_logger, setup_logger
from pytorch_mastery_hub.utils.memory_utils import (
    MemoryTracker,
    clear_memory,
    get_memory_usage,
    tensor_bytes,
)
from pytorch_mastery_hub.utils.model_utils import (
    count_parameters,
    freeze,
    get_model_size,
    init_weights,
    model_summary,
    param_groups_with_weight_decay,
    summarize,
    unfreeze,
)
from pytorch_mastery_hub.utils.reproducibility import (
    capture_rng_state,
    isolated_rng,
    make_generator,
    restore_rng_state,
    seed_everything,
    seed_worker,
)

# --------------------------------------------------------------------------- device


class TestDevice:
    def test_get_device_returns_device(self):
        dev = get_device()
        assert isinstance(dev, torch.device)
        assert dev.type in {"cpu", "cuda", "mps"}

    def test_explicit_cpu(self):
        assert get_device("cpu") == torch.device("cpu")

    def test_unavailable_accelerator_falls_back(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert get_device("cuda").type in {"cpu", "mps"}

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_HUB_DEVICE", "cpu")
        assert get_device() == torch.device("cpu")

    def test_autocast_dtype_cpu(self):
        assert autocast_dtype("cpu") == torch.bfloat16

    def test_move_nested_structures(self):
        @dataclass
        class Batch:
            x: torch.Tensor
            meta: dict

        batch = {
            "a": torch.zeros(2),
            "b": [torch.ones(1), (torch.ones(1), "keep")],
            "c": Batch(torch.zeros(1), {"id": 7}),
        }
        moved = move_to_device(batch, "cpu")
        assert moved["a"].device.type == "cpu"
        assert moved["b"][1][1] == "keep"
        assert moved["c"].meta["id"] == 7

    def test_device_info_keys(self):
        info = device_info()
        assert {"torch", "device", "cuda_available"} <= info.keys()

    def test_synchronize_noop_on_cpu(self):
        synchronize("cpu")


# ------------------------------------------------------------------ reproducibility


class TestReproducibility:
    def test_seed_everything_makes_torch_deterministic(self):
        seed_everything(123)
        a = torch.randn(4)
        seed_everything(123)
        b = torch.randn(4)
        assert torch.equal(a, b)

    def test_seed_everything_seeds_python_and_numpy(self):
        seed_everything(7)
        r1, n1 = random.random(), np.random.rand()
        seed_everything(7)
        assert (random.random(), np.random.rand()) == (r1, n1)

    def test_invalid_seed(self):
        with pytest.raises(ValueError):
            seed_everything(-1)

    def test_deterministic_flag(self):
        seed_everything(1, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False

    def test_isolated_rng_restores_outer_state(self):
        seed_everything(0)
        expected = torch.randn(3)
        seed_everything(0)
        with isolated_rng(999):
            torch.randn(50)
            np.random.rand(5)
        assert torch.equal(torch.randn(3), expected)

    def test_capture_restore(self):
        state = capture_rng_state()
        a = torch.rand(2)
        restore_rng_state(state)
        assert torch.equal(torch.rand(2), a)

    def test_make_generator_is_reproducible(self):
        g1, g2 = make_generator(5), make_generator(5)
        assert torch.equal(torch.rand(3, generator=g1), torch.rand(3, generator=g2))

    def test_seed_worker_runs(self):
        seed_worker(0)


# ------------------------------------------------------------------------ logging


class TestLogging:
    def test_setup_logger_idempotent(self, capsys):
        logger = setup_logger("hub.test", level="DEBUG")
        setup_logger("hub.test", level="DEBUG")
        assert len(logger.handlers) == 1
        logger.info("hello")
        assert capsys.readouterr().out.count("hello") == 1

    def test_setup_logger_file(self, temp_dir):
        path = temp_dir / "logs" / "run.log"
        logger = setup_logger("hub.file", log_file=path)
        logger.warning("to-file")
        for h in logger.handlers:
            h.flush()
        assert "to-file" in path.read_text()
        for h in list(logger.handlers):  # release the file so Windows can delete temp_dir
            h.close()
            logger.removeHandler(h)

    def test_get_logger_namespacing(self):
        assert get_logger("x").name == "pytorch_mastery_hub.x"
        assert get_logger().name == "pytorch_mastery_hub"
        assert isinstance(get_logger("x"), logging.Logger)

    def test_metrics_logger_roundtrip(self, temp_dir):
        path = temp_dir / "m.jsonl"
        with MetricsLogger(path) as ml:
            ml.log(0, loss=1.0, acc=0.1)
            ml.log(1, loss=0.5, acc=0.4)
            assert ml.last("loss") == 0.5
            assert ml.best("loss") == (1, 0.5)
            assert ml.best("acc", mode="max") == (1, 0.4)
            assert len(ml) == 2
        loaded = MetricsLogger.load(path)
        assert loaded.history["loss"] == [1.0, 0.5]
        df = loaded.to_dataframe()
        assert list(df["step"]) == [0, 1]

    def test_metrics_logger_empty_best(self):
        assert MetricsLogger().best("nope") is None


# ---------------------------------------------------------------------- model utils


@pytest.fixture
def mlp():
    return nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.ReLU(), nn.Linear(16, 3))


class TestModelUtils:
    def test_count_parameters(self, mlp):
        expected = 8 * 16 + 16 + 16 + 16 + 16 * 3 + 3
        assert count_parameters(mlp) == expected
        freeze(mlp, ["0"])
        assert count_parameters(mlp, trainable_only=True) == expected - (8 * 16 + 16)

    def test_get_model_size(self, mlp):
        assert get_model_size(mlp, "B") == count_parameters(mlp) * 4
        with pytest.raises(ValueError):
            get_model_size(mlp, "TB")

    def test_summarize_shapes(self, mlp):
        rows = summarize(mlp, (2, 8))
        assert [r.output_shape for r in rows] == [[2, 16], [2, 16], [2, 16], [2, 3]]
        assert rows[0].params == 8 * 16 + 16

    def test_summarize_requires_input(self, mlp):
        with pytest.raises(ValueError):
            summarize(mlp)

    def test_model_summary_text(self, mlp, capsys):
        text = model_summary(mlp, (1, 8))
        assert "Total params" in text
        assert "Linear" in capsys.readouterr().out
        assert "Total params" in model_summary(mlp, print_summary=False)

    def test_freeze_unfreeze(self, mlp):
        n = freeze(mlp)
        assert n == 6 and not any(p.requires_grad for p in mlp.parameters())
        unfreeze(mlp, ["3.*"])
        assert mlp[3].weight.requires_grad and not mlp[0].weight.requires_grad

    def test_param_groups(self, mlp):
        groups = param_groups_with_weight_decay(mlp, 0.1)
        decay_ids = {id(p) for p in groups[0]["params"]}
        assert id(mlp[0].weight) in decay_ids and id(mlp[3].weight) in decay_ids
        assert id(mlp[0].bias) not in decay_ids
        assert id(mlp[1].weight) not in decay_ids  # LayerNorm excluded
        assert groups[1]["weight_decay"] == 0.0
        total = sum(p.numel() for g in groups for p in g["params"])
        assert total == count_parameters(mlp)

    @pytest.mark.parametrize(
        "method",
        [
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            "orthogonal",
            "trunc_normal",
        ],
    )
    def test_init_weights(self, mlp, method):
        init_weights(mlp, method)
        assert torch.all(mlp[0].bias == 0)
        assert torch.all(mlp[1].weight == 1)

    def test_init_weights_unknown(self, mlp):
        with pytest.raises(ValueError):
            init_weights(mlp, "bogus")


# --------------------------------------------------------------------- memory utils


class TestMemory:
    def test_get_memory_usage_cpu(self):
        stats = get_memory_usage("cpu")
        assert stats["process_rss_mb"] > 0

    def test_tensor_bytes_counts_shared_storage_once(self):
        t = torch.zeros(1024, dtype=torch.float32)
        assert tensor_bytes(t, t[:10]) == 4096

    def test_tracker(self):
        tracker = MemoryTracker("cpu")
        with tracker:
            tracker.snapshot("mid")
        assert [s.tag for s in tracker.snapshots] == ["enter", "mid", "exit"]
        assert tracker.snapshots[0].delta_mb is None
        assert tracker.snapshots[1].delta_mb is not None
        assert tracker.peak() > 0
        assert "mid" in tracker.report()
        assert len(tracker.to_records()) == 3
        tracker.reset()
        assert tracker.report().endswith("no snapshots")

    def test_clear_memory_runs(self):
        clear_memory("cpu")
