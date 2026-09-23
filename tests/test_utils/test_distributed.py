"""Tests for distributed helpers: single-process fallbacks and a real 2-process gloo group."""

from __future__ import annotations

import sys

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_mastery_hub.utils import distributed as du


class TestSingleProcess:
    def test_introspection_defaults(self):
        assert not du.is_distributed()
        assert du.get_rank() == 0 and du.get_world_size() == 1 and du.is_main_process()
        assert du.local_rank() == 0
        assert isinstance(du.local_device(), torch.device)
        assert du.init_distributed() is False  # no WORLD_SIZE -> no-op
        du.cleanup()
        du.barrier()

    def test_collectives_are_identity(self):
        t = torch.arange(3.0)
        assert torch.equal(du.all_reduce_mean(t), t)
        assert du.reduce_dict({"a": 1.0, "b": torch.tensor(2.0)}) == {"a": 1.0, "b": 2.0}
        assert du.reduce_dict({}) == {}
        assert torch.equal(du.all_gather_tensors(t), t)
        assert du.all_gather_object({"x": 1}) == [{"x": 1}]

    def test_model_and_sampler_passthrough(self):
        model = nn.Linear(2, 2)
        assert du.wrap_ddp(model) is model
        assert du.unwrap_ddp(model) is model
        ds = TensorDataset(torch.zeros(4, 2))
        assert du.make_sampler(ds) is None
        du.set_epoch(DataLoader(ds), 3)  # no sampler.set_epoch -> silently ignored

    def test_decorators(self):
        calls = []

        @du.main_process_only
        def log(x):
            calls.append(x)
            return x

        assert log(1) == 1 and calls == [1]
        with du.main_process_first():
            calls.append("inside")
        assert calls[-1] == "inside"

    def test_find_free_port(self):
        assert 1024 < du.find_free_port() < 65536


def _worker(rank: int, world_size: int, tmpdir: str):
    assert du.is_distributed() and du.get_world_size() == world_size and du.get_rank() == rank
    # reduce_dict averages
    out = du.reduce_dict({"loss": float(rank + 1)})
    assert out["loss"] == pytest.approx((1 + world_size) / 2)
    # all_reduce_mean
    assert du.all_reduce_mean(torch.tensor([rank * 2.0])).item() == pytest.approx(world_size - 1)
    # uneven all_gather
    g = du.all_gather_tensors(torch.full((rank + 1, 2), float(rank)))
    assert g.shape[0] == sum(range(1, world_size + 1))
    assert du.all_gather_object(rank) == list(range(world_size))
    # DDP wrap + sampler + a training step that syncs grads
    torch.manual_seed(0)
    model = du.wrap_ddp(nn.Linear(4, 1))
    assert isinstance(model, nn.parallel.DistributedDataParallel)
    assert du.unwrap_ddp(model) is model.module
    ds = TensorDataset(torch.randn(16, 4), torch.randn(16, 1))
    sampler = du.make_sampler(ds, shuffle=True)
    loader = DataLoader(ds, sampler=sampler, batch_size=4)
    du.set_epoch(loader, 1)
    x, y = next(iter(loader))
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    # After backward, DDP has all-reduced gradients: identical on every rank.
    grads = du.all_gather_tensors(model.module.weight.grad.flatten()[None])
    assert torch.allclose(grads[0], grads[-1])
    with du.main_process_first():
        pass
    du.barrier()


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="gloo file rendezvous is flaky on Windows CI")
def test_two_process_gloo_group(tmp_path):
    du.spawn(_worker, world_size=2, args=(str(tmp_path),))
