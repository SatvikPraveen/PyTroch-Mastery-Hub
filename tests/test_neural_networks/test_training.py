"""Tests for the Trainer engine, functional train/validate loops, callbacks and EMA."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_mastery_hub.neural_networks.ema import ModelEMA, steps_to_reach
from pytorch_mastery_hub.neural_networks.training import (
    Callback,
    EarlyStopping,
    LambdaCallback,
    LearningRateSchedulerCallback,
    ModelCheckpoint,
    ProgressCallback,
    Trainer,
    TrainerConfig,
    train_epoch,
    train_with_mixed_precision,
    validate_epoch,
)


def make_loaders(n=64, d=8, c=3, batch=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    w = torch.randn(d, c, generator=g)
    y = (x @ w).argmax(1)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch, shuffle=False), DataLoader(ds, batch_size=batch)


def make_model(d=8, c=3, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, c))


@pytest.fixture
def setup():
    train, val = make_loaders()
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, opt, train, val


# ------------------------------------------------------------------ functional API


class TestFunctional:
    def test_train_epoch_returns_loss_and_accuracy(self, setup):
        model, opt, train, _ = setup
        out = train_epoch(model, train, nn.CrossEntropyLoss(), opt, "cpu")
        assert set(out) >= {"loss", "accuracy"}
        assert 0 <= out["accuracy"] <= 100

    def test_train_epoch_reduces_loss(self, setup):
        model, opt, train, _ = setup
        first = train_epoch(model, train, nn.CrossEntropyLoss(), opt, "cpu")["loss"]
        for _ in range(5):
            last = train_epoch(model, train, nn.CrossEntropyLoss(), opt, "cpu")["loss"]
        assert last < first

    def test_regression_has_no_accuracy(self):
        x = torch.randn(32, 4)
        y = x.sum(1, keepdim=True)
        loader = DataLoader(TensorDataset(x, y), batch_size=8)
        model = nn.Linear(4, 1)
        out = train_epoch(
            model, loader, nn.MSELoss(), torch.optim.SGD(model.parameters(), 0.01), "cpu"
        )
        assert "accuracy" not in out

    def test_grad_accumulation_matches_full_batch(self):
        """Accumulating 4 micro-batches must equal one step on the full batch."""
        x = torch.randn(32, 8)
        y = torch.randint(0, 3, (32,))
        full = DataLoader(TensorDataset(x, y), batch_size=32)
        micro = DataLoader(TensorDataset(x, y), batch_size=8)

        m1, m2 = make_model(seed=1), make_model(seed=1)
        o1 = torch.optim.SGD(m1.parameters(), lr=0.1)
        o2 = torch.optim.SGD(m2.parameters(), lr=0.1)
        train_epoch(m1, full, nn.CrossEntropyLoss(), o1, "cpu")
        train_epoch(m2, micro, nn.CrossEntropyLoss(), o2, "cpu", accumulation_steps=4)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)

    def test_partial_accumulation_window_is_flushed(self):
        x = torch.randn(24, 8)
        y = torch.randint(0, 3, (24,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8)  # 3 batches, accum=2
        model = make_model()
        before = [p.clone() for p in model.parameters()]
        train_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), 0.1),
            "cpu",
            accumulation_steps=2,
        )
        assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))

    def test_clip_grad_norm_reports_norm(self, setup):
        model, opt, train, _ = setup
        out = train_epoch(model, train, nn.CrossEntropyLoss(), opt, "cpu", clip_grad_norm=0.5)
        assert "grad_norm" in out and out["grad_norm"] > 0

    def test_per_step_scheduler_is_stepped(self, setup):
        model, opt, train, _ = setup
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5**s)
        train_epoch(model, train, nn.CrossEntropyLoss(), opt, "cpu", scheduler=sched)
        assert sched.last_epoch == len(train)

    def test_custom_metrics_and_dict_batches(self):
        class DictModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(8, 3)

            def forward(self, x):
                return self.lin(x)

        x = torch.randn(16, 8)
        y = torch.randint(0, 3, (16,))
        loader = DataLoader([{"x": x[i], "labels": y[i]} for i in range(16)], batch_size=8)
        model = DictModel()
        out = train_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), 0.1),
            "cpu",
            metrics={"max_logit": lambda o, t: o.max()},
        )
        assert "max_logit" in out

    def test_bad_batch_type(self):
        model = make_model()
        loader = [torch.randn(4, 8)]
        with pytest.raises(TypeError):
            train_epoch(
                model,
                loader,
                nn.CrossEntropyLoss(),
                torch.optim.SGD(model.parameters(), 0.1),
                "cpu",
            )

    def test_validate_epoch_with_dataset_metrics(self, setup):
        model, _, _, val = setup
        out = validate_epoch(
            model,
            val,
            nn.CrossEntropyLoss(),
            "cpu",
            compute_metrics=lambda o, t: {"n": float(len(t))},
        )
        assert out["n"] == 64
        assert not model.training

    def test_cpu_bf16_autocast(self, setup):
        model, opt, train, val = setup
        out = train_epoch(model, train, nn.CrossEntropyLoss(), opt, "cpu", precision="auto")
        assert torch.isfinite(torch.tensor(out["loss"]))
        assert "loss" in validate_epoch(model, val, nn.CrossEntropyLoss(), "cpu", precision="bf16")
        assert "loss" in train_with_mixed_precision(model, train, nn.CrossEntropyLoss(), opt, "cpu")

    def test_bad_precision(self, setup):
        model, opt, train, _ = setup
        with pytest.raises(ValueError):
            train_epoch(model, train, nn.CrossEntropyLoss(), opt, "cpu", precision="int8")


# ------------------------------------------------------------------------ trainer


class TestTrainer:
    def test_fit_history_keys(self, setup):
        model, opt, train, val = setup
        trainer = Trainer(
            model, nn.CrossEntropyLoss(), opt, "cpu", config=TrainerConfig(epochs=2, verbose=False)
        )
        hist = trainer.fit(train, val)
        assert {"loss", "accuracy", "val_loss", "val_accuracy", "lr", "epoch_time"} <= hist.keys()
        assert len(hist["loss"]) == 2
        assert trainer.global_step == 2 * len(train)

    def test_legacy_positional_signature(self, setup):
        model, opt, train, _ = setup
        trainer = Trainer(model, nn.CrossEntropyLoss(), opt, torch.device("cpu"), None, 1.0, 2)
        assert trainer.config.clip_grad_norm == 1.0 and trainer.config.accumulation_steps == 2
        hist = trainer.fit(train, epochs=1, verbose=False)
        assert "loss" in hist and "val_loss" not in hist

    def test_epoch_scheduler_and_plateau(self, setup):
        model, opt, train, val = setup
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        trainer = Trainer(
            model,
            nn.CrossEntropyLoss(),
            opt,
            "cpu",
            scheduler=sched,
            config=TrainerConfig(epochs=2, verbose=False),
        )
        hist = trainer.fit(train, val)
        assert hist["lr"] == pytest.approx([0.05, 0.025])

        opt2 = torch.optim.SGD(model.parameters(), lr=0.1)
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=0, factor=0.1)
        trainer = Trainer(
            model,
            nn.CrossEntropyLoss(),
            opt2,
            "cpu",
            scheduler=plateau,
            config=TrainerConfig(epochs=3, verbose=False),
        )
        trainer.fit(train, val)
        assert plateau.last_epoch == 3

    def test_early_stopping_restores_best(self, setup):
        model, opt, train, val = setup
        es = EarlyStopping(monitor="val_loss", patience=1, verbose=False)
        # Use a huge LR so validation loss gets worse after the first epoch.
        opt = torch.optim.SGD(model.parameters(), lr=50.0)
        trainer = Trainer(
            model,
            nn.CrossEntropyLoss(),
            opt,
            "cpu",
            callbacks=[es],
            config=TrainerConfig(epochs=20, verbose=False),
        )
        trainer.fit(train, val)
        assert trainer.stop_training and es.stopped_epoch is not None
        assert trainer.current_epoch < 20
        restored = validate_epoch(model, val, nn.CrossEntropyLoss(), "cpu")["loss"]
        assert restored == pytest.approx(es.best_score, rel=1e-4)

    def test_early_stopping_bad_mode(self):
        with pytest.raises(ValueError):
            EarlyStopping(mode="sideways")

    def test_model_checkpoint_and_resume(self, setup, temp_dir):
        model, opt, train, val = setup
        ckpt = ModelCheckpoint(
            temp_dir / "ep{epoch:02d}.pt", save_best_only=False, save_last=True, verbose=False
        )
        trainer = Trainer(
            model,
            nn.CrossEntropyLoss(),
            opt,
            "cpu",
            callbacks=[ckpt],
            config=TrainerConfig(epochs=2, verbose=False, ema_decay=0.9),
        )
        trainer.fit(train, val)
        assert (temp_dir / "ep01.pt").exists() and (temp_dir / "ep02.pt").exists()
        assert ckpt.last_path is not None and ckpt.last_path.exists()
        assert ckpt.best_path is not None

        # Resume into a fresh trainer and continue for one more epoch.
        model2 = make_model(seed=99)
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.1)
        trainer2 = Trainer(
            model2,
            nn.CrossEntropyLoss(),
            opt2,
            "cpu",
            config=TrainerConfig(epochs=3, verbose=False, ema_decay=0.9),
        )
        extra = trainer2.load_checkpoint(ckpt.last_path)
        assert trainer2.current_epoch == 2 and trainer2.global_step == trainer.global_step
        assert "epoch_logs" in extra
        for a, b in zip(model.parameters(), model2.parameters()):
            assert torch.equal(a, b)
        hist = trainer2.fit(train, val)
        assert len(hist["loss"]) == 3

    def test_ema_evaluate_and_predict(self, setup):
        model, opt, train, val = setup
        trainer = Trainer(
            model,
            nn.CrossEntropyLoss(),
            opt,
            "cpu",
            config=TrainerConfig(epochs=1, verbose=False, ema_decay=0.5),
        )
        trainer.fit(train, val)
        raw = trainer.evaluate(val, use_ema=False)
        ema = trainer.evaluate(val)
        assert raw["loss"] != ema["loss"]
        preds = trainer.predict(val)
        assert preds.shape == (64, 3)
        # swap() must leave the live weights untouched
        assert trainer.evaluate(val, use_ema=False)["loss"] == pytest.approx(raw["loss"])

    def test_callbacks_receive_hooks(self, setup, capsys):
        model, opt, train, val = setup
        calls: list[str] = []

        class Spy(Callback):
            def on_train_begin(self, trainer):
                calls.append("begin")

            def on_epoch_end(self, trainer, epoch, logs):
                calls.append(f"epoch{epoch}")

            def on_batch_end(self, trainer, step, logs):
                calls.append("batch")

            def on_train_end(self, trainer):
                calls.append("end")

        lam = LambdaCallback(on_validation_end=lambda t, e, logs: calls.append("val"))
        trainer = Trainer(
            model,
            nn.CrossEntropyLoss(),
            opt,
            "cpu",
            callbacks=[Spy(), lam, ProgressCallback()],
            config=TrainerConfig(epochs=2, verbose=False),
        )
        trainer.add_callback(LearningRateSchedulerCallback(torch.optim.lr_scheduler.StepLR(opt, 1)))
        trainer.fit(train, val)
        assert calls[0] == "begin" and calls[-1] == "end"
        assert calls.count("batch") == 2 * len(train)
        assert "epoch0" in calls and "val" in calls
        assert "Epoch 1/2" in capsys.readouterr().out

    def test_lambda_callback_rejects_unknown_hook(self):
        with pytest.raises(ValueError):
            LambdaCallback(on_coffee_break=lambda: None)

    def test_verbose_printing(self, setup, capsys):
        model, opt, train, val = setup
        Trainer(
            model,
            nn.CrossEntropyLoss(),
            opt,
            "cpu",
            config=TrainerConfig(epochs=1, log_every_n_steps=2),
        ).fit(train, val)
        out = capsys.readouterr().out
        assert "Epoch 1/1" in out and "step 2" in out

    def test_config_validation(self):
        with pytest.raises(ValueError):
            TrainerConfig(accumulation_steps=0)
        with pytest.raises(ValueError):
            TrainerConfig(epochs=-1)

    def test_seeded_trainer_is_reproducible(self):
        """With config.seed the shuffle order and dropout masks are identical across runs."""
        results = []
        for _ in range(2):
            x = torch.arange(64, dtype=torch.float32).view(64, 1).repeat(1, 8)
            y = torch.arange(64) % 3
            train = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
            model = nn.Sequential(nn.Dropout(0.5), nn.Linear(8, 3))
            with torch.no_grad():
                model[1].weight.fill_(0.01)
                model[1].bias.zero_()
            cfg = TrainerConfig(epochs=1, verbose=False, seed=3)
            t = Trainer(
                model,
                nn.CrossEntropyLoss(),
                torch.optim.SGD(model.parameters(), 0.1),
                "cpu",
                config=cfg,
            )
            t.fit(train)
            results.append(model[1].weight.detach().clone())
        assert torch.equal(results[0], results[1])


# ---------------------------------------------------------------------------- EMA


class TestEMA:
    def test_update_moves_towards_model(self):
        model = nn.Linear(2, 2)
        ema = ModelEMA(model, decay=0.5)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema.update(model)
        expected = 0.5 * ema.module.weight + 0.5  # was ema0 (copy of init) -> mix
        assert torch.allclose(ema.module.weight, expected) or ema.step == 1
        before = ema.module.weight.clone()
        ema.update(model)
        assert torch.all((1.0 - ema.module.weight).abs() < (1.0 - before).abs() + 1e-7)

    def test_warmup_decay(self):
        ema = ModelEMA(nn.Linear(1, 1), decay=0.999, warmup_steps=10)
        assert ema.effective_decay() < 0.999
        ema.step = 10_000
        assert ema.effective_decay() == pytest.approx(0.999)

    def test_swap_and_copy_to(self):
        model = nn.Linear(2, 2)
        ema = ModelEMA(model, decay=0.0)  # decay 0 -> EMA == latest model
        with torch.no_grad():
            model.weight.fill_(3.0)
        ema.update(model)
        with torch.no_grad():
            model.weight.fill_(7.0)
        with ema.swap(model):
            assert torch.all(model.weight == 3.0)
        assert torch.all(model.weight == 7.0)
        ema.copy_to(model)
        assert torch.all(model.weight == 3.0)

    def test_state_dict_roundtrip(self):
        ema = ModelEMA(nn.Linear(2, 2), decay=0.9)
        ema.step = 5
        other = ModelEMA(nn.Linear(2, 2), decay=0.1)
        other.load_state_dict(ema.state_dict())
        assert other.decay == 0.9 and other.step == 5
        assert torch.equal(other.module.weight, ema.module.weight)
        assert "ModelEMA" in repr(other)

    def test_invalid_decay(self):
        with pytest.raises(ValueError):
            ModelEMA(nn.Linear(1, 1), decay=1.5)

    def test_steps_to_reach(self):
        assert steps_to_reach(0.9, 0.99) == 44
        with pytest.raises(ValueError):
            steps_to_reach(1.0)
