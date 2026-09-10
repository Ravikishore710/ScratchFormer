"""Custom training loop built on tf.GradientTape (no model.fit)."""
from __future__ import annotations

import time
from pathlib import Path

import tensorflow as tf

from ..config import Config
from .losses import masked_accuracy, masked_loss
from .lr_schedule import TransformerLR


class Trainer:
    def __init__(self, model: tf.keras.Model, cfg: Config,
                 checkpoint_dir: str | None = None):
        self.model = model
        self.cfg = cfg
        self.pad_id = 0
        self.schedule = TransformerLR(cfg.d_model, cfg.warmup_steps)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_acc = tf.keras.metrics.Mean(name="train_acc")
        self._step = tf.function(self._step_impl)
        self._eval_step = tf.function(self._eval_step_impl)
        self.ckpt = self.manager = None
        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
            self.manager = tf.train.CheckpointManager(self.ckpt, checkpoint_dir, max_to_keep=1)

    # ---------------- single steps (compiled) ----------------
    def _step_impl(self, src, dec_in, tgt):
        with tf.GradientTape() as tape:
            logits = self.model((src, dec_in), training=True)
            loss = masked_loss(tgt, logits, self.pad_id, self.cfg.label_smoothing)
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.cfg.clip_norm:
            grads = [tf.clip_by_norm(g, self.cfg.clip_norm) if g is not None else g
                     for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        acc = masked_accuracy(tgt, logits, self.pad_id)
        return loss, acc

    def _eval_step_impl(self, src, dec_in, tgt):
        logits = self.model((src, dec_in), training=False)
        loss = masked_loss(tgt, logits, self.pad_id, self.cfg.label_smoothing)
        return loss, masked_accuracy(tgt, logits, self.pad_id)

    # ---------------- public API ----------------
    def train_step(self, src, dec_in, tgt):
        loss, acc = self._step(src, dec_in, tgt)
        self.train_loss.update_state(loss)
        self.train_acc.update_state(acc)
        return float(loss), float(acc)

    def evaluate(self, ds, max_batches: int | None = None):
        loss_m = tf.keras.metrics.Mean()
        acc_m = tf.keras.metrics.Mean()
        for i, batch in enumerate(ds):
            if max_batches is not None and i >= max_batches:
                break
            (src, dec_in), tgt = batch
            loss, acc = self._eval_step(src, dec_in, tgt)
            loss_m.update_state(loss)
            acc_m.update_state(acc)
        return float(loss_m.result()), float(acc_m.result())

    def fit(self, train_ds, val_ds, log_every: int = 100):
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val = float("inf")
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            self.train_loss.reset_state()
            self.train_acc.reset_state()
            for i, batch in enumerate(train_ds, start=1):
                (src, dec_in), tgt = batch
                self.train_step(src, dec_in, tgt)
                if i % log_every == 0:
                    print(f"  epoch {epoch} step {i:>4} "
                          f"loss={float(self.train_loss.result()):.4f} "
                          f"acc={float(self.train_acc.result()):.4f}")
            val_loss, val_acc = self.evaluate(val_ds)
            dt = time.time() - t0
            history["train_loss"].append(float(self.train_loss.result()))
            history["train_acc"].append(float(self.train_acc.result()))
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(f"Epoch {epoch:>2}/{self.cfg.epochs} | "
                  f"train loss {history['train_loss'][-1]:.4f} "
                  f"acc {history['train_acc'][-1]:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f} | {dt:.0f}s")
            if val_loss < best_val:
                best_val = val_loss
                self.save_checkpoint()
        return history

    # ---------------- checkpoints ----------------
    def save_checkpoint(self):
        if self.manager is not None:
            self.manager.save()
            print(f"  [checkpoint] saved to {self.manager.directory}")

    def restore_latest(self) -> bool:
        if self.manager is None:
            return False
        latest = self.manager.latest_checkpoint
        if latest is None:
            return False
        self.ckpt.restore(latest).expect_partial()
        print(f"  [checkpoint] restored {latest}")
        return True

    # ---------------- tiny-subset overfit verification ----------------
    def overfit_check(self, train_ds, steps: int = 200, log_every: int = 40) -> list[float]:
        """Intentionally overfit ONE batch. If loss does not collapse, stop and debug."""
        batch = next(iter(train_ds))
        (src, dec_in), tgt = batch
        losses = []
        for step in range(1, steps + 1):
            loss, _ = self.train_step(src, dec_in, tgt)
            losses.append(loss)
            if step % log_every == 0 or step == 1:
                print(f"  overfit step {step:>4}: loss={loss:.4f}")
        if losses[-1] > losses[0] * 0.5:
            raise RuntimeError(
                "Tiny-subset overfit check FAILED: model could not overfit a single "
                "batch. Debug the architecture/loss before full training.")
        print("  overfit check PASSED: model can memorize a small subset.")
        return losses
