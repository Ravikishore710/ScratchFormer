"""Tests for the Attention Is All You Need learning rate schedule."""
import tensorflow as tf

from src.training.lr_schedule import TransformerLR


def test_lr_schedule_warmup_linear_growth():
    d_model = 128
    warmup_steps = 4000
    sched = TransformerLR(d_model=d_model, warmup_steps=warmup_steps)

    # In warmup: lr(step) = d_model^(-0.5) * step * warmup^(-1.5)
    lr_10 = float(sched(10))
    lr_20 = float(sched(20))
    lr_100 = float(sched(100))

    # Linear proportionality
    assert abs(lr_20 - 2.0 * lr_10) < 1e-7
    assert abs(lr_100 - 10.0 * lr_10) < 1e-7


def test_lr_schedule_peak_and_decay():
    d_model = 128
    warmup_steps = 1000
    sched = TransformerLR(d_model=d_model, warmup_steps=warmup_steps)

    lr_before = float(sched(warmup_steps - 10))
    lr_peak = float(sched(warmup_steps))
    lr_after = float(sched(warmup_steps + 10))
    lr_far = float(sched(warmup_steps * 4))

    assert lr_peak >= lr_before
    assert lr_peak >= lr_after
    assert lr_far < lr_peak
    # Decay follows 1/sqrt(step): at 4x step, decay is ~0.5x
    assert abs(lr_far - 0.5 * lr_peak) < 1e-4


def test_lr_schedule_keras_config():
    sched = TransformerLR(d_model=256, warmup_steps=8000)
    config = sched.get_config()
    assert config["d_model"] == 256
    assert config["warmup_steps"] == 8000
