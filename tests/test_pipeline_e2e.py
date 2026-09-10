"""End-to-end integration test verifying the entire ScratchFormer pipeline."""
import tempfile
from pathlib import Path

import tensorflow as tf

from src.config import Config
from src.data.tokenizer import WordTokenizer
from src.evaluation.evaluate import quick_gen_metrics
from src.inference.generate import translate
from src.model.embedding import sinusoidal_position_encoding
from src.model.transformer import Transformer
from src.training.trainer import Trainer
from src.visualization.plots import plot_training_curves, visualize_attentions


def test_end_to_end_mini_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Config
        cfg = Config(
            d_model=32, num_heads=4, num_layers=1, d_ff=64,
            max_seq_len=16, batch_size=4, epochs=1, output_dir=str(tmp_path)
        )

        # 2. Tokenizers
        src_texts = ["hello world", "good morning", "how are you", "thank you"]
        tgt_texts = ["bonjour monde", "bonjour", "comment allez vous", "merci"]

        src_tok = WordTokenizer(vocab_size=30).build(src_texts)
        tgt_tok = WordTokenizer(vocab_size=30).build(tgt_texts)

        # 3. Model
        model = Transformer(cfg, len(src_tok), len(tgt_tok))
        ckpt_dir = tmp_path / "model" / "test_ckpt"
        trainer = Trainer(model, cfg, checkpoint_dir=str(ckpt_dir))

        # 4. Tiny synthetic dataset
        src_ids = tf.constant([[src_tok.encode(t, max_len=16) + [0] * (16 - len(src_tok.encode(t, max_len=16)))]
                               for t in src_texts], dtype=tf.int64)
        src_ids = tf.squeeze(src_ids, axis=1)

        tgt_ids = tf.constant([[tgt_tok.encode(t, max_len=15) + [tgt_tok.eos_id] + [0] * (15 - len(tgt_tok.encode(t, max_len=15)))]
                               for t in tgt_texts], dtype=tf.int64)
        tgt_ids = tf.squeeze(tgt_ids, axis=1)

        dec_in = tf.concat([tf.fill([4, 1], tf.constant(tgt_tok.sos_id, dtype=tf.int64)), tgt_ids[:, :-1]], axis=1)
        ds = tf.data.Dataset.from_tensor_slices(((src_ids, dec_in), tgt_ids)).batch(4)

        # 5. Overfit gate check
        losses = trainer.overfit_check(ds, steps=50)
        assert len(losses) == 50
        assert losses[-1] < losses[0]

        # 6. Fit 1 epoch and save checkpoint
        history = trainer.fit(ds, ds)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 1

        # 7. Checkpoint restore
        restored_trainer = Trainer(model, cfg, checkpoint_dir=str(ckpt_dir))
        assert restored_trainer.restore_latest()

        # 8. Inference
        translated = translate(model, src_tok, tgt_tok, "hello world", max_len=10)
        assert isinstance(translated, str)

        # 9. Plotting
        curve_paths = plot_training_curves(history, str(tmp_path / "curves"))
        assert len(curve_paths) == 2
        for p in curve_paths:
            assert Path(p).exists()
