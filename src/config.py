"""Central configuration for ScratchFormer.

Every experiment is fully described by one Config object so that every
run is reproducible from a single JSON file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path


@dataclass
class Config:
    # ---- dataset ----
    dataset_url: str = "https://www.manythings.org/anki/fra-eng.zip"
    data_dir: str = "data"
    max_pairs: int = 60000
    max_seq_len: int = 40
    val_size: float = 0.10
    test_size: float = 0.10
    seed: int = 42

    # ---- vocabulary ----
    src_vocab_size: int = 20000
    tgt_vocab_size: int = 25000
    min_freq: int = 2

    # ---- model ----
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    use_positional_encoding: bool = True

    # ---- training ----
    batch_size: int = 128
    epochs: int = 10
    warmup_steps: int = 4000
    label_smoothing: float = 0.0
    clip_norm: float = 1.0

    # ---- inference ----
    max_decode_len: int = 40

    # ---- io ----
    output_dir: str = "outputs"
    exp_name: str = "baseline"

    # --------------------------------------------------------------
    def validate(self) -> "Config":
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid).validate()

    @classmethod
    def load_json(cls, path: str) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def override(self, **kwargs) -> "Config":
        return replace(self, **kwargs).validate()
