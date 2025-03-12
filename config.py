import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from utils import current_commit, from_json


@dataclass(frozen=True)
class Config:
    git_hash: str = current_commit()

    # Dataset related configuration, provided by GrandPiano.
    ipad_shape: tuple[int, int] = (256, 1024)
    spad_len: int = 128
    max_chord: int = 12
    vocab_size: int = -1

    # Image embedder config.
    image_reducer: int = 64
    embed_size: int = 256

    # Sequence embedder config.
    sequence_reducer: int = 64

    # Transformer config.
    num_head: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feed_forward: int = 1024
    dropout: float = 0.1

    # Training config.
    batch_size = 16

    def save(self, path: Path):
        with open(path, "w+") as fp:
            json.dump(asdict(self), fp, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'Config':
        with open(path, "r") as fp:
            obj = json.load(fp)
        return cast(Config, from_json(cls, obj))
