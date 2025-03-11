import logging
import math
import os
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Iterable, cast

import click
import torch
from torch import Tensor, utils
from torchvision.io import decode_image
from torchvision.transforms import v2

from config import Config


class FixedHeightResize(v2.Transform):

    height: int     # Requested height

    def __init__(self, height):
        super(FixedHeightResize, self).__init__()
        self.height = height

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        if height == self.height:
            return image
        else:
            ratio = float(self.height) / float(height)
            return v2.functional.resize(image, [self.height, math.ceil(width * ratio)])


class Vocab:
    PAD_T = (0, "PAD")        # Padding for image and sequence length value.
    UNK_T = (1, "UNK")        # Unknown sequence token.
    SOS_T = (2, "SOS")        # Start of sequence token.
    EOS_T = (3, "EOS")        # End of sequence token.
    SIL_T = (4, "SIL")        # Chord padding to Stats.max_chord.
    RESERVED_TOKENS = [PAD_T, UNK_T, SOS_T, EOS_T, SIL_T]

    PAD, UNK, SOS, EOS, SIL = map(lambda x: x[0], RESERVED_TOKENS)

    config: Config
    _tok2i: dict[str, int]
    _i2tok: dict[int, str]

    def __init__(self, config: Config, _tok2i: dict[str, int]):
        self.config = config
        self._tok2i = _tok2i
        self._i2tok = {ival: key for key, ival in _tok2i.items()}

    def __len__(self):
        return len(self._tok2i)

    def tok2i(self, tokens: list[str]) -> Tensor:
        c = self.config
        tensor = torch.full((c.max_chord, ), self.SIL)
        for idx, tok in enumerate(tokens):
            tensor[idx] = self._tok2i.get(tok, self.UNK)
        return tensor

    def i2tok(self, ids: Tensor | Iterable[int]) -> list[str]:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        return [self._i2tok.get(id, self.UNK_T[1]) for id in ids]


class Dataset(utils.data.Dataset):

    data: list[Path]

    config: Config
    vocab: Vocab

    i_sos: Tensor
    i_eos: Tensor
    s_sos: Tensor
    s_eos: Tensor

    def __init__(self, config, vocab: Vocab, data: list[Path]):
        self.config = config
        self.vocab = vocab
        self.data = data
        # Image transforms
        self.transform = v2.Compose([
            v2.Grayscale(),
            FixedHeightResize(self.config.ipad_shape[0]),
            v2.ToDtype(torch.float),
            v2.Normalize(mean=[228.06], std=[62.78])
        ])

        # Precomputes padding tensors (start/end) x (image, seq)
        self.s_sos = torch.full((1, self.config.max_chord), Vocab.SOS)
        self.s_eos = torch.full((1, self.config.max_chord), Vocab.EOS)
        self.i_sos = torch.full((1, self.config.ipad_shape[0]), Vocab.SOS)
        self.i_eos = torch.full((1, self.config.ipad_shape[0]), Vocab.EOS)

    def _load_sequence(self, path: Path) -> Tensor:
        c, v = self.config, self.vocab
        with open(path.with_suffix(".tokens"), "r") as file:
            records = list(file)
            width = len(records)
            assert width + 2 <= c.spad_len, \
                f"{path} exceeds pad length {c.spad_len}"
            tensor = torch.full(
                (c.spad_len - 2, c.max_chord), v.PAD, dtype=torch.int32
            )
            for idx, record in enumerate(records):
                tensor[idx, :] = v.tok2i(record.strip().split())
        return torch.cat([self.s_sos, tensor, self.s_eos])

    def _load_image(self, path: Path) -> Tensor:
        c, v = self.config, self.vocab
        image = decode_image(Path(path.with_suffix(".jpg")).as_posix())
        image = self.transform(image)
        image = image.squeeze(0).permute(1, 0)
        width, height = image.shape
        ipad_height, ipad_len = c.ipad_shape
        assert width + 2 <= ipad_len and height == ipad_height, \
            f"{path} exceeds pad length {ipad_len} or height mismatches."
        tensor = torch.full(
            (ipad_len - 2, height), v.PAD, dtype=torch.float32
        )
        tensor[:width, :] = image
        return torch.cat([self.i_sos, tensor, self.i_eos])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self._load_image(self.data[idx]), self._load_sequence(self.data[idx])


class Factory:

    config: Config
    home: Path
    data: list[Path]
    vocab: Vocab

    def __init__(self, home: Path, refresh: bool = False):
        """Initializes the dataset.

        Given a freshly unzipped GrandPiano dataset in the HOME directory,
        this will create two files:
        - A list of all available samples in {home}/list.pkl
        - The corresponding vocab file in {home}/vocab.pkl


        Args:
            home (Path): The directory containing the raw GrandPiano dataset.
            force (bool): Recreates both the list and vocab files even if they exist.
        """
        self.config = Config()
        self.home = home
        self._load(refresh)
        self._vocab(refresh)
        self.config = replace(self.config, vocab_size=len(self.vocab))

    def datasets(self, valid_split: float) -> tuple['Dataset', 'Dataset']:
        train_len = int((1.0 - valid_split) * len(self.data))
        return (
            Dataset(self.config, self.vocab, self.data[:train_len]),
            Dataset(self.config, self.vocab, self.data[train_len:])
        )

    def dataset(self):
        return Dataset(self.config, self.vocab, self.data)

    def _accept(self, tokens_path: Path) -> bool:
        if not tokens_path.with_suffix(".jpg").exists():
            return False
        # Checks the image size after rescaling.
        _, h, w = decode_image(
            tokens_path.with_suffix(".jpg").as_posix()).shape
        w = 1 + int(w / (h / self.config.ipad_shape[0]))
        if w + 2 > self.config.ipad_shape[1]:
            return False
        # Checks the sequence length.
        with open(tokens_path, "r") as records:
            if len(list(records)) > self.config.spad_len:
                return False

        return True

    def _load(self, refresh: bool):
        pkl_path = Path(self.home) / 'data.pkl'
        if refresh or not pkl_path.exists():
            logging.info("Creating data set.")
            self.data = []
            for root, _, names in os.walk(self.home):
                for name in names:
                    path = Path(root) / name
                    if path.suffix == '.tokens':
                        if self._accept(path):
                            self.data.append(path.with_suffix(""))
                        else:
                            logging.info(f"{path} rejected (too large).")
            with open(pkl_path, "wb+") as f:
                pickle.dump(self.data, f)
        else:
            logging.info(f"Loading data set from {pkl_path}.")
            with open(pkl_path, "rb") as f:
                self.data = cast(list[Path], pickle.load(f))

    def _vocab(self, refresh: bool = False):
        pkl_path = Path(self.home) / "vocab.pkl"
        if refresh or not pkl_path.exists():
            logging.info("Creating vocabulary.")
            tok2i = {s: i for i, s in Vocab.RESERVED_TOKENS}
            for path in self.data:
                with open(path.with_suffix(".tokens"), "r") as f:
                    for record in f:
                        for token in record.strip().split():
                            if tok2i.get(token, None) is None:
                                tok2i[token] = len(tok2i)
            with open(pkl_path, "wb+") as f:
                pickle.dump(tok2i, f)
        else:
            logging.info(f"Loading vocabulary from {pkl_path}")
            with open(pkl_path, "rb") as f:
                tok2i = pickle.load(f)
        self.vocab = Vocab(self.config, tok2i)


@click.command()
@click.argument("home",
                type=click.Path(file_okay=False, dir_okay=True, exists=True),
                default=Path("/home/anselm/datasets/GrandPiano"))
def init_dataset(home: Path):
    factory = Factory(Path(home))
    dataset = factory.dataset()
    loader = utils.data.DataLoader(dataset, batch_size=16)
    count = 0
    for images, _ in loader:
        count += len(images)
    logging.info(f"{home} inited - {count:,} samples.")
