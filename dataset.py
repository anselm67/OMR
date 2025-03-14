import json
import logging
import math
import os
import pickle
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import click
import torch
from torch import Tensor, utils
from torchvision.io import decode_image
from torchvision.transforms import v2

from config import Config
from sequence import display_sequence
from utils import from_json
from vocab import Vocab


class FixedHeightResize(v2.Transform):

    height: int     # Requested height

    def __init__(self, height):
        super(FixedHeightResize, self).__init__()
        self.height = height

    def forward(self, image: Tensor) -> Tensor:
        _, height, width = image.shape
        if height == self.height:
            return image
        else:
            ratio = float(self.height) / float(height)
            return v2.functional.resize(image, [self.height, math.ceil(width * ratio)])


class Binarize(v2.Transform):

    def forward(self, image: Tensor) -> Tensor:
        return 255 - 255 * (image > 200)


class Dataset(utils.data.Dataset):

    data: list[Path]

    config: Config
    vocab: Vocab

    i_sos: Tensor
    i_eos: Tensor
    s_sos: Tensor
    s_eos: Tensor

    transforms: v2.Compose

    def __init__(self, config, vocab: Vocab, data: list[Path], transforms: v2.Compose):
        self.config = config
        self.vocab = vocab
        self.data = data
        # Image transforms
        self.transforms = transforms

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
                (c.spad_len - 1, c.max_chord), v.PAD, dtype=torch.int32
            )
            for idx, r in enumerate(records):
                tensor[idx, :] = v.tok2i(r.strip().split())
            tensor[width, :] = self.s_eos
        return torch.cat([self.s_sos, tensor])

    def _load_image(self, path: Path) -> Tensor:
        c, v = self.config, self.vocab
        image = decode_image(Path(path.with_suffix(".jpg")).as_posix())
        image = self.transforms(image)
        image = image.squeeze(0).permute(1, 0)
        width, height = image.shape
        ipad_height, ipad_len = c.ipad_shape
        assert width + 2 <= ipad_len and height == ipad_height, \
            f"{path} exceeds pad length {ipad_len} or height mismatches."
        tensor = torch.full(
            (ipad_len - 1, height), v.PAD, dtype=torch.float32
        )
        tensor[:width, :] = image
        tensor[width, :] = self.i_eos
        return torch.cat([self.i_sos, tensor])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self._load_image(self.data[idx]), self._load_sequence(self.data[idx])


@dataclass
class Stats:
    count: int = 0

    ipad_shape: tuple[int, int] = (0, 0)
    i_mean: float = 0
    i_std: float = 0

    spad_len: int = 0

    def update(self, img: Tensor, seq: list[str]):
        self.count += 1
        # Updates image statistics.
        self.ipad_shape = (
            max(img.shape[1], self.ipad_shape[0]),
            max(img.shape[2], self.ipad_shape[1])
        )
        self.i_mean += img.mean().item()
        self.i_std += img.std().item()
        # Update sequence statistics.
        seq_len = len(seq)
        self.spad_len = max(seq_len, self.spad_len)

    def log(self):
        logging.info(
            "Dataset created:\n" +
            f"ipad_shape    = {self.ipad_shape},\n" +
            f"i_mean, i_std = {self.i_mean / self.count:.3f}, {self.i_std / self.count:.3f}\n" +
            f"spad_len      = {self.spad_len}"
        )

    def save(self, json_path: Path):
        with open(json_path, "w+") as fp:
            json.dump(asdict(self), fp, indent=2)

    @classmethod
    def from_json(cls, json_path) -> 'Stats':
        with open(json_path, "r") as fp:
            return cast(Stats, from_json(cls, json.load(fp)))


class Factory:

    config: Config
    home: Path
    data: list[Path]
    vocab: Vocab

    train_transforms: v2.Compose
    stats_transforms: v2.Compose

    # An empty Dataset() that allows us to load and prepare images and sequences.
    data_loader: Dataset

    def __init__(self, home: Path, config: Config, refresh: bool = False):
        """Initializes the dataset.

        Given a freshly unzipped GrandPiano dataset in the HOME directory,
        this will create two files:
        - A list of all available samples in {home}/list.pkl
        - The corresponding vocab file in {home}/vocab.pkl


        Args:
            home (Path): The directory containing the raw GrandPiano dataset.
            force (bool): Recreates both the list and vocab files even if they exist.
        """
        self.home = home
        self.config = config or Config()
        self.stats_transforms = v2.Compose([
            v2.Grayscale(),
            Binarize(),
            FixedHeightResize(self.config.ipad_shape[0]),
            v2.ToDtype(torch.float),
        ])
        self.train_transforms = v2.Compose([
            v2.Grayscale(),
            Binarize(),
            FixedHeightResize(self.config.ipad_shape[0]),
            v2.ToDtype(torch.float),
            v2.Normalize(mean=[37.1844], std=[89.7948])
        ])
        self._load(refresh)
        self._vocab(refresh)
        self.config = replace(self.config, vocab_size=len(self.vocab))
        self.data_loader = Dataset(
            self.config, self.vocab, [], self.train_transforms)

    def datasets(self, valid_split: float) -> tuple['Dataset', 'Dataset']:
        train_len = int((1.0 - valid_split) * len(self.data))
        return (
            Dataset(self.config, self.vocab,
                    self.data[:train_len], self.train_transforms),
            Dataset(self.config, self.vocab,
                    self.data[train_len:], self.train_transforms)
        )

    def dataset(self):
        return Dataset(self.config, self.vocab, self.data, self.train_transforms)

    def _accept(self, tokens_path: Path, stats: Stats) -> bool:
        if not tokens_path.with_suffix(".jpg").exists():
            return False
        # Checks the image size after rescaling.
        img = decode_image(tokens_path.with_suffix(".jpg").as_posix())
        _, h, w = img.shape
        w = 1 + int(w / (h / self.config.ipad_shape[0]))
        if w + 2 > self.config.ipad_shape[1]:
            return False
        # Checks the sequence length.
        with open(tokens_path, "r") as records:
            seq = list(records)
        if len(seq) > self.config.spad_len:
            return False
        stats.update(self.stats_transforms(img), seq)
        return True

    def _load(self, refresh: bool):
        pkl_path = Path(self.home) / 'data.pkl'
        if refresh or not pkl_path.exists():
            logging.info("Creating data set.")
            stats = Stats()
            self.data = []
            for root, _, names in os.walk(self.home):
                for name in names:
                    path = Path(root) / name
                    if path.suffix == '.tokens':
                        if self._accept(path, stats):
                            self.data.append(path.with_suffix(""))
                        else:
                            logging.info(f"{path} rejected (too large).")
            with open(pkl_path, "wb+") as f:
                pickle.dump(self.data, f)
            stats.save(pkl_path.with_name("stats.json"))
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

    def load_image(self, path: Path) -> Tensor:
        return self.data_loader._load_image(path)

    def load_sequence(self, path: Path) -> Tensor:
        return self.data_loader._load_sequence(path)


@click.command()
@click.argument("home",
                type=click.Path(file_okay=False, dir_okay=True, exists=True),
                default=Path("/home/anselm/datasets/GrandPiano"))
@click.pass_context
def init_dataset(ctx, home: Path):
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    factory = Factory(Path(home), context.config)
    dataset = factory.dataset()
    logging.info(f"{home} inited - {len(dataset):,} samples.")


@click.command()
@click.pass_context
def show(ctx):
    import cv2

    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    factory = context.require_factory()
    dataset = factory.dataset()
    loader = utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for image, seq in loader:
        image, seq = image.squeeze(0), seq.squeeze(0)
        print(display_sequence(factory.vocab, seq))
        cv2.imshow("score", image.permute(1, 0).cpu().numpy())
        if cv2.waitKey(0) == ord('q'):
            return
