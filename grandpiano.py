
import math
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import click
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.io import decode_image
from torchvision.transforms import v2


class FixedHeightResize(v2.Transform):

    height: int     # Requested height

    def __init__(self, height):
        super(FixedHeightResize, self).__init__()
        self.height = height

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        channels, height, width = image.shape
        if height == self.height:
            return image
        else:
            ratio = float(self.height) / float(height)
            return v2.functional.resize(image, [self.height, math.ceil(width * ratio)])


class GrandPiano:
    CHORD_MAX = 12          # Maximum number of concurrent notes in dataset.

    PAD = (0, "PAD")        # Sequence vertical aka chord padding value.
    UNK = (1, "UNK")        # Unknown sequence token.
    SOS = (2, "SOS")        # End of sequence token.
    EOS = (3, "EOS")        # Beginning of sequence token.
    RESERVED_TOKENS = [PAD, UNK, EOS, SOS]

    @dataclass
    class Filter:
        max_image_width: int = -1
        max_sequence_length: int = -1

        def accept_image(self, width: int) -> bool:
            return width < self.max_image_width

        def accept_sequence(self, length: int) -> bool:
            return length < self.max_sequence_length

    # TODO Have an option to the stats command to generate this.
    @dataclass
    class Stats:
        image_height: int = 256
        max_image_width: int = 3058
        image_mean: float = 22.06
        image_std: float = 62.78
        max_sequence_length: int = 207

    STATS = Stats()

    datadir: Path
    data: List[Path] = list([])
    tok2i: Dict[str, int]
    i2tok: Dict[int, str]
    position: int = 0

    image_height: int   # Image - constant - height in dataset.
    ipad_len: int       # Width for padding images, largers dropped.
    spad_len: int       # Length for padding sequences, longers dropped.

    transform: v2.Compose
    transform_and_norm: v2.Compose

    filter: Optional[Filter]

    @property
    def vocab_size(self):
        return len(self.tok2i)

    def __init__(self,
                 datadir: Path,
                 filter: Optional[Filter] = None):
        self.datadir = datadir
        self.filter = filter
        self.image_height = self.STATS.image_height
        # Computes the padding lengths.
        # These act as filter as items longer than the pad size are filtered out of
        # the dataset. You can use the 'histo' command from main.py to adjust these
        # numbers.
        self.ipad_len = self.STATS.max_image_width
        self.spad_len = self.STATS.max_sequence_length
        if filter is not None:
            if filter.max_image_width > 0:
                self.ipad_len = filter.max_image_width
            if filter.max_sequence_length > 0:
                self.spad_len = filter.max_sequence_length
        # Initializes image transforms, with/without norming.
        self.transform = v2.Compose([
            v2.Grayscale(),
            FixedHeightResize(self.STATS.image_height),
            v2.ToDtype(torch.float)
        ])
        self.transform_and_norm = v2.Compose([
            self.transform,
            v2.Normalize(mean=[228.06], std=[62.78])
        ])
        self.list(create=True)
        self.load_vocab(create=True)

    def list(self, create: bool = False, refresh: bool = False) -> int:
        list_path = Path(self.datadir) / 'list.pickle'
        if list_path.exists() and not refresh:
            with open(list_path, "rb") as f:
                self.data = pickle.load(f)
        elif create or refresh:
            self.data = list([])
            for root, _, filenames in os.walk(self.datadir):
                for filename in filenames:
                    path = Path(root) / filename
                    if path.suffix == '.tokens' and path.with_suffix(".jpg").exists():
                        self.data.append(path.with_suffix(""))
            with open(list_path, "wb+") as f:
                pickle.dump(self.data, f)
            print(f"{len(self.data)} samples found.")
        else:
            raise FileNotFoundError(f"List file {list_path} not found.")
        # Loads the set of samples.
        return len(self.data)

    def load_vocab(self, create: bool = False):
        # Loads the vocab for sequences.
        vocab_path = Path(self.datadir) / "vocab.pickle"
        if vocab_path.exists():
            # Reads in the existing vocab file.
            with open(vocab_path, "rb") as f:
                obj = pickle.load(f)
            self.tok2i = obj['tok2i']
            self.i2tok = obj['i2tok']
        elif create:
            self.create_vocab()
            self.save_vocab()
        else:
            raise FileNotFoundError(f"Pickle file {vocab_path} not found.")

    def save_vocab(self):
        assert self.tok2i and self.i2tok, "Vocab not computed yet."
        vocab_path = Path(self.datadir, "vocab.pickle")
        with open(vocab_path, "wb+") as f:
            pickle.dump({
                "tok2i": self.tok2i,
                "i2tok": self.i2tok
            }, f)

    def create_vocab(self):
        self.tok2i = {key: value for key,               # type: ignore
                      value in self.RESERVED_TOKENS}
        self.i2tok = {value: key for value, key in self.RESERVED_TOKENS}
        token_count = len(self.tok2i)
        for path in self.data:
            file = path.with_suffix(".tokens")
            with open(file, "r") as input:
                for line in input:
                    for token in line.strip().split("\t"):
                        token_count += 1
                        if self.tok2i.get(token, None) is None:
                            token_id = len(self.tok2i)
                            self.tok2i[token] = token_id
                            self.i2tok[token_id] = token

        print(f"{token_count:,} tokens, {len(self.tok2i):,} uniques.")

    def load_sequence(
        self,
        path: Path,
        pad: bool = False,
        device: str = "cpu"
    ) -> Tuple[Optional[torch.Tensor], int]:
        with open(path, "r") as file:
            records = list(file)
            width = len(records)
            if self.filter and not self.filter.accept_sequence(width+2):
                return None, 0
            length = self.spad_len if pad else width+2
            assert len(records)+2 <= length, f"{path} length {
                len(records)} exceeds padding length {self.spad_len}"
            tensor = torch.full(
                (length, self.CHORD_MAX), self.PAD[0]).to(device)
            tensor[0, :], tensor[1+width, :] = self.SOS[0], self.EOS[0]
            for idx, record in enumerate(records):
                row = torch.Tensor([
                    self.tok2i.get(tok, self.UNK[0])for tok in record.strip().split()
                ])
                tensor[1+idx, :len(row)] = row
        return tensor, width+2

    def decode(self, tokens: List[int]):
        return [self.i2tok.get(token, "UNK") for token in tokens]

    def load_image(
        self, path: Path, norm: bool = True, pad: bool = False, device: str = "cpu"
    ) -> Tuple[Optional[torch.Tensor], int]:
        image = decode_image(Path(path).as_posix()).to(device)
        image = (self.transform_and_norm if norm else self.transform)(image)
        image = image.squeeze(0).permute(1, 0)
        width, height = image.shape
        if self.filter and not self.filter.accept_image(width+2):
            return None, 0
        length = self.ipad_len if pad else width+2
        assert width+2 <= length, f"{
            path} width {width} exceeds padding width {self.ipad_len}"
        tensor = torch.full((length, height), self.PAD[0], dtype=torch.float32)
        tensor[0, :], tensor[1+width,
                             :] = float(self.SOS[0]), float(self.EOS[0])
        tensor[1:1+width, :] = image
        return tensor, width+2

    def len(self) -> int:
        return len(self.data)

    def next(
        self,
        pad: bool = False,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, int, torch.Tensor, int]:
        start_position = self.position
        while True:
            if self.position >= len(self.data):
                self.position = 0
            path = self.data[self.position]
            self.position += 1
            image, width = self.load_image(path.with_suffix(
                ".jpg"), pad=pad, device=device)
            sequence, length = self.load_sequence(path.with_suffix(
                ".tokens"), pad=pad, device=device)
            if image is not None and sequence is not None:
                return (image, width, sequence, length)
            assert self.position != start_position, "All samples"

    @ staticmethod
    def sequence_length(args: Tuple['GrandPiano', Path]) -> int:
        gp, path = args
        _, length = gp.load_sequence(path)
        return length

    def sequences_length(self) -> torch.Tensor:
        with ProcessPoolExecutor(2) as executor:
            stats = list(executor.map(GrandPiano.sequence_length, [
                (self, path.with_suffix(".tokens")) for path in self.data], chunksize=500))
        return torch.tensor([length for length in stats if length > 0], dtype=torch.int)

    @staticmethod
    def image_stats(args: Tuple['GrandPiano', Path]) -> Optional[Tuple[int, float, float]]:
        gp, path = args
        image, _ = gp.load_image(path.with_suffix(".jpg"), norm=False)
        if image is not None:
            return image.shape[0], image.mean(dim=[0, 1]).item(), image.std(dim=[0, 1]).item()
        else:
            return None

    def images_length(self) -> torch.Tensor:
        with ProcessPoolExecutor(2) as executor:
            stats = executor.map(GrandPiano.image_stats, [
                (self, path.with_suffix(".jpg")) for path in self.data], chunksize=500)
        stats = [stat for stat in stats if stat is not None]
        return torch.Tensor([l for l, m, s in stats])

    def images_stats(self) -> Dict[str, str]:
        with ProcessPoolExecutor(2) as executor:
            stats = list(executor.map(GrandPiano.image_stats, [
                (self, path.with_suffix(".jpg")) for path in self.data], chunksize=500))
        stats = [stat for stat in stats if stat is not None]
        lengths = [l for l, m, s in stats]
        means = sum([m for l, m, s in stats])
        stds = sum([s for l, m, s in stats])
        return {
            "image min len": f"{min(lengths)}",
            "image max len": f"{max(lengths)}",
            "image avg len": f"{sum(lengths) / len(lengths):.2f}",
            "image mean": f"{means / len(lengths):.2f}",
            "image std": f"{stds / len(lengths):.2f}",
        }

    def stats(self) -> Dict[str, str]:
        # Computes image stats.
        start_time = time.time()
        image_stats = self.images_stats()
        print(f"Image stats in {time.time() - start_time:.2f}")

        # Computes sequence lengths.
        start_time = time.time()
        sequence_lengths = self.sequences_length()
        print(f"Sequence length in {time.time() - start_time:.2f}")

        # Packs and returns all available stats.
        return {
            "dataset size": f"{len(self.data):,}",
            "vocab size": f"{len(self.tok2i):,}",
            "sequence min length": f"{torch.min(sequence_lengths).item()}",
            "sequence max length": f"{torch.max(sequence_lengths).item()}",
            "sequence avg length": f"{torch.sum(sequence_lengths).item() / len(sequence_lengths):.2f}",
        } | image_stats


@click.command
@click.pass_context
def make_vocab(ctx):
    """
        Creates the vocabulary file 'vocab.pickle' for the DATASET.
    """
    gp = ctx.obj
    gp.create_vocab()


@click.command
@click.pass_context
def refresh_list(ctx):
    """
        Creates the vocabulary file 'vocab.pickle' for the DATASET.
    """
    gp = cast(GrandPiano, ctx.obj)
    gp.list(refresh=True)


@click.command
@click.pass_context
def stats(ctx):
    """
        Creates the vocabulary file 'vocab.pickle' for the DATASET.
    """
    gp = cast(GrandPiano, ctx.obj)
    for key, value in gp.stats().items():
        print(f"{key:<20}: {value}")


@click.command
@click.pass_context
def histo(ctx):
    """

        Plots the histogram of sequence and image lengths.

    """
    gp = cast(GrandPiano, ctx.obj)
    sequence_lengths = gp.sequences_length()
    image_lengths = gp.images_length()
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.hist(sequence_lengths.numpy(), bins=50, cumulative=True, density=True)
    ax1.set_xlabel('Sequence lengths')
    ax2.hist(image_lengths.numpy(), bins=50, cumulative=True, density=True)
    ax2.set_xlabel('Image widths')
    fig.tight_layout()
    plt.show()


@click.command
@click.argument("path",
                type=click.Path(file_okay=True),
                required=True)
@click.option("--pad", is_flag=True,
              help="Pad the itemafter loading.")
@ click.pass_context
def load(ctx, path: Path, pad: bool = False):
    """
        Loads and tokenizes PATH.

        PATH can be either .tokens or a .jpg file, and will be tokenized accordingly.
    """
    gp = cast(GrandPiano, ctx.obj)
    match Path(path).suffix:
        case ".tokens":
            tensor = gp.load_sequence(path, pad=pad)
        case ".jpg":
            tensor = gp.load_image(path, pad=pad)
        case _:
            raise ValueError(
                "Files of {path.suffix} suffixes can't be tokenized.")
    print(tensor)
