
import math
import os
import pickle
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union, cast

import click
import matplotlib.pyplot as plt
import torch
from torchvision.io import decode_image
from torchvision.transforms import v2

from utils import DeviceType

DatasetName = Literal["train", "valid", "all"]


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

    PAD = (0, "PAD")        # Padding for image and sequence length value.
    UNK = (1, "UNK")        # Unknown sequence token.
    SOS = (2, "SOS")        # Start of sequence token.
    EOS = (3, "EOS")        # End of sequence token.
    SIL = (4, "SIL")        # Chord padding to Stats.max_chord.
    RESERVED_TOKENS = [PAD, UNK, SOS, EOS, SIL]

    @dataclass
    class Filter:
        max_image_width: int = -1
        max_sequence_length: int = -1

        def accept_image(self, width: int) -> bool:
            return width < self.max_image_width

        def accept_sequence(self, length: int) -> bool:
            return length <= self.max_sequence_length

    # TODO Have an option to the stats command to generate this.
    @dataclass
    class Stats:
        max_chord: int = 12
        image_height: int = 256
        max_image_width: int = 3058
        image_mean: float = 22.06
        image_std: float = 62.78
        max_sequence_length: int = 207

    STATS = Stats()

    datadir: Path
    train_data: List[Path] = list([])
    valid_data: List[Path] = list([])
    tok2i: Dict[str, int]
    i2tok: Dict[int, str]

    image_height: int   # Image - constant - height in dataset.
    ipad_len: int       # Width for padding images, largers dropped.
    spad_len: int       # Length for padding sequences, longers dropped.

    transform: v2.Compose
    transform_and_norm: v2.Compose

    filter: Optional[Filter]

    @property
    def vocab_size(self) -> int:
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
                self.spad_len = 1 + filter.max_sequence_length
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

    _original_filter: Optional[Filter] = None

    @contextmanager
    def unfiltered(self):
        """Context to cancel filter, allowing access to very long images and sequences."""
        assert self._original_filter is None, "Can't nest unfiltered contexts."
        self._original_filter = self.filter
        self.filter = None
        try:
            yield
        finally:
            self.filter = self._original_filter
            self._original_filter = None

    def list(
        self, create: bool = False, refresh: bool = False, split: float = 0.9
    ) -> Tuple[int, int]:
        """
        Lists all samples available in the datadir, and caches the train/valid split.

        Args:
            create (bool, optional): Creates the cache if iit doesn't exist. Defaults to False.
            refresh (bool, optional): Force a cache refresh. Defaults to False.

        Raises:
            FileNotFoundError: If create is False and the cache doesn't exist.

        Returns:
            Tuple[int, int]: train and valid sizes.
        """
        list_path = Path(self.datadir) / 'list.pickle'
        if list_path.exists() and not refresh:
            with open(list_path, "rb") as f:
                obj = pickle.load(f)
                self.train_data, self.valid_data = obj["train_data"], obj["valid_data"]
        elif create or refresh:
            data = []
            for root, _, filenames in os.walk(self.datadir):
                for filename in filenames:
                    path = Path(root) / filename
                    if path.suffix == '.tokens' and path.with_suffix(".jpg").exists():
                        data.append(path.with_suffix(""))
            random.shuffle(data)
            train_size = int(len(data) * split)
            self.train_data = data[:train_size]
            self.valid_data = data[train_size:]
            with open(list_path, "wb+") as f:
                pickle.dump({
                    "train_data": self.train_data,
                    "valid_data": self.valid_data
                }, f)
            print(
                f"{len(data):,} samples found, " +
                f"split: {len(self.train_data):,}/{len(self.valid_data):,}."
            )
        else:
            raise FileNotFoundError(f"List file {list_path} not found.")
        # Loads the set of samples.
        return len(self.train_data), len(self.valid_data)

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
            self.create_vocab(save=True)
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

    def create_vocab(self, save: bool = True):
        self.tok2i = {key: value for key,               # type: ignore
                      value in self.RESERVED_TOKENS}
        self.i2tok = {value: key for value, key in self.RESERVED_TOKENS}
        token_count = len(self.tok2i)
        for path in self.train_data + self.valid_data:
            file = path.with_suffix(".tokens")
            with open(file, "r") as input:
                for line in input:
                    for token in line.strip().split("\t"):
                        token_count += 1
                        if self.tok2i.get(token, None) is None:
                            token_id = len(self.tok2i)
                            self.tok2i[token] = token_id
                            self.i2tok[token_id] = token
        if save:
            self.save_vocab()
        print(f"{token_count:,} tokens, {len(self.tok2i):,} uniques.")

    def load_sequence(
        self,
        path: Path,
        pad: bool = False,
        device: Optional[DeviceType] = None
    ) -> Tuple[Optional[torch.Tensor], int]:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with open(path, "r") as file:
            records = list(file)
            width = len(records)
            if self.filter and not self.filter.accept_sequence(width+2):
                return None, 0
            length = self.spad_len if pad else width+2
            assert len(records)+2 <= length, f"{path} length {
                len(records)} exceeds padding length {self.spad_len}"
            tensor = torch.full(
                (length, self.Stats.max_chord), self.PAD[0]).to(device)
            tensor[0, :], tensor[1+width, :] = self.SOS[0], self.EOS[0]
            for idx, record in enumerate(records):
                row = torch.Tensor([
                    self.tok2i.get(tok, self.UNK[0])for tok in record.strip().split()
                ])
                tensor[1+idx, len(row):] = self.SIL[0]
                tensor[1+idx, :len(row)] = row
        return tensor, width+2

    def decode(self, tokens: Union[torch.Tensor, List[int]]):
        if isinstance(tokens, torch.Tensor):
            tokens = [int(id.item()) for id in tokens]
        return [self.i2tok.get(token, "UNK") for token in tokens]

    def display(self, seq: torch.Tensor):
        """Displays a sequence, that is a Tensor representing a sequence of chords.

        Each chord is displayed as a line.

        Args:
            seq (torch.Tensor): Tensor of shape (any, Stats.max_chord)
        """
        for chord in seq:
            # Skips SOS and EOS.
            if all([id.item() == GrandPiano.SOS[0] for id in chord]):
                continue
            if all([id.item() == GrandPiano.EOS[0] for id in chord]):
                continue
            # Otherwise, displays anything but PAD.
            if any([id != GrandPiano.PAD[0] for id in chord]):
                texts = self.decode([
                    int(id.item()) for id in chord if id != GrandPiano.SIL[0]
                ])
                print("\t".join([text for text in texts if text]))

    def load_image(
        self, path: Path, norm: bool = True, pad: bool = False, device: Optional[DeviceType] = None
    ) -> Tuple[Optional[torch.Tensor], int]:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        image = decode_image(Path(path).as_posix()).to(device)
        image = (self.transform_and_norm if norm else self.transform)(image)
        image = image.squeeze(0).permute(1, 0)
        width, height = image.shape
        if self.filter and not self.filter.accept_image(width+2):
            return None, 0
        length = self.ipad_len if pad else width+2
        assert width+2 <= length, f"{
            path} width {width} exceeds padding width {self.ipad_len}"
        tensor = torch.full(
            (length, height), self.PAD[0], dtype=torch.float32).to(device)
        tensor[0, :], tensor[1+width,
                             :] = float(self.SOS[0]), float(self.EOS[0])
        tensor[1:1+width, :] = image
        return tensor, width+2

    def get_dataset(self, dataset_name: DatasetName) -> List[Path]:
        match dataset_name:
            case "train":
                return self.train_data
            case "valid":
                return self.valid_data
            case "all":
                return self.train_data + self.valid_data
            case _:
                raise ValueError(f"Invalid dataset '{dataset}'.")

    def len(self, dataset_name: DatasetName) -> int:
        return len(self.get_dataset(dataset_name))

    def next(
        self,
        dataset_name: DatasetName,
        pad: bool = False,
        device: Optional[DeviceType] = None
    ) -> Tuple[Path, torch.Tensor, int, torch.Tensor, int]:
        dataset = self.get_dataset(dataset_name)
        while True:
            position = random.randint(0, len(dataset) - 1)
            path = dataset[position]
            image, width = self.load_image(
                path.with_suffix(".jpg"), pad=pad, device=device)
            sequence, length = self.load_sequence(
                path.with_suffix(".tokens"), pad=pad, device=device)
            if image is not None and sequence is not None:
                return (path, image, width, sequence, length)

    @ staticmethod
    def sequence_length(args: Tuple['GrandPiano', Path]) -> int:
        gp, path = args
        _, length = gp.load_sequence(path)
        return length

    def sequences_length(self, dataset_name: DatasetName = "all") -> torch.Tensor:
        dataset = self.get_dataset(dataset_name)
        stats = map(GrandPiano.sequence_length, [
            (self, path.with_suffix(".tokens")) for path in dataset])
        return torch.tensor([length for length in stats if length > 0], dtype=torch.int)

    @staticmethod
    def image_stats(args: Tuple['GrandPiano', Path]) -> Optional[Tuple[int, float, float]]:
        gp, path = args
        image, _ = gp.load_image(path.with_suffix(".jpg"), norm=False)
        if image is not None:
            return image.shape[0], image.mean(dim=[0, 1]).item(), image.std(dim=[0, 1]).item()
        else:
            return None

    def images_length(self, dataset_name: DatasetName = "all") -> torch.Tensor:
        dataset = self.get_dataset(dataset_name)
        stats = map(GrandPiano.image_stats, [
            (self, path.with_suffix(".jpg")) for path in dataset])
        stats = [stat for stat in stats if stat is not None]
        return torch.Tensor([l for l, m, s in stats])

    def images_stats(self, dataset_name: DatasetName = "all") -> Dict[str, str]:
        dataset = self.get_dataset(dataset_name)
        stats = map(GrandPiano.image_stats, [
            (self, path.with_suffix(".jpg")) for path in dataset])
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
            "train dataset size": f"{len(self.train_data):,}",
            "valid dataset size": f"{len(self.valid_data):,}",
            "vocab size": f"{len(self.tok2i):,}",
            "sequence min length": f"{torch.min(sequence_lengths).item()}",
            "sequence max length": f"{torch.max(sequence_lengths).item()}",
            "sequence avg length": f"{torch.sum(sequence_lengths).item() / len(sequence_lengths):.2f}",
        } | image_stats


@click.command()
@click.pass_context
def make_vocab(ctx):
    """
        Creates the vocabulary file 'vocab.pickle' for the DATASET.
    """
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    gp = context.require_dataset()
    gp.create_vocab()


@click.command()
@click.option("--split", type=click.FloatRange(0.0, 1.0), default=0.9,
              help="Split train / valid data in this ratio.")
@click.pass_context
def refresh_list(ctx, split: float):
    """Splits the dataset into train abd valid, and caches it.
    """
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    gp = context.require_dataset()
    gp.list(refresh=True, split=split)


@click.command()
@click.pass_context
def stats(ctx):
    """Computes various statistics on the dataset.

    Some of these statistics should be incorporated into the code via
    the Stats dataclass.
    """
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    gp = context.require_dataset()
    for key, value in gp.stats().items():
        print(f"{key:<20}: {value}")


@click.command()
@click.pass_context
def histo(ctx):
    """

        Plots the histogram of sequence and image lengths.

    """
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    gp = context.require_dataset()
    sequence_lengths = gp.sequences_length()
    image_lengths = gp.images_length()
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.hist(sequence_lengths.numpy(), bins=50, cumulative=True, density=True)
    ax1.set_xlabel('Sequence lengths')
    ax2.hist(image_lengths.numpy(), bins=50, cumulative=True, density=True)
    ax2.set_xlabel('Image widths')
    fig.tight_layout()
    plt.show()


@click.command()
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
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    gp = context.require_dataset()
    match Path(path).suffix:
        case ".tokens":
            tensor = gp.load_sequence(path, pad=pad)
        case ".jpg":
            tensor = gp.load_image(path, pad=pad)
        case _:
            raise ValueError(
                "Files of {path.suffix} suffixes can't be tokenized.")
    print(tensor)
