
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, cast

import click
import torch
import torch.nn.functional as F
from torchvision.io import decode_image
from torchvision.transforms import v2


class GrandPiano:
    CHORD_MAX = 12          # Maximum number of concurrent notes in dataset.

    PAD = (0, "PAD")        # Sequence vertical aka chord padding value.
    UNK = (1, "UNK")        # Unknown sequence token.
    SOS = (2, "SOS")        # End of sequence token.
    EOS = (3, "EOS")        # Beginning of sequence token.
    RESERVED_TOKENS = [PAD, UNK, EOS, SOS]

    datadir: Path
    data: List[Path] = list([])
    tok2i: Dict[str, int]
    i2tok: Dict[int, str]
    position: int = 0

    image_height: int   # Image - constant - height in dataset.
    ipad_len: int       # Width for padding images, largers dropped.
    spad_len: int       # Length for padding sequences, longers dropped.

    @property
    def vocab_size(self):
        return len(self.tok2i)

    def __init__(self,
                 datadir: Path,
                 image_height: int = 256,
                 ipad_len: int = 2048,
                 spad_len: int = 100):
        self.datadir = datadir
        self.image_height = image_height
        self.ipad_len = ipad_len
        self.spad_len = spad_len
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

    def load_sequence(self, path: Path, pad: bool = False) -> Tuple[torch.Tensor, int]:
        with open(path, "r") as file:
            records = list(file)
            width = len(records)
            length = self.spad_len if pad else width+2
            assert len(records)+2 <= length, f"{path} length {
                len(records)} exceeds padding length {self.spad_len}"
            tensor = torch.full((length, self.CHORD_MAX), self.PAD[0])
            tensor[0, :], tensor[1+width, :] = self.SOS[0], self.EOS[0]
            for idx, record in enumerate(records):
                row = torch.Tensor([
                    self.tok2i.get(tok, self.UNK[0])for tok in record.strip().split()
                ])
                tensor[1+idx, :len(row)] = row
        return tensor, width+2

    def decode(self, tokens: List[int]):
        return [self.i2tok.get(token, "UNK") for token in tokens]

    TRANSFORM = v2.Compose([
        v2.Grayscale()
    ])
    TRANSFORM_AND_NORM = v2.Compose([
        TRANSFORM,
        v2.Normalize(mean=[227.11], std=[62.71], inplace=True)
    ])

    def load_image(
        self, path: Path, norm: bool = True, pad: bool = False
    ) -> Tuple[torch.Tensor, int]:
        image = decode_image(Path(path).as_posix()).to(torch.float32)
        image = (self.TRANSFORM_AND_NORM if norm else self.TRANSFORM)(image)
        image = image.squeeze(0).permute(1, 0)
        width, height = image.shape
        length = self.ipad_len if pad else width+2
        assert width+2 <= length, f"{
            path} width {width} exceeds padding width {self.ipad_len}"
        tensor = torch.full((length, height), self.PAD[0], dtype=torch.float32)
        tensor[0, :], tensor[1+width,
                             :] = float(self.SOS[0]), float(self.EOS[0])
        tensor[1:1+width, :] = image
        return tensor, width+2

    def next(self, pad: bool = False) -> Tuple[torch.Tensor, int, torch.Tensor, int]:
        if self.position >= len(self.data):
            self.position = 0
        path = self.data[self.position]
        self.position += 1
        return (
            *self.load_image(path.with_suffix(".jpg"), pad=pad),
            *self.load_sequence(path.with_suffix(".tokens"), pad=pad)
        )

    @ staticmethod
    def sequence_length(args) -> int:
        gp, path = args
        return len(gp.load_sequence(path))

    def sequences_length(self) -> torch.Tensor:
        with ProcessPoolExecutor(2) as executor:
            return torch.tensor(list(executor.map(GrandPiano.sequence_length, [
                (self, path.with_suffix(".tokens")) for path in self.data], chunksize=500)),
                dtype=torch.int
            )

    @staticmethod
    def image_stats(args) -> Tuple[int, float, float]:
        gp, path = args
        image = gp.load_image(path.with_suffix(".jpg"), norm=False)
        return image.shape[1], image.mean(dim=[0, 1]), image.std(dim=[0, 1])

    def images_stats(self) -> Dict[str, str]:
        with ProcessPoolExecutor(2) as executor:
            stats = list(executor.map(GrandPiano.image_stats, [
                (self, path.with_suffix(".jpg")) for path in self.data], chunksize=500))
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


@ click.command
@ click.pass_context
def make_vocab(ctx):
    """
        Creates the vocabulary file 'vocab.pickle' for the DATASET.
    """
    gp = ctx.obj
    gp.create_vocab()


@ click.command
@ click.pass_context
def refresh_list(ctx):
    """
        Creates the vocabulary file 'vocab.pickle' for the DATASET.
    """
    gp = cast(GrandPiano, ctx.obj)
    gp.list(refresh=True)


@ click.command
@ click.pass_context
def stats(ctx):
    """
        Creates the vocabulary file 'vocab.pickle' for the DATASET.
    """
    gp = cast(GrandPiano, ctx.obj)
    for key, value in gp.stats().items():
        print(f"{key:<20}: {value}")


@ click.command
@ click.argument("path",
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
