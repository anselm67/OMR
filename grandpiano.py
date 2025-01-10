#!/usr/bin/env python3

import os
import pickle
from pathlib import Path
from typing import Dict, List

import click
import torch
import torch.nn.functional as F


class GrandPiano:
    CHORD_MAX = 6

    PAD = (0, "PAD")        # Sequence vertical aka chord padding value.
    UNK = (1, "UNK")        # Unknown sequence token.
    EOS = (2, "EOS")        # Beginning of sequence token.
    BOS = (3, "BOS")        # End of sequence token.
    RESERVED_TOKENS = [UNK, EOS, BOS]

    datadir: Path
    data: List[Path] = list([])
    tok2i: Dict[str, int]
    i2tok: Dict[int, str]

    def __init__(self, datadir: Path):
        self.datadir = datadir
        # self.list()
        self.load_vocab(create=True)

    def list(self) -> int:
        # Loads the set of samples.
        for root, _, filenames in os.walk(self.datadir):
            for filename in filenames:
                path = Path(root) / filename
                if path.suffix == '.tokens' and path.with_suffix(".jpg").exists():
                    self.data.append(path.with_suffix(""))
        return len(self.data)

    def load_vocab(self, create: bool = False):
        # Loads the vocab for sequences.
        vocab_path = Path(self.datadir, "vocab.pickle")
        if not vocab_path.exists():
            if create:
                self.create_vocab()
                self.save_vocab()
                return
            else:
                raise FileNotFoundError(f"Pickle file {vocab_path} not found.")
        # Reads in the existing vocab file.
        with open(vocab_path, "rb") as f:
            obj = pickle.load(f)
        self.tok2i = obj['tok2i']
        self.i2tok = obj['i2tok']

    def save_vocab(self):
        assert self.tok2i and self.i2tok, "Vocab not computed yet."
        vocab_path = Path(self.datadir, "vocab.pickle")
        with open(vocab_path, "wb+") as f:
            pickle.dump({
                "tok2i": self.tok2i,
                "i2tok": self.i2tok
            }, f)

    def create_vocab(self):
        self.tok2i = {key: value for key, value in self.RESERVED_TOKENS}
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

    def load_sequence(self, path: Path) -> torch.Tensor:
        tensor = torch.empty((1, self.CHORD_MAX), dtype=torch.int)
        with open(path, "r") as file:
            for line in file:
                chord = torch.Tensor([
                    self.tok2i.get(tok, self.UNK[0])for tok in line.strip().split()
                ])
                chord = F.pad(chord, (0, self.CHORD_MAX -
                              chord.shape[0]), value=self.PAD[0])
                tensor = torch.cat((tensor, chord.unsqueeze(0)), dim=0)
        return tensor


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
def stats(ctx):
    """
        Creates the vocabulary file 'vocab.pickle' for the DATASET.
    """
    gp = ctx.obj
    print(f"{len(gp.data):,} samples, {len(gp.tok2i):,} tokens.")


@click.command
@click.argument("path",
                type=click.Path(file_okay=True),
                required=True)
@click.pass_context
def load(ctx, path: Path):
    """
        Loads and tokenizes PATH.

        PATH can be either .tokens or a .jpg file, and will be tokenized accordingly. 
    """
    gp = ctx.obj
    if Path(path).suffix == ".tokens":
        gp.load_sequence(path)
    else:
        raise ValueError("Files of {path.suffix} suffixes can't be tokenized.")


@click.group
@click.option('--dataset', '-d', 'datadir',
              type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='/home/anselm/Downloads/GrandPiano')
@click.pass_context
def cli(ctx: click.Context, datadir: Path):
    ctx.obj = GrandPiano(datadir)


cli.add_command(make_vocab)
cli.add_command(stats)
cli.add_command(load)

if __name__ == '__main__':
    cli()
