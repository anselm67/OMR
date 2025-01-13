#!/usr/bin/env python3

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt

from grandpiano import GrandPiano, histo, load, make_vocab, refresh_list, stats


@click.command
@click.argument("jsonpath",
                type=click.Path(file_okay=True),
                default=Path("untracked/train_log.json"))
def plot(jsonpath: Path):
    with open(jsonpath, "r") as f:
        log = json.load(f)
    plt.plot(log["losses"])
    plt.show()


@click.group
@click.option('--dataset', '-d', 'datadir',
              type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='/home/anselm/Downloads/GrandPiano')
@click.pass_context
def cli(ctx: click.Context, datadir: Path):
    ctx.obj = GrandPiano(
        datadir, filter=GrandPiano.Filter(max_image_width=1024, max_sequence_length=128))


cli.add_command(make_vocab)
cli.add_command(stats)
cli.add_command(load)
cli.add_command(refresh_list)
cli.add_command(histo)
cli.add_command(plot)

if __name__ == '__main__':
    cli()
