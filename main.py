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
    """

    Plots and tracks the training losses emitted while training the model.
    """
    # State and function to quit the tracking loop.
    quit: bool = False

    def on_key(event):
        nonlocal quit
        quit = (event.key == 'q')

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_key)
    line, = ax.plot([], [])
    print("Press 'q' to quit.")

    while not quit:
        with open(jsonpath, "r") as f:
            log = json.load(f)
        losses = log["losses"]
        line.set_xdata(range(0, len(losses)))
        line.set_ydata(losses)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(1)


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
