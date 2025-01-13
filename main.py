#!/usr/bin/env python3

import json
from pathlib import Path
from typing import List, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from grandpiano import GrandPiano, histo, load, make_vocab, refresh_list, stats
from train import predict


def moving_average(y: NDArray[np.float32], window_size: int = 10) -> NDArray[np.float32]:
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')


@click.command
@click.argument("jsonpath",
                type=click.Path(file_okay=True),
                default=Path("untracked/train_log.json"))
@click.option("--smooth/--no-smooth", default=True,
              help="Smooth the curves before plotting them.")
def plot(jsonpath: Path, smooth: bool):
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
    loss, = ax.plot([], [], 'b', label='Train loss.')
    vloss, = ax.plot([], [], 'r', label='Valid loss.')
    print("Press 'q' to quit.")

    while not quit:
        with open(jsonpath, "r") as f:
            log = json.load(f)
        losses, vlosses = log["losses"], log["vlosses"]
        if smooth:
            losses = moving_average(np.array(losses, dtype=np.float32))
            vlosses = moving_average(np.array(vlosses, dtype=np.float32))
        loss.set_xdata(range(0, len(losses)))
        loss.set_ydata(losses)
        vloss.set_xdata(range(0, len(vlosses)))
        vloss.set_ydata(vlosses)
        ax.relim()
        ax.autoscale_view()
        ax.legend()
        fig.canvas.draw_idle()
        plt.pause(1)


@click.group
@click.option('--dataset', '-d', 'datadir',
              type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='/home/anselm/Downloads/GrandPiano')
@click.pass_context
def cli(ctx: click.Context, datadir: Path):
    ctx.obj = GrandPiano(
        datadir,
        filter=GrandPiano.Filter(max_image_width=1024, max_sequence_length=128)
    )


cli.add_command(make_vocab)
cli.add_command(stats)
cli.add_command(load)
cli.add_command(refresh_list)
cli.add_command(histo)
cli.add_command(plot)
cli.add_command(predict)

if __name__ == '__main__':
    cli()
