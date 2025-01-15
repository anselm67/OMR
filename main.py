#!/usr/bin/env python3

import json
from pathlib import Path
from typing import cast

import click
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from click_context import ClickContext
from client import infos, predict, random_check
from grandpiano import histo, load, make_vocab, refresh_list, stats
from kernnorm import tokenize
from train import train


def moving_average(y: NDArray[np.float32], window_size: int = 10) -> NDArray[np.float32]:
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')


@click.command()
@click.option("--smooth/--no-smooth", default=True,
              help="Smooth the curves before plotting them.")
@click.pass_context
def plot(ctx, smooth: bool):
    context = cast(ClickContext, ctx.obj)
    jsonpath = context.get_train_log()
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
        losses, valid_losses = log["losses"], log["valid_losses"]
        if smooth:
            losses = moving_average(np.array(losses, dtype=np.float32))
            valid_losses = moving_average(
                np.array(valid_losses, dtype=np.float32))
        loss.set_xdata(range(0, len(losses)))
        loss.set_ydata(losses)
        vloss.set_xdata(range(0, len(valid_losses)))
        vloss.set_ydata(valid_losses)
        ax.relim()
        ax.autoscale_view()
        ax.legend()
        fig.canvas.draw_idle()
        plt.pause(1)


@click.group
@click.option('--dataset-path', '-d', 'dataset_path',
              type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='/home/anselm/Downloads/GrandPiano')
@click.option('--model-path', '-m', 'model_path',
              type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default="untracked/")
@click.option('--name', '-n', 'model_name', type=str, default="model")
@click.pass_context
def cli(ctx: click.Context, dataset_path: Path, model_path: Path, model_name: str):
    ctx.obj = ClickContext(Path(dataset_path), Path(model_path), model_name)


# GrandPiano dataset commands:
cli.add_command(make_vocab)
cli.add_command(stats)
cli.add_command(load)
cli.add_command(refresh_list)
cli.add_command(histo)

# Training commands:
cli.add_command(plot)
cli.add_command(train)

# Client commands:
cli.add_command(infos)
cli.add_command(predict)
cli.add_command(random_check)

# kernnom / aka tokenizer commands.
cli.add_command(tokenize)

if __name__ == '__main__':
    cli()
