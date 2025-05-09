
import json
import logging
from pathlib import Path
from typing import Any, cast

import click
import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.loggers import Logger
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray


class SimpleLogger(Logger):

    log_path: Path
    metrics: dict[str, list[tuple[int, float]]]
    _name: str

    def __init__(self, log_path: Path, name: str = "logger-name"):
        super().__init__()
        self.log_path = log_path
        self._name = name
        self._load_if_exists()

    def _load_if_exists(self):
        self.metrics = dict()
        if self.log_path.exists():
            with open(self.log_path, "r") as fp:
                self.metrics = json.load(fp)

    def reload(self) -> dict[str, list[tuple[int, float]]]:
        self._load_if_exists()
        return self.metrics

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "1.0"

    def log_hyperparams(self, params: dict[str, Any]):
        logging.info("Not logging hyper parameters.")

    def log_metrics(self, metrics: dict[str, float], step: int):
        for k, v in metrics.items():
            values = self.metrics.get(k, None)
            if values is None:
                self.metrics[k] = [(step, v)]
            else:
                values.append((step, v))

    def save(self):
        with open(self.log_path, "w+") as fp:
            json.dump(self.metrics, fp, indent=2)

    def finalize(self, status: str):
        logging.info(f"Finalizing log with {status}")


def moving_average(y: NDArray[np.float32], window_size: int = 10) -> NDArray[np.float32]:
    if len(y) < window_size:
        return y
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')


@click.command()
@click.option("--smooth/--no-smooth", default=True,
              help="Smooth the curves before plotting them.")
@click.option("--hide", "-h", multiple=True, type=str,
              help="Hide metric from plot, multiple allowed.")
@click.pass_context
def plot(ctx, hide: list[str], smooth: bool = True):
    """Plots the selected model training log metrics.

    Args:
        hide (list[str]): Don't plot these metrics.
        smooth (bool, optional): Smoothes the curves before plotting them.
    """
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    logger = context.require_logger()

    # State and function to quit the tracking loop.
    quit: bool = False

    def on_key(event):
        nonlocal quit
        quit = (event.key == 'q')

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_key)
    print("Press 'q' to quit.")

    lines = {}

    while not quit:

        metrics = logger.reload()
        for key, metric in metrics.items():
            if key in hide:
                continue
            line = lines.get(key, None)
            if line is None:
                line, = ax.plot([], [], label=key)
                lines[key] = line

            x, y = zip(*metric)
            if smooth:
                y = moving_average(np.array(y))
                x = x[-len(y):]
            line.set_xdata(x)
            line.set_ydata(y)
            ax.relim()
            ax.autoscale_view()
            ax.legend()
        fig.canvas.draw_idle()
        plt.pause(1)
