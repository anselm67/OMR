#!/usr/bin/env python3

from pathlib import Path

import click

from click_context import ClickContext
from client import infos, predict, random_check
from dataset import init_dataset
from logger import plot
from tokenizer import kern_stats, tokenize
from train2 import test, train


@click.group
@click.option('--dataset-directory', '-d', 'dataset_directory',
              type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default='/home/anselm/datasets/GrandPiano')
@click.option('--model-directory', '-m', 'model_directory',
              type=click.Path(file_okay=False, dir_okay=True, writable=True),
              default="untracked/train")
@click.pass_context
def cli(
    ctx: click.Context,
    dataset_directory: Path,
    model_directory: Path
):
    ctx.obj = ClickContext(Path(dataset_directory), Path(model_directory))


# dataset dataset commands:
cli.add_command(init_dataset)

# Training commands:
cli.add_command(train)
cli.add_command(test)
cli.add_command(plot)

# Client commands:
cli.add_command(infos)
cli.add_command(predict)
cli.add_command(random_check)

# kernnom / aka tokenizer commands.
cli.add_command(tokenize)
cli.add_command(kern_stats)

if __name__ == '__main__':
    cli()
