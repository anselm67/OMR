#!/usr/bin/env python3

from pathlib import Path

import click

from grandpiano import GrandPiano, load, make_vocab, stats


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
