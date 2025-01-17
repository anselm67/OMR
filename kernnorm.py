#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, TextIO, Tuple, Type, cast

import click

from kern.parser import Parser
from kern.typing import (
    Bar,
    Chord,
    Clef,
    Continue,
    Duration,
    Key,
    Meter,
    Note,
    Pitch,
    Rest,
    SpinePath,
    Token,
)


class Spine:

    in_beam: int = 0
    in_tie: bool = False


class IgnoredSpine(Spine):

    pass


class TokenFormatter:
    formatters: Dict[Type, Callable[[Spine, Token], str]]

    def __init__(self):
        self.formatters = {
            Bar: self.format_bar,
            Rest: self.format_rest,
            Clef: self.format_clef,
            Key: self.format_key,
            Meter: self.format_meter,
            Continue: self.format_continue,
            Note: self.format_note,
            Chord: self.format_chord,
            SpinePath: self.format_spine_path,
        }

    def format_unknown(self, spine: Spine, token: Token) -> str:
        raise ValueError(f"No format for token {token}")

    def format_duration(self, duration: Duration) -> str:
        match duration:
            case Duration(duration=d, dots=0):
                return str(d)
            case _:
                return f"{duration.duration}:{duration.dots}"

    def format_pitch(self, pitch: Pitch) -> str:
        return pitch.name

    def format_bar(self, spine: Spine, _: Token) -> str:
        return "="

    def format_rest(self, spine: Spine, token: Token) -> str:
        rest = cast(Rest, token)
        return f"rest/{self.format_duration(rest.duration)}"

    def format_clef(self, spine: Spine, token: Token) -> str:
        clef = cast(Clef, token)
        return f"clef-{self.format_pitch(clef.pitch)}"

    def format_key(self, spine: Spine, token: Token) -> str:
        key = cast(Key, token)
        return f"key{("-" if key.is_flats else "#") * key.count}"

    def format_meter(self, spine: Spine, token: Token) -> str:
        meter = cast(Meter, token)
        return f"{meter.numerator}/{meter.denominator}"

    def format_continue(self,  spine: Spine, _: Token) -> str:
        return "."

    def format_note(self, spine: Spine, token: Token) -> str:
        note = cast(Note, token)
        accidentals = ("#" * note.sharps) or ("-" * note.flats)
        duration_text = ""
        if (duration := note.duration) is None:
            assert note.is_gracenote, "Only gracenotes don't have duration."
            duration_text = "/q"
        else:
            duration_text = f"/{self.format_duration(duration)}"
        text = (
            self.format_pitch(note.pitch) + accidentals +
            duration_text
        )
        return text

    def format_chord(self, spine: Spine, token: Token) -> str:
        chord = cast(Chord, token)
        text = "\t".join([self.format_note(spine, note)
                          for note in chord.notes])
        return text

    def format_spine_path(self, spine: Spine, token: Token) -> str:
        return self.format_continue(spine, Continue())

    def format(self, spine: Spine, token: Token) -> str:
        return self.formatters.get(
            token.__class__, self.format_unknown)(spine, token)


class NormHandler(Parser[Spine].Handler):

    formatter: TokenFormatter = TokenFormatter()
    output: Optional[TextIO]
    spines: List[Spine]

    def __init__(self, output_path: Optional[Path]):
        super(NormHandler, self).__init__()
        self.spines = list([])
        self.output = output_path and open(output_path, 'w+')

    def position(self, spine) -> int:
        return self.spines.index(spine)

    def open_spine(self,
                   spine_type: Optional[str] = None,
                   parent: Optional[Spine] = None) -> Spine:
        match spine_type:
            case "**dynam":
                spine = IgnoredSpine()
            case _:
                spine = Spine()
        self.spines.append(spine)
        return spine

    def close_spine(self, spine: Spine):
        self.spines.remove(spine)

    def branch_spine(self, source: Spine) -> Spine:
        branch = Spine()
        self.spines.insert(self.position(source), branch)
        return branch

    def merge_spines(self, source: Spine, into: Spine):
        # The source will be close_spine() by the parser.
        pass

    def rename_spine(self, spine: Spine, name: str):
        pass

    last_metric: Optional[Meter] = None

    def should_skip(self, tokens: List[Tuple[Spine, Token]]) -> bool:
        # Sometimes we get both a 4/4 and C meter, skip.
        token = tokens[0][1]
        if isinstance(token, Meter) and all([t == token for _, t in tokens]):
            if token == self.last_metric:
                return True
            self.last_metric = token
        else:
            self.last_metric = None
        # Pure spine paths aren't interesting to us.
        if all([isinstance(token, SpinePath) for _, token in tokens]):
            return True
        return False

    def append(self, tokens: List[Tuple[Spine, Token]]):
        tokens = [(spine, token) for spine, token in tokens
                  if not isinstance(spine, IgnoredSpine)]
        if self.should_skip(tokens):
            return
        if self.output:
            self.output.write('\t'.join([
                self.formatter.format(spine, token)for spine, token in tokens
            ]) + "\n")

    def finish(self):
        if self.output:
            self.output.close()


def tokenize_file(path: Path, write_output: bool = True, show_failed: bool = True) -> bool:
    try:
        output_path = path.with_suffix(".tokens") if write_output else None
        handler = NormHandler(output_path)
        h = Parser.from_file(path, handler)
        h.parse()
        handler.finish()
        return True
    except Exception as e:
        if show_failed:
            print(f"{path.name}: {e}")
        return False


def tokenize_directory(
    path: Path, write_output: bool = True, show_failed: bool = True
) -> Tuple[int, int]:
    count, failed = 0, 0
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            path = Path(root) / filename
            if path.suffix == '.krn' and not path.name.startswith("."):
                count += 1
                if not tokenize_file(path, write_output, show_failed):
                    failed += 1
                    print(f"{path}")
    return count, failed


@click.command()
@click.argument("source", nargs=-1,
                type=click.Path(dir_okay=True, exists=True, readable=True),
                required=True)
@click.option("--show-failed/--no-show-failed", "show_failed", default=True,
              help="Displays the path of files we failed to tokenize.")
@click.option("--no-output", "no_output", is_flag=True,
              help="Don't write out the .tokens file, simply parse.")
def tokenize(source: List[Path], no_output: bool = False, show_failed: bool = True):
    """Parses all .kern files in DATADIR and outputs a .tokens file.

    The .tokens file is essentially a simplified, normal form for the kern files:
    - It removes any info not needed for score playback, such as beams.
    - It collects all chors in the equivalent of a single "spine".

    Args:
        datadir (Path): The directory to inspect and tokenize.
    """
    count, failed = 0, 0
    for path in [Path(s) for s in source]:
        if path.is_dir():
            dir_count, dir_failed = tokenize_directory(
                path, write_output=not no_output, show_failed=show_failed)
            count += dir_count
            failed += dir_failed
        else:
            count += 1
            if not tokenize_file(path, write_output=not no_output, show_failed=show_failed):
                failed += 1
    print(f"Tokenized {count} files, {failed} failed.")
