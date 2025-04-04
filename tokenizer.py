#!/usr/bin/env python3

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO, Type, cast

import click

from kern import (
    Bar,
    Chord,
    Clef,
    Comment,
    Continue,
    Duration,
    DurationToken,
    Key,
    Meter,
    Note,
    Parser,
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
    formatters: dict[Type, Callable[[Token], str]]
    barno: int
    display_bar_number: bool

    def __init__(self):
        self.barno = 1
        self.display_bar_number = False
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

    def format_unknown(self, token: Token) -> str:
        raise ValueError(f"No format for token {token}")

    def format_duration(self, duration: Duration) -> str:
        match duration:
            case Duration(duration=d, dots=0):
                return str(d)
            case _:
                return f"{duration.duration}:{duration.dots}"

    def format_pitch(self, pitch: Pitch) -> str:
        return pitch.name

    def format_bar(self, token: Token) -> str:
        bar = cast(Bar, token)
        if self.display_bar_number:
            barno_str = str(bar.barno) if bar.barno >= 0 else ""
        else:
            barno_str = ""
        if bar.is_final:
            return f"=={barno_str}"
        else:
            return f"={barno_str}"

    def format_rest(self, token: Token) -> str:
        rest = cast(Rest, token)
        assert rest.duration
        return f"rest/{self.format_duration(rest.duration)}"

    def format_clef(self, token: Token) -> str:
        clef = cast(Clef, token)
        return f"clef-{self.format_pitch(clef.pitch)}"

    def format_key(self, token: Token) -> str:
        key = cast(Key, token)
        return f"key{("-" if key.is_flats else "#") * key.count}"

    def format_meter(self, token: Token) -> str:
        meter = cast(Meter, token)
        return f"{meter.numerator}/{meter.denominator}"

    def format_continue(self, _: Token) -> str:
        return "."

    def format_note(self, token: Token) -> str:
        note = cast(Note, token)
        accidentals = ("#" * note.sharps) or ("-" * note.flats)
        duration_text = ""
        if (duration := note.duration) is None:
            assert note.is_gracenote or note.is_groupetto, "Only gracenotes don't have duration."
            duration_text = "/q"
        else:
            duration_text = f"/{self.format_duration(duration)}"
        text = (
            self.format_pitch(note.pitch) + accidentals +
            duration_text
        )
        return text

    def format_chord(self, token: Token) -> str:
        chord = cast(Chord, token)
        text = "\t".join([self.format_note(note) for note in chord.notes])
        return text

    def format_spine_path(self, _: Token) -> str:
        return self.format_continue(Continue())

    def format(self, token: Token) -> str:
        text = self.formatters.get(token.__class__, self.format_unknown)(token)
        self.last_token = token
        return text


class BaseHandler(Parser[Spine].Handler):

    spines: list[Spine]

    def __init__(self):
        super(BaseHandler, self).__init__()
        self.spines = list([])

    def position(self, spine) -> int:
        return self.spines.index(spine)

    def open_spine(self,
                   spine_type: str | None = None,
                   parent: Spine | None = None) -> Spine:
        match spine_type:
            case "**dynam" | "**dynam/2" | "**mxhm" | "**recip" | "**fb":
                spine = IgnoredSpine()
            case _:
                spine = Spine()
        self.spines.append(spine)
        return spine

    def close_spine(self, spine: Spine):
        self.spines.remove(spine)

    def branch_spine(self, source: Spine) -> Spine:
        branch = type(source)()
        self.spines.insert(self.position(source), branch)
        return branch

    def merge_spines(self, source: Spine, into: Spine):
        # The source will be close_spine() by the parser.
        pass

    def rename_spine(self, spine: Spine, name: str):
        pass


class NormHandler(BaseHandler):

    formatter: TokenFormatter
    output: TextIO | None

    # The current bar number, when none provided.
    bar_numbering: bool
    bar_number: int
    bar_seen: bool
    bar_zero: bool

    def __init__(self, output_path: Path | None):
        super(NormHandler, self).__init__()
        self.output = output_path and open(output_path, 'w+')
        self.formatter = TokenFormatter()
        self.bar_numbering = False
        self.bar_number = 1
        self.bar_seen = False
        self.bar_zero = False

    last_metric: Meter | None = None

    def check_type(self, tokens: Iterable[Token], t: Type) -> bool:
        return all([isinstance(token, t) for token in tokens])

    def should_skip(self, tokens: list[tuple[Spine, Token]]) -> bool:
        # Sometimes we get both a 4/4 and C meter, skip.
        token = tokens[0][1]
        if isinstance(token, Meter) and all([t == token for _, t in tokens]):
            if token == self.last_metric:
                return True
            self.last_metric = token
        else:
            self.last_metric = None
        # Skips all Comment.
        if self.check_type((t for _, t in tokens), Comment):
            return True
        # Pure spine paths aren't interesting to us.
        if self.check_type((t for _, t in tokens), SpinePath):
            return True
        return False

    def fix_bar(self, tokens: list[tuple[Spine, Token]]) -> list[tuple[Spine, Token]] | None:

        def requires_bar(t: Token) -> bool:
            if isinstance(t, (Note, Chord, Rest)):
                return True
            # A non numbered repeat bar also requires a preceeding bar zero.
            if isinstance(t, Bar) and not cast(Bar, t).requires_valid_bar_number():
                return True
            return False

        # If we see a note or chord before any bar, emit a fake bar 0.
        if not self.bar_zero:
            if any([requires_bar(t) for _, t in tokens]):
                bar = Bar("*fake*", 0, False, False, False, False)
                if self.output:
                    self.output.write('\t'.join([
                        self.formatter.format(bar)for _, _ in tokens
                    ]) + "\n")
                self.bar_zero = True

        # Adjusts the bar number when none provided.
        if self.check_type((t for _, t in tokens), Bar):
            bars = [cast(Bar, token) for _, token in tokens]
            if self.bar_number <= 2:
                if all([bar.barno < 0 and bar.requires_valid_bar_number() for bar in bars]):
                    self.bar_numbering = True
                self.bar_zero = True

            if self.bar_numbering:
                bars = [replace(bar, barno=self.bar_number) for bar in bars]
                self.bar_number += 1
            elif (barno := max((bar.barno for bar in bars))) >= 0:
                self.bar_number = barno + 1
            elif any((bar.requires_valid_bar_number() for bar in bars if bar.barno < 0)):
                self.bar_number += 1

            bars = [
                replace(
                    bar, barno=self.bar_number) if bar.is_final and bar.barno < 0 else bar
                for bar in bars
            ]
            # TODO We're not supposed to see any more bars, so it's ok
            # not to incrememt self.bar_number

            if any((bar.barno >= 0 for bar in bars)):
                return list(zip([spine for spine, _ in tokens], bars))
            else:
                return None

        return tokens

    def unique(self, tokens: list[Token]) -> list[Token]:
        seen = set()
        return [t for t in tokens if not (t in seen or seen.add(t))]

    def flatten(self, tokens: list[tuple[Spine, Token]]) -> list[Token]:
        toks = [
            t for _, t in tokens if not isinstance(t, (Continue, SpinePath))
        ]
        if len(toks) == 0:
            return []
        if self.check_type(toks, Clef):
            toks = self.unique(toks)
            if len(toks) > 2:
                raise ValueError(
                    f"Got too many clefs ({len(toks)}), expected 2.")
        elif self.check_type(toks, Key):
            toks = self.unique(toks)
            if len(toks) != 1:
                raise ValueError(
                    f"Got too many keys ({len(toks)}), expected 1.")
            toks = [toks[0], toks[0]]
        elif self.check_type(toks, Meter):
            toks = self.unique(toks)
            if len(toks) != 1:
                raise ValueError(
                    f"Got too many meters ({len(toks)}), expected 1.")
            toks = [toks[0], toks[0]]
        elif self.check_type(toks, Bar):
            toks = self.unique(toks)
            if len(toks) != 1:
                raise ValueError(
                    f"Got too many bars ({len(toks)}), expected 1.")
        elif self.check_type(toks, Rest):
            toks = [max(toks)]
        else:
            notes = [note for n in toks for note in (
                n.notes if isinstance(n, Chord) else [n])]
            if self.check_type(notes, DurationToken):
                toks = sorted(notes)
            else:
                print(f"FIXME: got a mix of tokens.")

        return toks

    def append(self, tokens: list[tuple[Spine, Token]]):
        tokens = [(spine, token) for spine, token in tokens
                  if not isinstance(spine, IgnoredSpine)]
        if self.should_skip(tokens):
            return
        if not (fixed_bars := self.fix_bar(tokens)):
            return
        tokens = fixed_bars
        if self.output:
            self.output.write('\t'.join([
                self.formatter.format(tok) for tok in self.flatten(tokens)
            ]) + "\n")

    def done(self):
        if self.output:
            self.output.close()


class StatsHandler(BaseHandler):

    bar_count: int = 0
    chord_count: int = 0
    finish: Callable[['StatsHandler'], None] | None

    def __init__(self, finish: Callable[['StatsHandler'], None] | None = None):
        super(StatsHandler, self).__init__()
        self.finish = finish

    def has(self, cls, tokens) -> bool:
        return any([isinstance(token, cls) for _, token in tokens])

    def append(self, tokens: list[tuple[Spine, Token]]):
        tokens = [(spine, token) for spine, token in tokens
                  if not isinstance(spine, IgnoredSpine)]
        if self.has(Bar, tokens):
            self.bar_count += 1
        if self.has(Note, tokens) or self.has(Chord, tokens):
            self.chord_count += 1

    def done(self):
        if self.finish is not None:
            self.finish(self)


def parse_file(
    path: Path,
    handler_obj: Parser.Handler | Callable[[Path], Parser.Handler],
    show_failed: bool = True,
    enable_warnings: bool = False,
) -> bool:
    try:
        if isinstance(handler_obj, Parser.Handler):
            handler = cast(Parser.Handler, handler_obj)
        else:
            handler = cast(Callable[[Path], Parser.Handler], handler_obj)(path)
        parser = Parser.from_file(path, handler)
        parser.enable_warnings = enable_warnings
        parser.parse()
        return True
    except Exception as e:
        if show_failed:
            print(f"{path.name}: {e}")
        return False


def parse_directory(
    path: Path,
    handler_factory: Callable[[Path], Parser.Handler],
    show_failed: bool = True,
    enable_warnings: bool = False,
) -> tuple[int, int]:
    count, failed = 0, 0
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            path = Path(root) / filename
            if path.suffix == '.krn' and not path.name.startswith("."):
                count += 1
                if not parse_file(path, handler_factory, show_failed, enable_warnings):
                    failed += 1
                    print(f"{path}")
    return count, failed


def tokenize_directory(
    path: Path,
    write_output: bool = True,
    show_failed: bool = True,
    enable_warnings: bool = False,
) -> tuple[int, int]:
    def handler_factory(source: Path) -> Parser.Handler:
        output_path = source.with_suffix(".tokens") if write_output else None
        handler = NormHandler(output_path)
        return handler

    return parse_directory(path, handler_factory, show_failed, enable_warnings)


def stats_directory(
    path: Path,
    show_failed: bool = True,
    enable_warnings: bool = False,
) -> dict[str, Any]:
    chord_count = 0
    bar_count = 0

    def reduce(handler: StatsHandler):
        nonlocal chord_count, bar_count
        chord_count += handler.chord_count
        bar_count += handler.bar_count

    def handler_factory(_: Path) -> Parser.Handler:
        return StatsHandler(reduce)

    result = parse_directory(path, handler_factory,
                             show_failed, enable_warnings)

    return {
        "file_count": result[0],
        "failed_count": result[1],
        "chord_count": chord_count,
        "bar_count": bar_count,
    }


@click.command()
@click.argument("source", nargs=-1,
                type=click.Path(dir_okay=True, exists=True, readable=True),
                required=True)
@click.option("--show-failed/--no-show-failed", "show_failed", default=True,
              help="Displays the path of files we failed to tokenize.")
@click.option("--no-output", "no_output", is_flag=True,
              help="Don't write out the .tokens file, simply parse.")
@click.option("--enable-warnings", "enable_warnings", is_flag=True,
              help="Enables parser warnings.")
def tokenize(
    source: list[Path],
    no_output: bool = False,
    show_failed: bool = True,
    enable_warnings: bool = False
):
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
                path,
                write_output=not no_output,
                show_failed=show_failed, enable_warnings=enable_warnings
            )
            count += dir_count
            failed += dir_failed
        else:
            count += 1
            output_path = None if no_output else path.with_suffix(".tokens")
            handler = NormHandler(output_path)
            if not parse_file(path, handler, show_failed, enable_warnings):
                failed += 1
                if output_path is not None:
                    output_path.unlink(missing_ok=True)
    print(f"Tokenized {count} files, {failed} failed.")


@click.command()
@click.argument("source", nargs=-1,
                type=click.Path(dir_okay=True, exists=True, readable=True),
                required=True)
def kern_stats(
    source: list[Path],
    enable_warnings: bool = False
):
    """Parses all .kern files in DATADIR and computes some stats.

    Args:
        datadir (Path): The directory or file to inspect and tokenize.
    """
    for path in [Path(s) for s in source]:
        if path.is_dir():
            stats = stats_directory(path, enable_warnings=enable_warnings)
            print(
                f"file count : {stats['file_count']:,}\n"
                f"bad files  : {stats['failed_count']:,}\n"
                f"bar count  : {stats['bar_count']:,}\n"
                f"chord count: {stats['chord_count']:,}\n"
            )
        else:
            handler = StatsHandler()
            if not parse_file(path, handler, enable_warnings=enable_warnings):
                print(f"Failed to parse file {path}.")
            else:
                print(
                    f"bar count  : {handler.bar_count:,}\n"
                    f"chord count: {handler.chord_count:,}\n"
                )
