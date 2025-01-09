#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Type, cast

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
        return f"{duration.duration}:{duration.dots}" \
            if duration.dots > 0 else str(duration.duration)

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
        if note.starts_tie:
            spine.in_tie = True
        if note.starts_beam > 0:
            spine.in_beam = note.starts_beam
        assert spine.in_beam >= 0, f"Reached negative in_beam count {
            spine.in_beam}"
        accidentals = ("#" * note.sharps) or ("-" * note.flats)
        duration_text = ""
        if (duration := note.duration) is None:
            assert note.is_gracenote, "Only gracenotes don't have duration."
            duration_text = "/q"
        else:
            duration_text = f"/{self.format_duration(duration)}"
        tie_text = "_" if spine.in_tie else ""
        beam_count = note.has_left_beam + spine.in_beam + note.has_right_beam
        beam_text = "=" * beam_count
        text = (
            self.format_pitch(note.pitch) + accidentals +
            duration_text +
            tie_text +
            beam_text
        )
        if note.ends_beam > 0:
            spine.in_beam -= note.ends_beam
        if note.ends_tie:
            spine.in_tie = False
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
    spines: List[Spine]

    def __init__(self):
        super(NormHandler, self).__init__()
        self.spines = list([])

    def position(self, spine) -> int:
        return self.spines.index(spine)

    def open_spine(self) -> Spine:
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

    def should_skip(self, tokens: List[Tuple[Spine, Token]]) -> bool:
        return all([isinstance(token, SpinePath) for _, token in tokens])

    def append(self, tokens: List[Tuple[Spine, Token]]):
        if self.should_skip(tokens):
            return
        print('\t'.join([self.formatter.format(spine, token)
              for spine, token in tokens]))
        # for spine, token in tokens:
        #     spine.append(token)


def parse_one(path: Path) -> bool:
    print(path)
    try:
        handler = NormHandler()
        h = Parser.from_file(path, handler)
        h.parse()
        return True
    except Exception as e:
        print(f"{path.name}: {e}")
        return False


def parse_all():
    parsed, failed = 0, 0
    for root, _, filenames in os.walk(DATADIR):
        for filename in filenames:
            path = Path(root) / filename
            # if filename.name == "min3_down_m-0-4.krn":
            if path.suffix == '.krn' and not path.name.startswith("."):
                parsed += 1
                if not parse_one(path):
                    print(f"{path}")
    print(f"Parsed {parsed} files, {failed} failed.")


DATADIR = Path("/home/anselm/Downloads/GrandPiano/")


if __name__ == '__main__':
    parse_all()
    # parse_one(
    #     Path('/home/anselm/Downloads/GrandPiano/chopin/preludes/prelude28-24/original_m-56-61.krn'))
