# https://www.humdrum.org/guide/
# Formal syntax: https://www.humdrum.org/guide/ch05/
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Union, cast


class Pitch(Enum):
    C = (3, 1)
    D = (3, 2)
    E = (3, 3)
    F = (3, 4)
    G = (3, 5)
    A = (3, 6)
    B = (3, 7)

    CC = (2, 1)
    DD = (2, 2)
    EE = (2, 3)
    FF = (2, 4)
    GG = (2, 5)
    AA = (2, 6)
    BB = (2, 7)

    CCC = (1, 1)
    DDD = (1, 2)
    EEE = (1, 3)
    FFF = (1, 4)
    GGG = (1, 5)
    AAA = (1, 6)
    BBB = (1, 7)

    CCCC = (0, 1)
    DDDD = (0, 2)
    EEEE = (0, 3)
    FFFF = (0, 4)
    GGGG = (0, 5)
    AAAA = (0, 6)
    BBBB = (0, 7)

    c = (4, 1)
    d = (4, 2)
    e = (4, 3)
    f = (4, 4)
    g = (4, 5)
    a = (4, 6)
    b = (4, 7)

    cc = (5, 1)
    dd = (5, 2)
    ee = (5, 3)
    ff = (5, 4)
    gg = (5, 5)
    aa = (5, 6)
    bb = (5, 7)

    ccc = (6, 1)
    ddd = (6, 2)
    eee = (6, 3)
    fff = (6, 4)
    ggg = (6, 5)
    aaa = (6, 6)
    bbb = (6, 7)

    cccc = (7, 1)
    dddd = (7, 2)
    eeee = (7, 3)
    ffff = (7, 4)
    gggg = (7, 5)
    aaaa = (7, 6)
    bbbb = (7, 7)


def pitch_from_note_and_octave(note: str, octave: int) -> Pitch:
    index = ['c', 'd', 'e', 'f', 'g', 'a', 'b'].index(note.lower())
    assert index >= 0, f"Invalid note name: {note}, expected [A-Za-z]."
    return Pitch((octave, 1+index))


CLEF_RE = re.compile("^\\*clef([a-zA-Z])([0-9])$")


def pitch_from_clef(clef: str) -> Pitch:
    m = CLEF_RE.match(clef)
    assert m is not None, "Invalid clef specification."
    name = m.group(1)
    octave = int(m.group(2))
    return pitch_from_note_and_octave(name, octave)


@dataclass
class Symbol:
    pass


@dataclass
class Clef(Symbol):
    pitch: Pitch


@dataclass
class Key(Symbol):
    is_flats: bool
    count: int


@dataclass
class Meter(Symbol):
    numerator: int
    denominator: int


@dataclass
class Bar(Symbol):
    symbol: str


@dataclass
class Null(Symbol):
    pass


@dataclass
class Rest(Symbol):
    duration: int


@dataclass
class Note(Symbol):
    pitch: Pitch
    duration: int
    flats: int
    sharps: int
    starts_legato: bool
    ends_legato: bool
    starts_beam: bool
    ends_beam: bool
    is_gracenote: bool


@dataclass
class Chord(Symbol):
    notes: List[Note]


class HumdrumParser:

    path: Union[str, Path]
    file: TextIO
    lineno: int = 0
    verbose: bool = False

    spines: Dict[str, List[Symbol]] = {}

    def __init__(self, path: Union[str, Path]):
        if not Path(path).exists():
            raise FileNotFoundError(f"Can't open file {path}")
        self.path = path
        self.file = open(self.path, 'r')

    def error(self, msg: str):
        raise ValueError(f"{self.path}, {self.lineno}: {msg}")

    COMMENT_RE = re.compile("^!!.*$")

    def next(self, throw_on_end: bool = False) -> Optional[str]:
        while True:
            line = self.file.readline().strip()
            self.lineno += 1
            if line is None:
                if throw_on_end:
                    self.error("Unexpected end of file.")
            if line is None or not self.COMMENT_RE.match(line):
                return line

    def end(self):
        symbol = self.next()
        assert not symbol, f"Unexpected symbol '{symbol}' after spine end."

    NOTE_RE = re.compile("^([\\d]+)?(\\.*)?([a-gA-G]+)(.*)$")

    def parse_note(self, token) -> Note:
        if not (m := self.NOTE_RE.match(token)):
            self.error(f"Invalid duration or note in token '{token}'")
        additional = m.group(4)
        # Checks for a valid pitch:
        if m.group(3) not in Pitch.__members__:
            self.error(f"Unknown pitch '{m.group(3)}'.")
        # Computes duration with optional dots
        duration = -1
        if m.group(1):
            duration = int(m.group(1))
            if (dots := m.group(2)):
                duration += len(dots)   # TODO Fix this duration computation
        else:
            assert "q" in additional, "Gracenotes expected without duration."
        return Note(
            pitch=Pitch[m.group(3)],
            duration=duration,
            flats=token.count("#"),
            sharps=token.count("-"),
            starts_legato="[" in token,
            ends_legato="]" in token,
            starts_beam="J" in token,
            ends_beam="L" in token,
            is_gracenote="q" in token,
        )

    REST_RE = re.compile("^([0-9]+)(\\.*)r$")
    BAR_RE = re.compile("^=+.*$")

    def parse_event(self, spine: List[Symbol], symbol: str):
        if self.BAR_RE.match(symbol):
            spine.append(Bar(symbol))
        elif symbol == '.':
            spine.append(Null())
        elif symbol.startswith("!"):
            pass
        elif (m := self.REST_RE.match(symbol)):
            spine.append(Rest(int(m.group(1))))
        else:
            notes = list([])
            for note in symbol.split():
                notes.append(self.parse_note(note))
            if len(notes) == 1:
                spine.append(notes[0])
            else:
                spine.append(Chord(notes))

    CLEF_RE = re.compile("^\\*clef([a-zA-Z])([0-9])$")
    SIGNATURE_RE = re.compile("^\\*k\\[([a-z#-]+)\\]")
    METER_RE = re.compile("^\\*M(\\d)/(\\d)$")
    METRICAL_RE = re.compile("^\\*met\\((C\\|?)\\)$")

    def parse(self):
        self.header()
        while True:
            line = cast(str, self.next(throw_on_end=True))
            for spine, symbol in zip(self.spines.values(), line.split("\t")):
                if (m := self.CLEF_RE.match(symbol)):
                    spine.append(Clef(pitch_from_note_and_octave(
                        m.group(1), int(m.group(2)))))
                elif (m := self.SIGNATURE_RE.match(symbol)):
                    accidental = m.group(1)
                    # TODO Check that accidental is really valid as the RE isn't prefect.
                    spine.append(Key(
                        is_flats=(accidental[-1] == '-'),
                        count=len(accidental) // 2
                    ))
                elif (m := self.METER_RE.match(symbol)):
                    spine.append(Meter(
                        int(m.group(1)),
                        int(m.group(2))
                    ))
                elif (m := self.METRICAL_RE.match(symbol)):
                    if m.group(1) == 'C':
                        spine.append(Meter(4, 4))
                    elif m.group(1) == "C|":
                        spine.append(Meter(2, 2))
                elif (symbol == '*-'):
                    self.end()
                    return
                else:
                    self.parse_event(spine, symbol)

    def header(self):
        kerns = self.next(throw_on_end=True).split()    # type: ignore
        for idx, kern in enumerate(kerns):
            if kern != "**kern":
                self.error("Expeced a **kern symbol.")
            self.spines[f"spine-{idx+1}"] = list([])


DATADIR = Path(
    "/home/anselm/Downloads/GrandPiano/chopin/preludes")


def parse_all():
    parsed, failed = 0, 0
    for root, _, filenames in os.walk(DATADIR):
        for filename in filenames:
            path = Path(root) / filename
            # if filename.name == "min3_down_m-0-4.krn":
            if path.suffix == '.krn':
                try:
                    parsed += 1
                    h = HumdrumParser(path)
                    h.parse()
                except Exception as e:
                    failed += 1
                    print(f"{path.name}: {e}")
    print(f"Parsed {parsed} files, {failed} failed.")


parse_all()
