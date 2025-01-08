import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


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


CLEF_RE = re.compile(r'^\*clef([a-zA-Z])([0-9])$')


def pitch_from_clef(clef: str) -> Pitch:
    m = CLEF_RE.match(clef)
    assert m is not None, "Invalid clef specification."
    name = m.group(1)
    octave = int(m.group(2))
    return pitch_from_note_and_octave(name, octave)


@dataclass
class Duration:
    duration: int
    dots: int = 0


@dataclass
class Token:
    pass


@dataclass
class Clef(Token):
    pitch: Pitch


@dataclass
class Key(Token):
    is_flats: bool
    count: int


@dataclass
class Meter(Token):
    numerator: int
    denominator: int


@dataclass
class Bar(Token):
    symbol: str


@dataclass
class Null(Token):
    pass


@dataclass
class Rest(Token):
    duration: int


@dataclass
class Note(Token):
    pitch: Pitch
    duration: Optional[Duration]        # May be None e.g. for gracenotes
    flats: int = 0
    sharps: int = 0
    starts_tie: bool = False
    ends_tie: bool = False
    starts_beam: int = 0
    ends_beam: int = 0
    is_gracenote: bool = False
    has_left_beam: bool = False
    has_right_beam: bool = False


@dataclass
class Chord(Token):
    notes: List[Note]
