# https://www.humdrum.org/guide/
# Formal syntax: https://www.humdrum.org/guide/ch05/
# Note tokens: https://www.humdrum.org/Humdrum/representations/kern.html#Note%20Tokens

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from kern.typing import (
    Bar,
    Chord,
    Clef,
    Comment,
    Continue,
    Duration,
    Key,
    Meter,
    Note,
    Pitch,
    Rest,
    SpinePath,
    Token,
    pitch_from_note_and_octave,
)
from utils import iterable_from_file

T = TypeVar("T")


class Parser(Generic[T]):

    class Handler(ABC):

        @abstractmethod
        def open_spine(self) -> T:
            pass

        @abstractmethod
        def close_spine(self, spine: T):
            pass

        @abstractmethod
        def branch_spine(self, source: T) -> T:
            pass

        @abstractmethod
        def merge_spines(self, source: T, into: T):
            pass

        @abstractmethod
        def rename_spine(self, spine: T, name: str):
            pass

        @abstractmethod
        def append(self, tokens: List[Tuple[T, Token]]):
            pass

    path: Union[str, Path]
    records: Iterator[str]
    lineno: int = 0
    verbose: bool = False

    spines: List[T]
    handler: Handler

    def __init__(self, path: Union[str, Path], records: Iterable[str], handler: Handler):
        self.path = path
        self.records = iter(records)
        self.handler = handler
        self.spines = list([])

    @staticmethod
    def from_file(path: Union[str, Path], handler: Handler) -> 'Parser':
        return Parser(path, iterable_from_file(path), handler)

    @staticmethod
    def from_text(text: str, handler: Handler) -> 'Parser':
        return Parser("text", iter(text.split("\n")), handler)

    @staticmethod
    def from_iterator(iterator: Iterable[str], handler: Handler) -> 'Parser':
        return Parser("iterator", iterator, handler)

    def error(self, msg: str):
        raise ValueError(f"{self.path}, {self.lineno}: {msg}")

    COMMENT_RE = re.compile(r'^!!.*$')

    def next(self, throw_on_end: bool = False) -> Optional[str]:
        while True:
            line = next(self.records, None)
            self.lineno += 1
            if not line:
                if throw_on_end:
                    self.error("Unexpected end of file.")
                return None
            line = line.strip()
            if not self.COMMENT_RE.match(line):
                return line

    NOTE_RE = re.compile(r'^([\d]+)?(\.*)?([a-gA-G]+)(.*)$')

    def parse_note(self, token) -> Note:
        if not (m := self.NOTE_RE.match(token)):
            self.error(f"Invalid duration or note in token '{token}'")
        additional = m.group(4)
        # Checks for a valid pitch:
        if m.group(3) not in Pitch.__members__:
            self.error(f"Unknown pitch '{m.group(3)}'.")
        # Computes duration with optional dots
        duration = None
        if m.group(1):
            duration = Duration(int(m.group(1)), len(m.group(2)))
        else:
            assert "q" in additional, "Gracenotes expected without duration."

        # https://www.humdrum.org/Humdrum/representations/kern.html
        # 3.5 Editorial signifiers: XxYy not handled.
        for x in r'TtMmWwsS$R\'/\\Q"`~^':
            if x in token:
                print(token)
        # TODO Handle 'n' as neither shapr nor flat.
        return Note(
            pitch=Pitch[m.group(3)],
            duration=duration,
            flats=token.count("-"),
            sharps=token.count("#"),
            starts_tie="[" in token,
            ends_tie="]" in token,
            starts_beam=token.count("L"),
            ends_beam=token.count("J"),
            is_gracenote="q" in token,
            has_left_beam="k" in token,
            has_right_beam="K" in token,
        )

    def position(self, spine: T) -> int:
        if (pos := self.spines.index(spine)) < 0:
            self.error(f"Spine {spine} missing.")
        return pos

    def insert_spine(self, at: int, spine: T):
        # Copying is required as these are called from within self.spines iterators.
        spines = list(self.spines)
        spines.insert(at, spine)
        self.spines = spines

    def open_spine(self, at: int) -> T:
        spine = self.handler.open_spine()
        self.insert_spine(at, spine)
        return spine

    def close_spine(self, spine: T):
        self.handler.close_spine(spine)
        # Copying is required as these are called from within self.spines iterators.
        spines = list(self.spines)
        spines.remove(spine)
        self.spines = spines

    def branch_spine(self, source: T) -> T:
        branch = self.handler.branch_spine(source)
        self.insert_spine(self.position(source), branch)
        return branch

    def merge_spines(self, source: T, into: T):
        self.handler.merge_spines(source, into)
        self.close_spine(source)

    INDICATOR_RE = re.compile(r'^\*([\w+]*)$')

    def parse_spine_indicator(
        self, spine, indicator: str,
        tokens_iterator: Iterator[Tuple[T, str]]
    ) -> Token:
        match indicator:
            case '*-':
                self.close_spine(spine)
            case '*+':
                self.open_spine(self.position(spine))
            case '*^':
                # Branch off into a new spine.
                self.branch_spine(spine)
            case '*v':
                for next_spine, next_token in tokens_iterator:
                    if spine and next_token == "*v":
                        self.merge_spines(next_spine, spine)
                    elif next_token == "*":
                        # No more merges allowed.
                        spine = None
                    else:
                        self.error(f"Invalid spine merge '{next_token}'")
            case '*x':
                self.error("Spine exchange not implemented.")
            case _ if (m := self.INDICATOR_RE.match(indicator)):
                # Noop spine indicator.
                if (indicator := m.group(1)):
                    self.handler.rename_spine(spine, indicator)
            case _:
                self.error(f"Unknown spine indicator '{indicator}'.")
        return SpinePath(indicator)

    REST_RE = re.compile(r'^([\d]+)?(\.*)(\.*)r$')
    BAR_RE = re.compile(r'^=+.*$')

    def parse_event(self, spine: T, text: str,
                    tokens_iterator: Iterator[Tuple[T, str]]) -> Token:
        if self.BAR_RE.match(text):
            return Bar(text)
        elif text == '.':
            return Continue()
        elif text.startswith("!"):
            # A comment, we don't .
            return Comment(text)
        elif (m := self.REST_RE.match(text)):
            return Rest(Duration(int(m.group(1)), len(m.group(2))))
        elif text.startswith("*"):
            return self.parse_spine_indicator(spine, text, tokens_iterator)
        else:
            notes = list([])
            for note in text.split():
                notes.append(self.parse_note(note))
            if len(notes) == 1:
                return notes[0]
            else:
                return Chord(notes)

    CLEF_RE = re.compile(r'^\*clef([a-zA-Z])([0-9])$')
    SIGNATURE_RE = re.compile(r'\*k\[(([a-z][#-])*)\]')
    METER_RE = re.compile(r'^\*M(\d+)/(\d+)$')
    METRICAL_RE = re.compile(r'^\*met\(([cC]\|?)\)$')

    def parse_token(self, spine: T, text: str, iterator: Iterator[Tuple[T, str]]) -> Token:
        if (m := self.CLEF_RE.match(text)):
            return Clef(pitch_from_note_and_octave(m.group(1), int(m.group(2))))
        elif (m := self.SIGNATURE_RE.match(text)):
            # Empty key signature is allowed.
            if (accidental := m.group(1)):
                # TODO Check that accidental is really valid as the RE isn't prefect.
                return Key(
                    is_flats=(accidental[-1] == '-'),
                    count=len(accidental) // 2
                )
            else:
                # Empty key signatures are ok.
                return Key(False, 0)
        elif (m := self.METER_RE.match(text)):
            return Meter(
                int(m.group(1)),
                int(m.group(2))
            )
        elif (m := self.METRICAL_RE.match(text)):
            metric = m.group(1).upper()
            if metric == 'C':
                return Meter(4, 4)
            elif metric == "C|":
                return Meter(2, 2)
            else:
                self.error(f"Invalid metric '{metric}'.")
        else:
            return self.parse_event(spine, text, iterator)

    def parse(self):
        self.header()
        while True:
            line = cast(str, self.next())
            if not line:
                return
            tokens = line.split("\t")
            if len(tokens) != len(self.spines):
                self.error(f"Got {len(tokens)} for {len(self.spines)} spines.")
            tokens_iterator = zip(self.spines, tokens)
            self.handler.append([
                (spine, self.parse_token(spine, text, tokens_iterator))
                for spine, text in tokens_iterator
            ])

    def header(self):
        kerns = self.next(throw_on_end=True).split()    # type: ignore
        for kern in kerns:
            if kern != "**kern":
                self.error(f"Expected a **kern symbol, got '{kern}'.")
            self.spines.append(self.handler.open_spine())
