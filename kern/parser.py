# https://www.humdrum.org/guide/
# Formal syntax: https://www.humdrum.org/guide/ch05/
# Note tokens: https://www.humdrum.org/Humdrum/representations/kern.html#Note%20Tokens
# TODO Handle < and > as in:
# /home/anselm/Downloads/GrandPiano/mozart/piano-sonatas/sonata01-1/min3_up_m-97-100.krn
import re
from abc import ABC, abstractmethod
from enum import Enum
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

    class SpineHolder:

        def __init__(self, spine: T):
            self.spine = spine

    class KernSpine(SpineHolder):
        pass

    class DynamSpine(SpineHolder):
        pass

    class Handler(ABC):

        @abstractmethod
        def open_spine(self,
                       spine_type: Optional[str] = None,
                       parent: Optional[T] = None
                       ) -> T:
            """A new spine is opened.

            This can happen either form the header of the score, in which case a spine_type
            will be provided (e.g. '**kern' or '**dynam') or as a spine is splitted in which
            case the parent spine is provided."""
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

    spines: List[SpineHolder]
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

    OPEN_NOTE_RE = re.compile(r'^([\{\[\(]+)(.*)$')
    NOTE_RE = re.compile(r'^([\d]+)?(\.*)?([a-gA-G]+)(.*)$')

    def parse_note(self, token) -> Note:
        # When an opening (tie, slur, and phrase) starts, keep it for the end.
        additional = ""
        if (m := self.OPEN_NOTE_RE.match(token)):
            additional += m.group(1)
            token = m.group(2)
        if not (m := self.NOTE_RE.match(token)):
            self.error(f"Invalid duration or note in token '{token}'")
        additional += m.group(4)
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
        # TODO Handle 'n' as neither sharp nor flat.
        return Note(
            pitch=Pitch[m.group(3)],
            duration=duration,
            flats=additional.count("-"),
            sharps=additional.count("#"),
            starts_tie="[" in additional,
            ends_tie="]" in additional,
            starts_slur="(" in additional,
            ends_slur=")" in additional,
            starts_phrase="{" in additional,
            ends_phrase="}" in additional,
            starts_beam=additional.count("L"),
            ends_beam=additional.count("J"),
            is_gracenote="q" in additional,
            has_left_beam="k" in additional,
            has_right_beam="K" in additional,
        )

    def position(self, spine_holder: SpineHolder) -> int:
        if (pos := self.spines.index(spine_holder)) < 0:
            self.error(f"Spine {spine_holder} missing.")
        return pos

    def insert_spine(self, at: int, spine_holder: SpineHolder):
        # Copying is required as these are called from within self.spines iterators.
        spines = list(self.spines)
        spines.insert(at, spine_holder)
        self.spines = spines

    def open_spine(self, at: int) -> SpineHolder:
        spine_holder = Parser.SpineHolder(self.handler.open_spine("*+"))
        self.insert_spine(at, spine_holder)
        return spine_holder

    def close_spine(self, spine_holder: SpineHolder):
        self.handler.close_spine(spine_holder.spine)
        # Copying is required as these are called from within self.spines iterators.
        spines = list(self.spines)
        spines.remove(spine_holder)
        self.spines = spines

    def branch_spine(self, source_holder: SpineHolder) -> T:
        branch = self.handler.branch_spine(source_holder.spine)
        self.insert_spine(self.position(source_holder),
                          Parser.SpineHolder(branch))
        return branch

    def merge_spines(self, source_holder: SpineHolder, into_holder: SpineHolder):
        self.handler.merge_spines(source_holder.spine, into_holder.spine)
        self.close_spine(source_holder)

    INDICATOR_RE = re.compile(r'^\*([#:/\w+]*)$')
    SECTION_LABEL_RE = re.compile(r'^\*>.*$')

    def parse_spine_indicator(
        self,
        spine_holder: SpineHolder,
        indicator: str,
        tokens_iterator: Iterator[Tuple[SpineHolder, str]]
    ) -> Token:
        match indicator:
            case '*-':
                self.close_spine(spine_holder)
            case '*+':
                self.open_spine(self.position(spine_holder))
            case '*^':
                # Branch off into a new spine.
                self.branch_spine(spine_holder)
            case '*v':
                holder = spine_holder
                for next_spine, next_token in tokens_iterator:
                    if holder and next_token == "*v":
                        self.merge_spines(next_spine, holder)
                    elif next_token == "*":
                        # No more merges allowed.
                        holder = None
                    else:
                        self.error(f"Invalid spine merge '{next_token}'")
            case '*x':
                self.error("Spine exchange not implemented.")
            case _ if self.SECTION_LABEL_RE.match(indicator):
                # Section labels https://www.humdrum.org/guide/ch20/ skip for now.
                for next_spine, next_token in tokens_iterator:
                    if not next_token.startswith('*>'):
                        self.error(f"Unexpected token {
                                   next_token} within section labels.")
            case _ if (m := self.INDICATOR_RE.match(indicator)):
                # Noop spine indicator.
                if (indicator := m.group(1)):
                    self.handler.rename_spine(spine_holder.spine, indicator)
            case _:
                self.error(f"Unknown spine indicator '{indicator}'.")
        return SpinePath(indicator)

    REST_RE = re.compile(r'^([\d]+)?(\.*)(\.*)r$')
    BAR_RE = re.compile(r'^=+.*$')

    def parse_event(
        self,
        spine_holder: SpineHolder,
        text: str,
        tokens_iterator: Iterator[Tuple[SpineHolder, str]]
    ) -> Token:
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
            return self.parse_spine_indicator(spine_holder, text, tokens_iterator)
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

    def parse_token(
        self,
        spine_holder: SpineHolder,
        text: str,
        iterator: Iterator[Tuple[SpineHolder, str]]
    ) -> Token:
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
            return self.parse_event(spine_holder, text, iterator)

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
                (spine_holder.spine, self.parse_token(
                    spine_holder, text, tokens_iterator))
                for spine_holder, text in tokens_iterator
            ])

    def header(self):
        kerns = self.next(throw_on_end=True).split()    # type: ignore
        for kern in kerns:
            holder = None
            match kern:
                case "**kern":
                    holder = Parser.SpineHolder(self.handler.open_spine(kern))
                case "**dynam":
                    holder = Parser.DynamSpine(self.handler.open_spine(kern))
                case _:
                    self.error(f"Expected a **kern symbol, got '{kern}'.")
            self.spines.append(holder)
