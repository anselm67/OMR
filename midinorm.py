#!/usr/bin/env python3
import array
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, cast

from midi.input import MidiInput
from midi.typing import (
    Channel,
    Event,
    EventType,
    Format,
    HeaderDataEvent,
    NoteEvent,
    NoteOffEvent,
    NoteOnEvent,
    Notes,
    TempoEvent,
    TimeSignatureEvent,
)

DATADIR = Path("/home/anselm/Downloads/maestro-v3.0.0")

filename = "IMSLP172781-WIMA.cb18-wtc01.mid"
filename = "IMSLP244002-PMLP01969-Etude,_Op._10,_No._12_'Revolutionary'.mid"
filename = "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi"
filename = "2006/MIDI-Unprocessed_22_R1_2006_01-04_ORIG_MID--AUDIO_22_R1_2006_02_Track02_wav.midi"
filename = "2018/MIDI-Unprocessed_Recital1-3_MID--AUDIO_02_R1_2018_wav--2.midi"
filename = "2018/MIDI-Unprocessed_Recital17-19_MID--AUDIO_19_R1_2018_wav--1.midi"


@dataclass
class NotePlay:
    clock: int
    channel: Channel
    note: Notes
    duration: int


class MidiNorm(MidiInput):

    runs: Dict[Notes, int] = {}
    clock: int = 0
    event_count: int = 0
    timeline: List[NotePlay] = list([])
    bars: int       # Bars frequency in clock ticks.
    verbose: bool = False

    def add(self, timestamp: int, e: NoteEvent, duration: int):
        self.timeline.append(NotePlay(timestamp, e.channel, e.note, duration))

    def digest(self):
        timeline = sorted(self.timeline, key=lambda play: play.clock)
        reduced: List[Tuple[int, List[NotePlay]]] = [(0, [timeline[0]])]

        for play in timeline[1:]:
            if play.clock == reduced[-1][0]:
                reduced[-1][1].append(play)
            else:
                reduced.append((play.clock, [play]))
        return reduced

    def handle(self, e: Event):
        if self.verbose:
            print(e)
        self.clock += e.dt
        self.event_count += 1
        if e.event_type == EventType.HeaderData:
            e = cast(HeaderDataEvent, e)
            self.bars = 4 * e.divisions
            print(f"{Format(e.format).name}: {
                  e.number_of_tracks} tracks, {e.divisions}.")
        elif e.event_type == EventType.TimeSignature:
            e = cast(TimeSignatureEvent, e)
            print(f"Time signature: {e.nn}/{e.dd **
                  2} - cc: {e.cc}, dd: {e.dd}")
        elif e.event_type == EventType.Tempo:
            e = cast(TempoEvent, e)
            print(f"Tempo {e.bpm}")
        elif e.event_type == EventType.OpenTrack:
            self.runs = {}
            self.clock = 0
            print("Open track.")
        elif e.event_type == EventType.CloseTrack:
            self.event_count = 0
            print("Close track.")
        if not e.event_type.is_channel():
            return
        if e.event_type == EventType.NoteOn:
            e = cast(NoteOnEvent, e)
            self.runs[e.note] = self.clock
        elif e.event_type == EventType.NoteOff:
            e = cast(NoteOffEvent, e)
            start = self.runs.get(e.note, -1)
            assert start >= 0, f"@ {e.dt} note {e.note} stopped not started."
            self.add(start, e, self.clock - start)


def parse_midi(filename: Path | str):
    clocks_per_bar = 4*384  # wtc -> 4 * divisions = 4 * 120 = 480
    channel_width = 18
    max_channels = 5
    with open(filename, 'rb') as file:
        # Read the entire file content as bytes
        content = file.read()
        parser = MidiNorm(array.array('B', content))
        parser.parse()
        bar_number = 0
        for clock, plays in parser.digest():
            # Displays the bar if needed.
            while clock // clocks_per_bar >= bar_number:
                print(f"== BAR {1 + bar_number} " + "=" * (max_channels * channel_width)
                      )
                bar_number += 1

            # Displays the playing notes (nicely).
            bychan = {play.channel: play for play in plays}

            def format(ch: Channel) -> str:
                if ch.value > max_channels:
                    return ""
                play = bychan.get(ch, None)
                if play is None:
                    return " " * channel_width
                else:
                    text = f"{play.duration:>3}:{play.note}"
                    return f"{text:<{channel_width}}"

            print(f"{clock:>6} {''.join([format(ch) for ch in Channel])}")


parse_midi(DATADIR / filename)
