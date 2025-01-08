#!/usr/bin/env python3

import os
from pathlib import Path
from typing import List, Optional, Tuple

from kern.parser import Handler, Parser
from kern.typing import Token


class Spine:
    name: str
    tokens: List[Token] = list([])
    parent: Optional[Tuple['Spine', int]]

    def __init__(self, parent: Optional['Spine'] = None):
        if parent is not None:
            self.parent = (parent, len(parent.tokens))

    def append(self, token: Token):
        self.tokens.append(token)

    def rename(self, name: str):
        self.name = name


class NormHandler(Handler[Spine]):
    spines: List[Spine]

    def __init_(self):
        self.spines = list([])

    def open_spine(self) -> Spine:
        return Spine()

    def close_spine(self, spine: Spine):
        pass

    def branch_spine(self, source: Spine) -> Spine:
        return Spine(source)

    def merge_spines(self, source: Spine, into: Spine):
        pass

    def append(self, spine: Spine, token: Token):
        spine.append(token)


def parse_one(path: Path) -> bool:
    try:
        h = Parser.from_file(path, NormHandler())
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
                    failed += 1
    print(f"Parsed {parsed} files, {failed} failed.")


DATADIR = Path("/home/anselm/Downloads/GrandPiano/")


if __name__ == '__main__':
    parse_all()
    # parse_one(
    #     Path('/home/anselm/Downloads/GrandPiano/beethoven/piano-sonatas/sonata26-2/maj2_down_m-1-6.krn'))
