#!/usr/bin/env python3
# To run this script, you'll need the following sources:
# KERN_SCORES_URL List of url to download zipped kern files from KernScore dataset
# A clone of https://github.com/fosfrancesco/asap-dataset
# Once you have these available, run the following commands:
# ./kernsheet.py make-kern-sheet TARGET_DIRECTORY
# ./kernsheet.py merge-asap ASAP_DIRECTORY TARGET_DIRECTORY
# At this point TARGET_DIRECTORY will contain the kernsheet dataset, with the
# following stats:
# file count : 689
# bad files  : 18
# bar count  : 72,808
# chord count: 565,965

import json
import logging
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field, replace
from functools import reduce
from pathlib import Path
from typing import Dict, List, Optional, cast
from urllib.parse import quote

import click

from imslp import IMSLP
from make_kernsheet import make_kern_sheet, merge_asap
from staffer import Staffer
from utils import path_substract


@dataclass(frozen=True)
class Entry:
    # The source dataset for this entry, "kern-score/*.zip" or "asap"
    source: str

    # The IMSLP url for this score, when available, and the Google query
    # we used to find it. See KernSheet.fix_imslp for details.
    imslp_query: str
    imslp_url: str

    pdf_urls: List[str]

    @staticmethod
    def from_dict(data):
        return Entry(**data)


@dataclass(frozen=True)
class Catalog:
    version: int = 1

    entries: Dict[str, Entry] = field(default_factory=dict)


class KernSheet:
    CATALOG_NAME = "catalog.json"

    datadir: Path
    version: int = 1
    entries: Dict[str, Entry]

    def __init__(self, datadir: Path):
        super().__init__()
        self.datadir = Path(datadir)
        self.load_catalog()

    def kern_path(self, key: str) -> Path:
        return (self.datadir / key).with_suffix(".krn")

    def pdf_path(self, key: str) -> Path:
        return (self.datadir / key).with_suffix(".pdf")

    def json_path(self, key: str) -> Path:
        return (self.datadir / key).with_suffix(".json")

    def load_catalog(self):
        path = self.datadir / self.CATALOG_NAME
        self.version = 1
        self.entries = {}
        if path.exists():
            with open(path, "r") as fp:
                obj = json.load(fp)
            self.version = obj["version"]
            self.entries = {kern_file: Entry.from_dict(
                entry_dict) for kern_file, entry_dict in obj["entries"].items()}

    def save_catalog(self):
        path = self.datadir / self.CATALOG_NAME
        entries = {k: asdict(e) for k, e in self.entries.items()}
        with open(path, "w+") as fp:
            json.dump({
                "version": 1,
                "entries": dict(sorted(entries.items()))
            }, fp, indent=4)

    KERN_SCORE_URL = "https://kern.humdrum.org/cgi-bin/ksdata?location=users/craig/classical/"

    def kernscore_url(self, kern_file: Path) -> str:
        return (
            self.KERN_SCORE_URL + str(kern_file.parent) +
            f"&file={quote(kern_file.name)}&format=pdf"
        )

    def stats(self):
        """Checks for orphaned .krn files with no entries in the catalog.
        """
        total_count, miss_count, pdf_count, json_count = 0, 0, 0, 0
        for root, _, filenames in os.walk(self.datadir):
            for filename in filenames:
                file = Path(root) / filename
                if file.suffix == ".krn":
                    total_count += 1
                    key = str(path_substract(
                        self.datadir, file).with_suffix(""))
                    if not key in self.entries:
                        miss_count += 1
                    if self.pdf_path(key).exists():
                        pdf_count += 1
                    if self.json_path(key).exists():
                        json_count += 1
        print(
            f"{total_count} kern files:\n"
            f"\twithout entries: {miss_count}\n"
            f"\twith pdf       : {pdf_count}\n"
            f"\twith json      : {json_count}\n"
        )

    KERN_KEYWORDS_RE = re.compile(r'^!!!(COM|OPR|OTL|OPS):\s*(.*)$')

    def google_keywords(self, key: str) -> str:
        kern_path = self.kern_path(key)
        keywords = list()

        def add(keyword: str):
            keyword = re.sub(r'[^[a-zA-Z0-9]+', " ", keyword)
            for word in keyword.lower().split():
                keywords.append(word.strip())

        # Checks inside the file for COM and OTL:
        with open(kern_path, "r") as fp:
            for line in fp:
                line = line.strip()
                if not line.startswith("!!!"):
                    break
                elif (m := self.KERN_KEYWORDS_RE.match(line)):
                    add(m.group(2).lower())
        # Adds in the path components:
        if len(keywords) == 0:
            add(key)
        return " ".join(
            reduce(lambda l, x: l.append(x)     # type: ignore
                   or l if x not in l else l, keywords, list())
        ) + " site:imslp.org"

    def fix_imslp(self):
        imslp = IMSLP()
        for key, entry in self.entries.items():
            if entry.imslp_query or entry.imslp_url:
                continue
            logging.info(f"[fix_imslp]: {key}")

            imslp_query = self.google_keywords(key)
            try:
                imslp_url = imslp.find_imslp(imslp_query)
                if imslp_url is None:
                    print("No Google match found for query.")
            except Exception as e:
                logging.exception(f"[fix_imslp]: {key} failed ({e})")
                raise
            # We always store at least the Google query we used.
            self.entries[key] = replace(
                entry,
                imslp_query=imslp_query,
                imslp_url=imslp_url or "")
            self.save_catalog()
            time.sleep(20 + random.randint(10, 20))

    def edit(
        self, key: Optional[str] = None, no_cache: bool = False, do_plot: bool = False
    ):
        if key is None:
            # Loops through all samples in need of verification, skips non pdf.
            for key, entry in self.entries.items():
                if not self.pdf_path(key).exists():
                    continue
                staffer = Staffer(
                    self.datadir, key, do_plot=do_plot, no_cache=no_cache
                )
                if not staffer.is_validated():
                    print(
                        f"Editing {key}\n"
                        f"\timslp_url: {entry.imslp_url}\n"
                        f"\tpdf_urls: {entry.pdf_urls}\n"
                    )
                    if not staffer.edit():
                        break
        else:
            # Edits the given entry.
            staffer = Staffer(
                self.datadir, key, do_plot=do_plot, no_cache=no_cache
            )
            staffer.edit()


@click.command()
@click.pass_context
def fix_imslp(ctx):
    kern_sheet = cast(KernSheet, ctx.obj)
    kern_sheet.fix_imslp()


@click.command()
@click.argument("kern_path", type=str, required=False, default=None)
@click.option("--no-cache", "no_cache", is_flag=True, default=False,
              help="Don't use cached versions of the pdf & staff.")
@click.option("--do-plot", "do_plot", is_flag=True, default=False,
              help="Don't use cached versions of the pdf & staff.")
@click.pass_context
def edit(ctx, kern_path: Optional[str], no_cache: bool, do_plot: bool):
    kern_sheet = cast(KernSheet, ctx.obj)
    kern_sheet.edit(kern_path, no_cache=no_cache, do_plot=do_plot)


@click.command()
@click.pass_context
def stats(ctx):
    kern_sheet = cast(KernSheet, ctx.obj)
    kern_sheet.stats()


@click.command()
@click.pass_context
def update(ctx):
    # By load & save we ensure all optional fields of Enrty are now available.
    kern_sheet = cast(KernSheet, ctx.obj)
    kern_sheet.save_catalog()


@click.group
@click.option("--dataset", "-d", required=False,
              type=click.Path(file_okay=False, dir_okay=True, exists=True),
              default="/home/anselm/datasets/kern-sheet/")
@click.pass_context
def cli(ctx, dataset: Path):
    ctx.obj = KernSheet(dataset)


# From make_kernsheet.py
cli.add_command(make_kern_sheet)
cli.add_command(merge_asap)


cli.add_command(fix_imslp)
cli.add_command(edit)
cli.add_command(stats)
cli.add_command(update)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cli()
