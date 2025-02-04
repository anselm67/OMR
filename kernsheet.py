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
from typing import Optional, cast
from urllib.parse import quote

import click

from imslp import IMSLP
from make_kernsheet import make_kern_sheet, merge_asap
from utils import from_json, path_substract


@dataclass
class Score:
    # Relative path to the pdf file.
    pdf_path: str
    pdf_url: str
    json_path: str


@dataclass
class Entry:
    # The source dataset for this entry, "kern-score/*.zip" or "asap"
    source: str

    # The IMSLP url for this score, when available, and the Google query
    # we used to find it. See KernSheet.fix_imslp for details.
    imslp_query: str
    imslp_url: str

    scores: list[Score] = field(default_factory=list)


@dataclass
class Catalog:
    version: int = 2

    entries: dict[str, Entry] = field(default_factory=dict)


class KernSheet:
    CATALOG_NAME = "catalog.json"

    datadir: Path
    catalog: Catalog

    @property
    def entries(self) -> dict[str, Entry]:
        return self.catalog.entries

    def __init__(self, datadir: Path):
        super().__init__()
        self.datadir = Path(datadir)
        self.load_catalog()

    def kern_path(self, key: str) -> Path:
        return (self.datadir / key).with_suffix(".krn")

    def relative(self, path: Path) -> str:
        return str(path_substract(self.datadir, path))

    def load_catalog(self):
        path = self.datadir / self.CATALOG_NAME
        if path.exists():
            with open(path, "r") as fp:
                obj = json.load(fp)
            self.catalog = cast(Catalog, from_json(Catalog, obj))

    def save_catalog(self):
        path = self.datadir / self.CATALOG_NAME
        with open(path, "w+") as fp:
            json.dump(asdict(self.catalog), fp, indent=4)

    KERN_SCORE_URL = "https://kern.humdrum.org/cgi-bin/ksdata?location=users/craig/classical/"

    def kernscore_url(self, kern_file: Path) -> str:
        return (
            self.KERN_SCORE_URL + str(kern_file.parent) +
            f"&file={quote(kern_file.name)}&format=pdf"
        )

    def check(self, verbose: bool = False):
        """Checks the catalog against the file structure.

        Reports on .krn files without an entry, and on entries without a
        .krn file. 
        """
        def v(msg: str):
            if verbose:
                print(msg)
        file_count, noent_count = 0, 0
        kern_seen = set()
        # Checks the file system against the catalog.
        for root, _, filenames in os.walk(self.datadir):
            for filename in filenames:
                file = Path(root) / filename
                if file.suffix == ".krn":
                    file_count += 1
                    key = str(path_substract(
                        self.datadir, file).with_suffix(""))
                    kern_seen.add(key)
                    if not key in self.entries:
                        noent_count += 1
        # Checks the catalog against the file system, and the scores
        nokern_count, score_count = 0, 0
        score_nopdf, score_nojson, broken_pdf, broken_json, score_nourl, score_tofetch = 0, 0, 0, 0, 0, 0

        for key, entry in self.entries.items():
            if not key in kern_seen:
                nokern_count += 1
                v(f"{key} has no .krn file.")
            if len(entry.scores) == 0:
                v(f"{key} has no scores.")
                continue
            # Checks each score.
            score_count += len(entry.scores)
            for s in entry.scores:
                if s.json_path:
                    if not (self.datadir / s.json_path).exists():
                        broken_json += 1
                        v(f"{s.json_path} not found.")
                else:
                    score_nojson += 1
                    v(f"{key} score has no json.")
                if s.pdf_path:
                    if not (self.datadir / s.pdf_path).exists():
                        broken_pdf += 1
                        v(f"{s.pdf_path} not found.")
                else:
                    score_nopdf += 1
                    v(f"{key} score has no pdf.")
                if not s.pdf_url:
                    score_nourl += 1
                elif not s.pdf_path:
                    score_tofetch += 1
        print(
            f"{file_count} kern files:\n"
            f"\twithout entries: {noent_count}\n"
            f"{len(self.entries)} entries:\n"
            f"\twithout .krn file: {nokern_count}\n"
            f"\tscores count: {score_count}\n"
            f"\t\tscores without source url: {score_nourl}\n"
            f"\t\tscores with url, no pdf:   {score_tofetch}\n"
            f"\t\tscores without pdf:        {score_nopdf}\n"
            f"\t\tscores without json:       {score_nojson}\n"
            f"\t\tscores with broken pdf:    {broken_pdf}\n"
            f"\t\tscores with broken json:   {broken_json}\n"
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

    def first_score(self, key: str, entry: Entry) -> tuple[bool, Optional[Score]]:
        score = next((s for s in entry.scores if s.pdf_path), None)
        if score is None:
            return False, None

        needs_save = False
        pdf_path = Path(self.datadir) / score.pdf_path
        if not pdf_path.exists():
            score.pdf_path = ""
            needs_save = True
        if score.json_path:
            json_path = Path(self.datadir) / score.json_path
            if not json_path.exists():
                score.json_path = ""
                needs_save = True
        if score.pdf_path and not score.json_path:
            json_path = self.kern_path(key).with_suffix(".json")
            score.json_path = self.relative(json_path)
            needs_save = True

        if not score.pdf_path or not score.json_path:
            return needs_save, None
        else:
            return needs_save, score

    def edit(
        self,
        key: str | Path,
        all: bool = True,
        fast_mode: bool = False,
        no_cache: bool = False, do_plot: bool = False
    ) -> bool:
        do_continue = True
        from staff_editor import StaffEditor
        from staffer import Staffer

        key = Path(key)
        if not Path(key).exists():
            raise FileNotFoundError(f"{Path(key)} - no such file.")
        key = self.relative(key.with_suffix(""))
        entry = self.entries.get(key, None)
        if entry is None:
            raise FileNotFoundError(f"{key} - no such key.")
        dirty, score = self.first_score(key, entry)
        if score is None:
            # No scores attached to that entry yet.
            return do_continue
        pdf_path = self.datadir / score.pdf_path
        json_path = self.datadir / score.json_path
        staffer = Staffer(
            self, key, pdf_path, json_path, Staffer.Config(
                pdf_cache=True, no_cache=no_cache, do_plot=do_plot)
        )
        if all or not staffer.is_validated():
            print(f"Editing {key} - {pdf_path.name}")
            editor = StaffEditor(staffer)
            do_continue = editor.edit(fast_mode)
        if dirty:
            self.save_catalog()
        return do_continue

    def ls(self):
        for key, entry in self.entries.items():
            has_pdf = any((s.pdf_path for s in entry.scores))
            has_jon = any((s.json_path for s in entry.scores))
            if has_jon and not has_pdf:
                print(key)


@click.command()
@click.pass_context
def fix_imslp(ctx):
    kern_sheet = cast(KernSheet, ctx.obj)
    kern_sheet.fix_imslp()


def edit_directory(
    kern_sheet: KernSheet, dir: Path,
    all: bool, no_cache: bool, do_plot: bool, fast_mode: bool
) -> bool:
    for path in dir.iterdir():
        if path.suffix == ".krn":
            return kern_sheet.edit(
                path, all=all, no_cache=no_cache,
                do_plot=do_plot, fast_mode=fast_mode
            )
        elif path.is_dir() and not edit_directory(
            kern_sheet, path, all, no_cache, do_plot, fast_mode
        ):
            return False
    return True


@click.command()
@click.argument("kern_path", nargs=-1,
                type=click.Path(dir_okay=True, exists=True, readable=True),
                required=False)
@click.option("--all", "all", is_flag=True, default=False,
              help="Edit all files, even validated ones.")
@click.option("--no-cache", "no_cache", is_flag=True, default=False,
              help="Don't use cached versions of the pdf & staff.")
@click.option("--do-plot", "do_plot", is_flag=True, default=False,
              help="Don't use cached versions of the pdf & staff.")
@click.option("--fast-mode", "fast_mode", is_flag=True, default=False,
              help="Turns fast mode on automatically for al scores.")
@click.pass_context
def edit(ctx, kern_path: list[Path], all: bool, no_cache: bool, do_plot: bool, fast_mode: bool):
    kern_sheet = cast(KernSheet, ctx.obj)
    if len(kern_path) == 0:
        kern_path = [kern_sheet.datadir]
    for path in kern_path:
        path = Path(path)
        if path.is_dir():
            if not edit_directory(kern_sheet, path, all, no_cache, do_plot, fast_mode):
                break
        elif not kern_sheet.edit(
            path, all=all, no_cache=no_cache,
            do_plot=do_plot, fast_mode=fast_mode
        ):
            break


@click.command()
@click.pass_context
def update(ctx):
    # By load & save we ensure all optional fields of Enrty are now available.
    kern_sheet = cast(KernSheet, ctx.obj)
    kern_sheet.save_catalog()


@click.command()
@click.pass_context
def ls(ctx):
    kern_sheet = cast(KernSheet, ctx.obj)
    kern_sheet.ls()


@click.command()
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Print all issues found in the catalog.")
@click.pass_context
def check(ctx, verbose: bool):
    kern_sheet = cast(KernSheet, ctx.obj)
    kern_sheet.check(verbose)


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
cli.add_command(update)
cli.add_command(ls)
cli.add_command(check)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cli()
