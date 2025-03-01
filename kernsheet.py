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
import pickle
import random
import re
import time
from dataclasses import asdict, dataclass, field, replace
from functools import reduce
from pathlib import Path
from typing import cast
from urllib.parse import quote

import click
from cv2.typing import MatLike

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

    def pdf_path(self, score: Score) -> Path:
        return self.datadir / score.pdf_path

    def json_path(self, score: Score) -> Path:
        if score.json_path:
            return self.datadir / score.json_path
        else:
            return self.pdf_path(score).with_suffix(".json")

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
        nokern_count, noscore_count, score_count = 0, 0, 0
        score_nopdf, score_nojson, broken_pdf, broken_json, score_nourl, score_tofetch = 0, 0, 0, 0, 0, 0

        for key, entry in self.entries.items():
            if not key in kern_seen:
                nokern_count += 1
                v(f"{key} has no .krn file.")
            if len(entry.scores) == 0:
                noscore_count += 1
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
            f"\twithout scores:    {noscore_count}\n"
            f"\tscores count:      {score_count}\n"
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

    def fetch_imslp(self, key: str, how_many: int = 2) -> bool:
        entry = self.entries.get(key)
        if entry is None or not entry.imslp_url:
            return False
        # Verifies we're ok killing all scores of this entry.
        for score in entry.scores:
            if score.pdf_path or score.json_path:
                logging.info(f"{key} has valid scores.")
                return False
        # Rebuilds a set of scores by picking some from IMSLP.
        imslp = IMSLP()
        links = imslp.find_pdf_links(entry.imslp_url)
        if len(links) == 0:
            return False
        elif len(links) > how_many:
            logging.info(f"Picking {how_many} links from {len(links)}")
            links = random.sample(links, how_many)
        scores = []
        for idx, link in enumerate(links):
            path = self.datadir / key
            path = path.with_stem(path.stem + f"-{idx}").with_suffix(".pdf")
            url = imslp.download_link(link)
            if url:
                imslp.save_pdf(url, path)
                scores.append(Score(
                    pdf_path=self.relative(path),
                    pdf_url=url,
                    json_path=""
                ))
                time.sleep(10 + random.randint(0, 20))
        # If we made it here, save the work.
        entry.scores = scores
        self.save_catalog()
        return True

    def edit(
        self,
        key: str, entry: Entry,
        all: bool = True,
        fast_mode: bool = False,
        no_cache: bool = False, do_plot: bool = False
    ) -> bool:
        from staff_editor import StaffEditor
        from staffer import Staffer

        # No scores? warns and keeps going.
        if len(entry.scores) == 0:
            logging.warning(f"{key} has no scores.")
            return True

        # Edits or reviews all scores of that entry.
        for score in entry.scores:
            if not score.pdf_path:
                logging.warning(
                    f"{key} - {score.pdf_url}:\n"
                    f"\tScore has no pdf (skipped)."
                )
                continue
            pdf_path = self.pdf_path(score)
            if not pdf_path.exists():
                logging.warning(
                    f"{key} - {score.pdf_url}:\n"
                    f"PDF {pdf_path} not found (skipped)."
                )
                continue
            staffer = Staffer(
                self, key, score, Staffer.Config(
                    no_cache=no_cache, do_plot=do_plot))
            do_continue = False
            if all or not staffer.is_validated():
                editor = StaffEditor(staffer)
                do_continue = editor.edit(fast_mode)
                # Updates the entry with a json path if one was created.
                if self.json_path(score).exists() and not score.json_path:
                    score.json_path = self.relative(self.json_path(score))
                    self.save_catalog()
                if not do_continue:
                    return False
        return True

    def extract_staves(self, dst_dir: Path) -> int:
        from staffer import Staffer
        count = 0
        for key, entry in self.entries.items():
            for score in entry.scores:
                if not score.pdf_path or not score.json_path:
                    continue
                staffer = Staffer(
                    self, key, score, Staffer.Config()
                )
                if staffer.is_validated():
                    for idx, sample in enumerate(staffer.extract_staves()):
                        count += 1
                        dst_file = (
                            dst_dir / f"{key}_{idx:03d}").with_suffix(".pkl")
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        print(f"{dst_file}: {len(sample[1])}")
                        with open(dst_file, "wb+") as fp:
                            pickle.dump(sample, fp)
        return count

    def delete_score(self, key: str, score: Score):
        entry = self.entries.get(key, None)
        if entry:
            entry.scores = [s for s in entry.scores if s != score]
            logging.info(f"{key}: removed score {score.pdf_url}")
            self.save_catalog()
        return False

    def stats(self):
        from staffer import Staffer

        score_count, staff_count, bar_count = 0, 0, 0
        for key, entry in self.entries.items():
            for score in entry.scores:
                score_count += 1
                if not score.pdf_path or not score.json_path:
                    continue
                staffer = Staffer(
                    self, key, score, Staffer.Config()
                )
                if staffer.is_validated():
                    s, b = staffer.counts()
                    staff_count, bar_count = s+staff_count, b+bar_count

        print(f"{score_count} scores: {
              staff_count:,} staves, {bar_count:,} bars.")


@click.command()
@click.pass_context
def fix_imslp(ctx):
    """Associates entries with an IMSLP page.

    Runs a Google query and attempts to match the given entry to 
    an IMSLP page. PDFs for entries that have an IMSLP page can be
    downloaded using the fetch-imslp command.

    Args:
        ctx (_type_): _description_
    """
    kern_sheet = cast(KernSheet, ctx.obj)
    kern_sheet.fix_imslp()


@click.command()
@click.argument("prefix", type=str, required=False, default="")
@click.option("--all", "all", is_flag=True, default=False,
              help="Edit all files, even validated ones.")
@click.option("--no-cache", "no_cache", is_flag=True, default=False,
              help="Don't use cached versions of the pdf & staff.")
@click.option("--do-plot", "do_plot", is_flag=True, default=False,
              help="Don't use cached versions of the pdf & staff.")
@click.option("--fast-mode", "fast_mode", is_flag=True, default=False,
              help="Turns fast mode on automatically for al scores.")
@click.pass_context
def edit(ctx, prefix: str, all: bool, no_cache: bool, do_plot: bool, fast_mode: bool):
    """Edit all entries of the set of entries matching PREFIX when given.

    Args:
        ctx (_type_): Click context for the dataset.
        prefix (str): Optional orefix of enrties to edit, all if none.
        all (bool): Edit all scores, even the ones that have been validated.
        no_cache (bool): Override the json cache, if any.
        do_plot (bool): Plots sheet music histgrams to tune the staffer.
        fast_mode (bool): Enables fast mode for all edits. In fast more, 
           moving to next page (or entry) will autmatically validate and save
           the current page.
    """
    kern_sheet = cast(KernSheet, ctx.obj)
    for key, entry in kern_sheet.entries.items():
        if prefix and not key.startswith(prefix):
            continue
        if not kern_sheet.edit(
            key, entry, all=all, no_cache=no_cache,
            do_plot=do_plot, fast_mode=fast_mode
        ):
            break


@click.command()
@click.pass_context
def stats(ctx):
    """Provides statistics on the dataset, e.g. staves and bar counts.

    Args:
        ctx (_type_): _description_
    """
    kern_sheet = cast(KernSheet, ctx.obj)
    kern_sheet.stats()


@click.command()
@click.argument("dst_dir", required=True,
                type=click.Path(file_okay=False, dir_okay=True, exists=False))
@click.pass_context
def extract_staves(ctx, dst_dir: Path):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    kern_sheet = cast(KernSheet, ctx.obj)
    count = kern_sheet.extract_staves(dst_dir)
    print(f"count: {count} samples")


@click.command()
@click.argument("prefix", type=str, required=False, default="")
@click.option("--how-many", type=int, default=2, required=False)
@click.pass_context
def fetch_imslp(ctx, prefix: str, how_many: int):
    """Fetches IMSLP pdf for a set of catalog entries.

    This will only lookup for entries that currently don't have 
    any scores. Entries with at least one score will be skipped.
    Args:
        ctx (_type_): Click context for the dataset.
        prefix (str): Count of pdf files to fetch per entries.
    """
    kern_sheet = cast(KernSheet, ctx.obj)
    for key, _ in kern_sheet.entries.items():
        if prefix and not key.startswith(prefix):
            continue
        if kern_sheet.fetch_imslp(key, how_many=how_many):
            time.sleep(30.0 + random.randint(0, 10))


@click.command()
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Print all issues found in the catalog.")
@click.pass_context
def check(ctx, verbose: bool):
    """Runs various sanity checks on the dataset.

    Args:
        ctx (_type_): Click ontext for the dataset.
        verbose (bool): Prints problems as they're found.
    """
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
cli.add_command(stats)
cli.add_command(check)
cli.add_command(fetch_imslp)

cli.add_command(extract_staves)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cli()
