#!/usr/bin/env python3

import os
from pathlib import Path
from typing import List, Optional, Tuple

import click
import cv2
import numpy as np
from cv2.typing import MatLike

from staffer import Staffer

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

SIZE = (1800, 1200)


points = []


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")
        points.append((x, y))
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            print(f"dx: {x2-x1}, dy: {y2 -
                  y1}, d: {((x2-x1)**2+(y2-y1)**2) ** 0.5}")
            points.clear()


def show(*images: MatLike,
         max_height: int = 800,
         window: Optional[str] = None):
    interrupted = False
    for image in images:
        height, width = image.shape[:2]
        if max_height > 0 and height > max_height:
            new_width = int(max_height * width / height)
            image = cv2.resize(image, (new_width, max_height))
        cv2.imshow(window or "window", image)
        cv2.setMouseCallback(window or "window", click_event)
        if window is None and (interrupted := (cv2.waitKey() == ord('q'))):
            break
    if window is None:
        cv2.destroyAllWindows()
    return interrupted


def compare(a: List[MatLike], b: List[MatLike], crop=None):
    for ia, ib in zip(a, b):
        show(ia, window="a - left")
        show(ib, window="b - right")
        if cv2.waitKey() == ord('q'):
            break
    cv2.destroyAllWindows()


def draw_staff(
    page: Staffer.Page,
    shape: Optional[Tuple[int, int]] = None,
    background: Optional[MatLike] = None,
    thickness: int = 2
) -> MatLike:
    if shape is None:
        assert background is not None, "One of SIZE or BACKGROUND must be provided."
        shape = background.shape
        image = background
    else:
        assert shape is not None, "One of SIZE or BACKGROUND must be provided."
        image = np.full(shape, 255, dtype=np.uint8)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    staff_color = (255, 0, 0)
    for staff in page.staves:
        # Draws the bars.
        for x in staff.bars:
            cv2.line(
                rgb_image,
                (x, staff.rh_top),
                (x, staff.lh_bot),
                staff_color, thickness
            )
        # Draws the top most and bottom most staff lines:
        cv2.line(
            rgb_image,
            (staff.bars[0], staff.rh_top),
            (staff.bars[-1], staff.rh_top),
            staff_color, thickness
        )
        # Left hand staff
        cv2.line(
            rgb_image,
            (staff.bars[0], staff.lh_bot),
            (staff.bars[-1], staff.lh_bot),
            staff_color, thickness
        )
    return rgb_image


def list_directory(datadir: Path) -> List[Path]:
    files = []
    for root, _, filenames in os.walk(datadir):
        for filename in filenames:
            path = Path(root) / filename
            if path.suffix == '.krn' and path.with_suffix(".pdf").exists():
                files.append(path.with_suffix(""))
    return files


def do_staff(file: Path, do_plot: bool, dont_show: bool):
    staffer = Staffer(file, do_plot=do_plot)
    for pageno, (image, page) in enumerate(staffer.staff()):
        try:
            image = draw_staff(page, background=image.copy())
            if (not dont_show) and show(image):
                return True
        except Exception as e:
            print(f"{file}, page: {pageno}, no staff found:\n\t{e}")


@click.command()
@click.argument("path",
                type=click.Path(file_okay=True, dir_okay=True),
                default="/home/anselm/Downloads/KernSheet/")
@click.option("--plot", "do_plot", is_flag=True, default=False,
              help="Debug - Plots various data as we go.")
@click.option("--refresh", "refresh", is_flag=True, default=False,
              help="Don't use cached .numpy files, refresh them.")
@click.option("--dont-show", "dont_show", is_flag=True, default=False,
              help="Don't show image + staff after processing.")
def staff(path: Path, do_plot: bool, refresh: bool, dont_show: bool):
    path = Path(path)
    if path.is_dir():
        dataset = list_directory(Path(path))
        for path in dataset:
            print(path)
            if do_staff(path, do_plot, dont_show):
                break
    else:
        do_staff(path, do_plot, dont_show)


@click.group
def cli():
    pass


cli.add_command(staff)

if __name__ == "__main__":
    cli()
