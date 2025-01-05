#!/usr/bin/env python3

import pickle
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from pdf2image import convert_from_path

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

DATADIR = Path("untracked")


@dataclass
class Staff:
    top_offset: int
    left: int
    right: int
    # For each double staff (right hand, left hand)
    # (rh_top: top of rh, lh_bot: bottom of lh)
    positions: List[Tuple[int, int]]


dataset = Path("/home/anselm/Downloads/dataset/pdfs")
wtc_one = dataset / "IMSLP222726-PMLP05948-Bach_Prelude_BWV846.pdf"
wtc_full = dataset / "IMSLP411479-PMLP05948-bach-wtk-ur-1.pdf"
cho_full = dataset / "IMSLP113278-PMLP01969-FChopin_Etudes,_Op.10_BH2.pdf"
moz_full = dataset / "IMSLP00230-Fantasy_in_d,_K_397.pdf"

wset = [
    (wtc_one, "wtc_1.pkl", 1),
    (wtc_full, "wtc_2.pkl", 3),
    (wtc_full, "wtc_3.pkl", 4),
    (wtc_full, "wtc_4.pkl", 5),
    (wtc_full, "wtc_5.pkl", 6),
    (cho_full, "wtc_6.pkl", 1),
    (cho_full, "wtc_7.pkl", 2),
    (moz_full, "wtc_8.pkl", 0),
]


def save_numpy(target: Path | str, array: MatLike):
    with open(DATADIR / target, "wb") as f:
        pickle.dump(array, f)


def load_numpy(source: Path) -> MatLike:
    with open(DATADIR / source, "rb") as f:
        return cast(MatLike, pickle.load(f))


def get_page(pdf: Path | str, pageno: int) -> MatLike:
    pages = convert_from_path(pdf)
    return cv2.cvtColor(np.array(pages[pageno]), cv2.COLOR_RGB2GRAY)


points = []


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            print(f"dx: {x2-x1}, dy: {y2 -
                  y1}, d: {((x2-x1)**2+(y2-y1)**2) ** 0.5}")
            points.clear()


def show(*images: MatLike,
         crop: Optional[Tuple[int, int]] = None,
         window: Optional[str] = None):
    interrupted = False
    for image in images:
        if crop is not None:
            height, width = crop
            image = image[0:height, 0:width]
        cv2.imshow(window or "window", image)
        cv2.setMouseCallback(window or "window", click_event)
        cv2.moveWindow(window or "window", 100, 100)
        if window is None and (interrupted := (cv2.waitKey() == ord('q'))):
            break
    if window is None:
        cv2.destroyAllWindows()
    return interrupted


def compare(a: List[MatLike], b: List[MatLike], crop=None):
    for ia, ib in zip(a, b):
        show(ia, window="left", crop=crop)
        show(ib, window="right", crop=crop)
        if cv2.waitKey() == ord('q'):
            break
    cv2.destroyAllWindows()


def save_some(some: List[Tuple[Path | str, str, int]]):
    for pdf, target, pageno in some:
        print(f"Extracting page {pageno} from {pdf}")
        save_numpy(target, get_page(pdf, pageno))


def load_some(*paths: str, width: int = 1200) -> List[MatLike]:
    def transform(image: MatLike) -> MatLike:
        h, w = image.shape
        scale = width / w
        return cv2.resize(image, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return [transform(load_numpy(Path(path))) for path in paths]


def denoise(image: MatLike, block_size: int = 11, C: int = 2) -> MatLike:
    if block_size > 0:
        return cv2.adaptiveThreshold(
            image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=C
        )
    else:
        _, image = cv2.threshold(
            image,
            127, 255,
            cv2.THRESH_OTSU
        )
        return image


def create_staff(
    staff: Staff,
    shape: Optional[Tuple[int, int]] = None,
    background: Optional[MatLike] = None,
    padding: int = 50,
    thickness: int = 2
) -> MatLike:
    if shape is None:
        assert background is not None, "One of SIZE or BACKGROUND must be provided."
        shape = background.shape
        image = background
    else:
        assert shape is not None, "One of SIZE or BACKGROUND must be provided."
        image = np.full(shape, 255, dtype=np.uint8)
    for rh_top, lh_bot in staff.positions:
        # Right hand staff:
        cv2.line(
            image,
            (staff.left - padding, rh_top),
            (staff.right + padding, rh_top),
            BLACK, thickness
        )
        # Left hand staff
        cv2.line(  # type: ignore
            image,
            (staff.left - padding, lh_bot),
            (staff.right + padding, lh_bot),
            BLACK, thickness
        )
        bottom = lh_bot
    cv2.line(
        image,
        (staff.left, staff.top_offset),
        (staff.left, bottom),
        BLACK, thickness
    )
    cv2.line(
        image,
        (staff.right, staff.top_offset),
        (staff.right, bottom),
        BLACK, thickness
    )
    return image


def dedup(y_lines: MatLike, min_gap: int = 2) -> MatLike:
    if not y_lines.size or len(y_lines) <= 1:
        return y_lines
    output = [y_lines[0]]
    for y in y_lines[1:]:
        if y - output[-1] > min_gap:
            output.append(y)
    return np.array(output)


def line_indices(lines: MatLike) -> List[Tuple[int, int]]:
    idx = 0
    out = []
    while idx+9 < lines.size:
        out.append((idx, idx+9))
        idx += 10
    return out


def find_staff(image: MatLike) -> Staff:
    staff = Staff(
        top_offset=0,
        left=0, right=0,
        positions=[],
    )
#    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
#    lines = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    lines = image
    lines = cv2.bitwise_not(lines)
    y_lines = np.sum(lines, 1)
    x_lines = np.sum(lines, 0)

    if False:
        plt.plot(y_lines)
        plt.show()

    # Left and right are top and last offsets of vertical lines.
    x = np.nonzero(x_lines)[0]
    if x.size == 0:
        raise ValueError(f"margin-detection: we got a blank image?")
    staff = replace(staff, left=x[0], right=x[-1])

    fudge = 0.6
    fudge_step = 0.05
    while True:
        high = fudge * np.max(y_lines)
        # Candidates horizontal lines.
        lines = np.where(y_lines > high)[0]
        lines = dedup(lines)
        if lines.size % 10 == 0:
            break
        elif lines.size % 10 > 5:
            # Decrease fudget factor to get more lines.
            fudge -= fudge_step
            fudge_step /= 2
        else:
            # Increase the fudge to get less lines.
            fudge += fudge_step
            fudge_step /= 2
        print(f"fudge {fudge}")

    staff = replace(staff, top_offset=lines[0])

    if lines.size % 10 != 0:
        raise ValueError(f"Number of lines {
                         lines.size} should be divisible by 10.")
    else:
        staff = replace(staff, positions=[
                        (lines[ridx], lines[lidx]) for ridx, lidx in line_indices(lines)])
    return staff


def histo(a: MatLike) -> bool:
    values, counts = np.unique(a, return_counts=True)
    for v, c in zip(values, counts):
        print(f"{v}: {c}")
    return len(values) == 2 and values[0] == 0 and values[1] == 255


def cut_sheet(image: MatLike, staff: Staff) -> List[MatLike]:
    interstaff = 0
    for (rh_top, _), (_, lh_bot) in zip(staff.positions[1:], staff.positions[:-1]):
        interstaff += (rh_top - lh_bot)
    interstaff //= len(staff.positions) - 1
    rolls = []
    for rh_top, lh_bot in staff.positions:
        top = rh_top - interstaff // 2
        bot = lh_bot + interstaff // 2
        rolls.append(image[top:bot, staff.left:staff.right])
        if show(rolls[-1]):
            break
    return rolls


crop = (800, 1200)
l = load_some(*[path for _, path, _ in wset])
tl = [denoise(x, block_size=11) for x in l]
# st = [find_staff(t) for t in tl]

# compare(tl, st, crop)


# img = create_staff((1552, 1200), default_staff)
# show(img)
for image in tl:
    staff = find_staff(image)
    cut_sheet(image, staff)
    # if show(create_staff(staff, background=image.copy())):
    #     break

sys.exit(0)
