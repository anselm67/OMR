#!/usr/bin/env python3

import os
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple, cast

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from pdf2image import convert_from_path
from scipy.signal import find_peaks

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

SIZE = (1800, 1200)


@dataclass
class Staff:
    top_offset: int
    left: int
    right: int
    # For each double staff (right hand, left hand)
    # (rh_top: top of rh, lh_bot: bottom of lh)
    positions: List[Tuple[int, int]]


def save_numpy(path: Path, array: List[MatLike]):
    with open(path, "wb") as f:
        pickle.dump(array, f)


def load_numpy(path: Path) -> List[MatLike]:
    with open(path, "rb") as f:
        return cast(List[MatLike], pickle.load(f))


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


def average_angle(image: MatLike, hough_thresold: int = 500) -> float:
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    hough_lines = cv2.HoughLines(
        edges, 1, np.pi / 14400, hough_thresold,
        min_theta=np.radians(80), max_theta=np.radians(100)
    )
    hough_lines = hough_lines.squeeze(1)
    # Computes the median average angle:
    return np.median(hough_lines[:, 1]).item()  # type: ignore


def deskew(
    image: MatLike,
    min_rotation_angle_degrees=0.05,
    hough_thresold: int = 240
):
    avg_angle = average_angle(image, hough_thresold=hough_thresold)
    angle = np.degrees(avg_angle) - 90.0
    if angle < min_rotation_angle_degrees:
        # For a 1200px width, 600px center to side:
        # height = 600 * sin(0.1 degrees) ~ 1 px
        return image
    print(f"\tapplying {angle:2.4f} rotation.")
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D((width // 2, height//2), angle, 1)
    return cv2.warpAffine(image, matrix, (width, height))


def pdf2numpy(
    path: Path, width: int = SIZE[1], refresh: bool = False
) -> List[MatLike]:
    # Returns the cache version when available.
    numpy_path = path.with_suffix(".numpy")
    if numpy_path.exists() and not refresh:
        return load_numpy(numpy_path)

    # Runs the pdf conversion with all transformations.
    pdf = convert_from_path(path.with_suffix(".pdf"))

    def transform(image: MatLike) -> MatLike:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        image = denoise(image)
        image = deskew(image)
        # Rescale to requested width and pad to requested height.
        h, w = image.shape
        scale = width / w
        return cv2.resize(
            image, (width, int(h * scale)), interpolation=cv2.INTER_AREA
        )

    # Caches the results.
    images = [transform(np.array(image)) for _, image in enumerate(pdf)]
    save_numpy(path.with_suffix(".numpy"), images)
    return images


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
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    staff_color = (255, 0, 0)
    for rh_top, lh_bot in staff.positions:
        # Right hand staff:
        cv2.line(
            rgb_image,
            (staff.left - padding, rh_top),
            (staff.right + padding, rh_top),
            staff_color, thickness
        )
        # Left hand staff
        cv2.line(  # type: ignore
            rgb_image,
            (staff.left - padding, lh_bot),
            (staff.right + padding, lh_bot),
            staff_color, thickness
        )
        bottom = lh_bot
    cv2.line(
        rgb_image,
        (staff.left, staff.top_offset),
        (staff.left, bottom),
        staff_color, thickness
    )
    cv2.line(
        rgb_image,
        (staff.right, staff.top_offset),
        (staff.right, bottom),
        staff_color, thickness
    )
    return rgb_image


def dedup(y_lines: MatLike, min_gap: int = 2) -> MatLike:
    if not y_lines.size or len(y_lines) <= 1:
        return y_lines
    output = [y_lines[0]]
    for y in y_lines[1:]:
        if y - output[-1] > min_gap:
            output.append(y)
    return np.array(output)


def line_indices(lines: List[int]) -> List[Tuple[int, int]]:
    idx = 0
    out = []
    while idx+9 < len(lines):
        out.append((idx, idx+9))
        idx += 10
    return out


def find_staff(image: MatLike, do_plot: bool = True) -> Staff:
    # Computes the vertical and horizontal projections.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    staff_lines = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    staff_lines = image
    staff_lines = cv2.bitwise_not(staff_lines)
    y_lines = np.sum(staff_lines, 1)
    x_lines = np.sum(staff_lines, 0)

    # Computes the moving average of the y projection and gets the peaks
    # of this series to obtain the rough position of each staff.
    wsize = 50
    y_proj = np.zeros(y_lines.shape)
    offset = (wsize - 1) // 2
    y_avg = np.convolve(y_lines, np.ones(wsize) / wsize, mode='valid')
    y_proj[offset:offset+len(y_avg)] = y_avg
    # The width=30 is the max allowed for chopin/mazurka/mazurka07-3
    staff_peaks, _ = find_peaks(y_proj, width=30, distance=50)

    # Around each staff peak, find the peaks corresponding to each staff line.
    staff_lines = []
    for peak in staff_peaks:
        staff_range = y_lines[peak-40:peak+40]
        line_peaks, properties = find_peaks(
            staff_range, height=75000, distance=7)
        if len(line_peaks) > 5:
            # We're pretty lax on our find_peaks() param, in case we
            # get too many peaks, we throw the smallest ones out.
            sorted_peaks = sorted(
                zip(line_peaks, properties['peak_heights'].tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            line_peaks = [line_peak for line_peak, _ in sorted_peaks]
            line_peaks = line_peaks[:5]

        staff_lines.extend([p+peak-40 for p in line_peaks])

    staff_lines = sorted(staff_lines)

    if do_plot:
        plt.subplot()
        plt.plot(y_lines, 'b', label='y_lines')
        plt.vlines(staff_peaks, ymin=0, ymax=300000, colors='r')
        plt.vlines(staff_lines, ymin=0, ymax=100000, colors='g')
        plt.show()

    staff_range = Staff(
        top_offset=0,
        left=0, right=0,
        positions=[],
    )
    staff_range = replace(staff_range, top_offset=staff_lines[0])

    if len(staff_lines) % 10 != 0:
        raise ValueError(
            f"Number of lines {len(staff_lines)} should be divisible by 10."
        )
    else:
        staff_range = replace(staff_range, positions=[
            (staff_lines[ridx], staff_lines[lidx]) for ridx, lidx in line_indices(staff_lines)])

    # Left and right are top and last offsets of vertical lines.
    x = np.nonzero(x_lines)[0]
    if x.size == 0:
        raise ValueError(f"margin-detection: we got a blank image?")
    staff_range = replace(staff_range, left=x[0], right=x[-1])

    return staff_range


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


def list(datadir: Path) -> List[Path]:
    files = []
    for root, _, filenames in os.walk(datadir):
        for filename in filenames:
            path = Path(root) / filename
            if path.suffix == '.krn' and path.with_suffix(".pdf").exists():
                files.append(path.with_suffix(""))
    return files


@click.command()
@click.option("--plot", "do_plot", is_flag=True, default=False)
def all(do_plot: bool):
    dataset = list(Path("/home/anselm/Downloads/KernSheet/chopin/mazurka"))
    for path in dataset:
        print(path)
        pages = pdf2numpy(path)
        for page in pages:
            staff = find_staff(page, do_plot=do_plot)
            image = draw_staff(staff, background=page)
            show(image)


@click.command()
@click.argument("path", type=click.Path(exists=False))
@click.option("--plot", "do_plot", is_flag=True, default=False)
def staff(path: Path, do_plot: bool):
    path = Path(path)
    pages = pdf2numpy(path)
    for page in pages:
        staff = find_staff(page, do_plot=do_plot)
        image = draw_staff(staff, background=page)
        show(image)


@click.group
def cli():
    pass


cli.add_command(all)
cli.add_command(staff)

if __name__ == "__main__":
    cli()

# for image in tl:
#     staff = find_staff(image)
#     cut_sheet(image, staff)
#     # if show(create_staff(staff, background=image.copy())):
#     #     break
