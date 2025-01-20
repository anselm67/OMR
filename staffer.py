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
    rh_top: int
    lh_bot: int
    bars: List[int]


@dataclass
class Page:
    # For each double staff (right hand, left hand)
    # (rh_top: top of rh, lh_bot: bottom of lh)
    staves: List[Staff]


def is_binary(image: MatLike) -> bool:
    height, width = image.shape
    return np.sum(image == 0) + np.sum(image == 255) == height * width


def save_numpy(path: Path, array: List[MatLike]):
    with open(path, "wb") as f:
        pickle.dump(array, f)


def load_numpy(path: Path) -> List[MatLike]:
    with open(path, "rb") as f:
        return cast(List[MatLike], pickle.load(f))


def denoise(image: MatLike, block_size: int = -1, C: int = 2) -> MatLike:
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
    median_angle = np.median(hough_lines[:, 1])  # type: ignore
    return np.degrees(median_angle) - 90.0


def deskew(
    image: MatLike,
    min_rotation_angle_degrees=0.05,
    hough_thresold: int = 240
):
    angle = average_angle(image)
    if angle < min_rotation_angle_degrees:
        # For a 1200px width, 600px center to side:
        # height = 600 * sin(0.1 degrees) ~ 1 px
        return image
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D((width // 2, height//2), angle, 1)
    image = cv2.warpAffine(image, matrix, (width, height))
    print(f"\t{angle:2.4f} rotation => {average_angle(image):2.4f} degrees.")
    return image


def pdf2numpy(
    path: Path, width: int = SIZE[1], refresh: bool = False
) -> List[MatLike]:
    # Returns the cache version when available.
    numpy_path = path.with_suffix(".numpy")
    if numpy_path.exists() and not refresh:
        return load_numpy(numpy_path)

    # Runs the pdf conversion with all transformations.
    pdf = convert_from_path(path.with_suffix(".pdf"))

    def transform(orig_image: MatLike) -> MatLike:
        image = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2GRAY)
        image = denoise(image)
        image = deskew(image)
        # Rescale to requested width and pad to requested height.
        h, w = image.shape
        scale = width / w
        image = cv2.resize(
            image, (width, int(h * scale)), interpolation=cv2.INTER_AREA
        )
        # Some of the operations above de-binarize the images.
        image = denoise(image)
        return image

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
    page: Page,
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


def line_indices(lines: List[int]) -> List[Tuple[int, int]]:
    idx = 0
    out = []
    while idx+9 < len(lines):
        out.append((idx, idx+9))
        idx += 10
    return out


def find_bars(
        image: MatLike, fill_rate: float = 0.85, do_plot: bool = True
) -> List[int]:
    x_proj = np.sum(image, 0)
    # A bar goes from top to bottom, so we can compute a min height for
    # our peak by assuming a percentage fill rate on that vertical line.
    peak_height = int(image.shape[0] * fill_rate * 255)
    bar_peaks, properties = find_peaks(
        x_proj, distance=50,  height=peak_height)
    bar_peaks = [p.item() for p in bar_peaks]
    # bar_heights = [h.item() for h in properties['peak_heights']]
    # print(list(zip(bar_peaks, bar_heights)))

    if do_plot:
        plt.subplot()
        plt.vlines(bar_peaks, ymin=0, ymax=50000, colors='r')
        plt.plot(x_proj)
        plt.show()

    return bar_peaks


def find_staff(orig_image: MatLike, do_plot: bool = True) -> Page:
    # Computes the vertical and horizontal projections.
    image = cv2.bitwise_not(orig_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    y_lines = np.sum(image, 1)

    # Computes the moving average of the y projection and gets the peaks
    # of this series to obtain the rough position of each staff.
    wsize = 50
    y_proj = np.zeros(y_lines.shape)
    offset = (wsize - 1) // 2
    y_avg = np.convolve(y_lines, np.ones(wsize) / wsize, mode='valid')
    y_proj[offset:offset+len(y_avg)] = y_avg
    # The width=20 is the max allowed for chopin/mazurka/mazurka30-3
    # Height of 45_000 is for mozart/piano/sonata/sonata01-1
    staff_peaks, _ = find_peaks(y_proj, width=20, distance=50, height=45_000)

    # Around each staff peak, find the peaks corresponding to each staff line.
    staff_lines = []
    for peak in staff_peaks:
        peak_height = 75000
        staff_range = y_lines[peak-40:peak+40]
        line_peaks, properties = find_peaks(
            staff_range, height=peak_height, distance=7)
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
        elif len(line_peaks) < 5:
            # Lowers the thresold until we get 5 lines.
            while peak_height > 0 and len(line_peaks) < 5:
                peak_height -= 2500
                line_peaks, properties = find_peaks(
                    staff_range, height=peak_height, distance=7)
        staff_lines.extend([p+peak-40 for p in line_peaks])

    staff_lines = sorted(staff_lines)

    if do_plot:
        plt.subplot()
        plt.plot(y_lines, 'b', label='y_lines')
        plt.plot(y_proj, 'b', label='y_lines', linestyle="dotted")
        plt.vlines(staff_peaks, ymin=0, ymax=300000, colors='r')
        plt.vlines(staff_lines, ymin=0, ymax=100000, colors='g')
        plt.show()

    if len(staff_lines) % 10 != 0:
        raise ValueError(
            f"Number of lines {len(staff_lines)} should be divisible by 10."
        )

    page = Page(staves=[])
    positions = [
        (staff_lines[ridx], staff_lines[lidx]) for ridx, lidx in line_indices(staff_lines)
    ]

    # Left and right are top and last offsets of vertical lines.
    image = cv2.bitwise_not(orig_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    for rh_top, lh_bot in positions:
        bars = find_bars(image[rh_top:lh_bot, :], do_plot=do_plot)
        page.staves.append(Staff(rh_top, lh_bot, bars))

    return page


def list_directory(datadir: Path) -> List[Path]:
    files = []
    for root, _, filenames in os.walk(datadir):
        for filename in filenames:
            path = Path(root) / filename
            if path.suffix == '.krn' and path.with_suffix(".pdf").exists():
                files.append(path.with_suffix(""))
    return files


def do_staff(file: Path, refresh: bool, do_plot: bool, dont_show: bool):
    pages = pdf2numpy(file, refresh=refresh)
    for page_number, page in enumerate(pages):
        try:
            staff = find_staff(page, do_plot=do_plot)
            image = draw_staff(staff, background=page.copy())
            if (not dont_show) and show(image):
                return True
        except Exception as e:
            print(f"{file}, page: {page_number}, no staff found:\n\t{e}")


@click.command()
@click.argument("path",
                type=click.Path(file_okay=True, dir_okay=True),
                default="/home/anselm/Downloads/KernSheet/")
@click.option("--plot", "do_plot", is_flag=True, default=False,
              help="Debug - Plots various data as we go.")
@click.option("--refresh", "refresh", is_flag=True, default=False,
              help="Don't use cached .numpy files, refresh them.")
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
            if do_staff(path, refresh, do_plot, dont_show):
                break
    else:
        do_staff(path, refresh, do_plot, dont_show)


@click.group
def cli():
    pass


cli.add_command(staff)

if __name__ == "__main__":
    cli()
