#!/usr/bin/env python3

import os
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple, cast

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
        if height > max_height:
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


def find_staff(image: MatLike) -> Staff:
    staff = Staff(
        top_offset=0,
        left=0, right=0,
        positions=[],
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    lines = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    lines = image
    lines = cv2.bitwise_not(lines)
    y_lines = np.sum(lines, 1)
    x_lines = np.sum(lines, 0)

    plt.plot(y_lines, 'b', label='y_lines')
    window_size = 50
    pad = (window_size - 1) // 2
    xx = np.zeros(y_lines.shape)
    vals = np.convolve(y_lines, np.ones(window_size) /
                       window_size, mode='valid')
    xx[pad:pad+len(vals)] = vals
    # plt.plot(xx, 'r', label='convol')

    peaks, _ = find_peaks(xx, width=40, distance=50)
    print(peaks)
    plt.vlines(peaks, ymin=0, ymax=300000, colors='r')

    lines = []
    for peak in peaks:
        staff = y_lines[peak-40:peak+40]
        line_peaks, properties = find_peaks(
            staff, height=75000, distance=7)
        heights = properties['peak_heights']
        if len(line_peaks) > 5:
            x = sorted(
                zip(line_peaks, heights.tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            line_peaks = [line_peak for line_peak, _ in x]
            line_peaks = line_peaks[:5]

        lines = lines + [p+peak-40 for p in line_peaks]
        pass

    plt.vlines(lines, ymin=0, ymax=100000, colors='g')
    # plt.show()

    staff = Staff(
        0, 0, 0, []
    )
    staff = replace(staff, top_offset=lines[0])

    if len(lines) % 10 != 0:
        raise ValueError(
            f"Number of lines {len(lines)} should be divisible by 10."
        )
    else:
        staff = replace(staff, positions=[
            (lines[ridx], lines[lidx]) for ridx, lidx in line_indices(lines)])

    # Left and right are top and last offsets of vertical lines.
    x = np.nonzero(x_lines)[0]
    if x.size == 0:
        raise ValueError(f"margin-detection: we got a blank image?")
    staff = replace(staff, left=x[0], right=x[-1])

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


def list(datadir: Path) -> List[Path]:
    files = []
    for root, _, filenames in os.walk(datadir):
        for filename in filenames:
            path = Path(root) / filename
            if path.suffix == '.krn' and path.with_suffix(".pdf").exists():
                files.append(path.with_suffix(""))
    return files


dataset = list(Path("/home/anselm/Downloads/KernSheet/chopin/mazurka"))

for path in dataset:
    print(path)
    pages = pdf2numpy(path)
    for page in pages:
        staff = find_staff(page)
        image = draw_staff(staff, background=page)
        show(image)

# for image in tl:
#     staff = find_staff(image)
#     cut_sheet(image, staff)
#     # if show(create_staff(staff, background=image.copy())):
#     #     break
