

import json
import logging
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from pdf2image import convert_from_path
from scipy.signal import find_peaks

from kernsheet import KernSheet

# Default width for the annotated images.
# The annnotations are - of course -dependent on that width which is
# also stored in each Page instance.
IMAGE_WIDTH = 1200

# Name of the pdf cache directory.
# All the images for a given pdf may be cached in this child of the
# directory that contains them.
PDF_CACHE = ".pdfcache"


class Staffer:
    """Finds the layout of a score.

        Given a pdf file, Staffer converts each page to a normalized
        image, ready for training a model, and provides the corrdinates
        of staves on each page of the score.
    """

    @dataclass(frozen=True)
    class Staff:
        rh_top: int
        lh_bot: int
        bars: List[int]

    @dataclass(frozen=True)
    class Page:
        # Page number in the pdf (counting from 0)
        page_number: int

        # Image size for the coordinates in this Page
        image_width: int
        image_height: int

        # Staves and validation.
        staves: List['Staffer.Staff']
        validated: bool

        image_rotation: float = 0.0

        @staticmethod
        def from_dict(obj: Any):
            return replace(
                Staffer.Page(**obj),
                staves=[Staffer.Staff(**x) for x in obj["staves"]]
            )

    dataset: KernSheet
    key: str
    width: int
    do_plot: bool
    no_cache: bool

    pdf_cache: Optional[Path]
    pdf_path: Path
    json_path: Path

    pages: Optional[tuple[Page, ...]]
    data: Optional[tuple[tuple[MatLike, Page], ...]]

    @dataclass
    class Config:
        width: int = IMAGE_WIDTH
        do_plot: bool = False
        no_cache: bool = False
        pdf_cache: bool = True

    @property
    def kern_path(self) -> Path:
        return self.dataset.kern_path(self.key)

    def __init__(
        self, dataset: KernSheet, key: str,
        pdf_path: Path, json_path: Path,
        config: Config
    ):
        self.dataset = dataset
        self.key = key
        self.pdf_path = pdf_path
        self.json_path = json_path
        self.pdf_cache = pdf_path.parent / PDF_CACHE if config.pdf_cache else None
        self.width = config.width
        self.do_plot = config.do_plot
        self.no_cache = config.no_cache
        self.pages = None
        self.data = None

    def denoise(self, image: MatLike, block_size: int = -1, C: int = 2) -> MatLike:
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

    def average_angle(self, image: MatLike, hough_thresold: int = 500) -> float:
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        hough_lines = cv2.HoughLines(
            edges, 1, np.pi / 14400, hough_thresold,
            min_theta=np.radians(80), max_theta=np.radians(100)
        )
        if hough_lines is not None:
            hough_lines = hough_lines.squeeze(1)
            # Computes the median average angle:
            median_angle = np.median(hough_lines[:, 1])  # type: ignore
            return np.degrees(median_angle) - 90.0
        else:
            return 0.0

    def deskew(
        self,
        image: MatLike,
        min_rotation_angle_degrees=0.05,
    ) -> Tuple[float, MatLike]:
        angle = self.average_angle(image)
        if abs(angle) < min_rotation_angle_degrees:
            # For a 1200px width, 600px center to side:
            # height = 600 * sin(0.1 degrees) ~ 1 px
            return 0, image
        height, width = image.shape
        matrix = cv2.getRotationMatrix2D((width // 2, height//2), angle, 1)
        image = cv2.warpAffine(image, matrix, (width, height))
        print(f"\t{angle:2.4f} rotation => {
              self.average_angle(image):2.4f} degrees.")
        return angle, image

    def line_indices(self, lines: List[int]) -> List[Tuple[int, int]]:
        idx = 0
        out = []
        while idx+9 < len(lines):
            out.append((idx, idx+9))
            idx += 10
        return out

    def find_bars(
            self, image: MatLike, fill_rate: float = 0.8
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

        if self.do_plot:
            plt.subplot()
            plt.vlines(bar_peaks, ymin=0, ymax=50000, colors='r')
            plt.plot(x_proj)
            plt.show()

        return bar_peaks

    def filter_staff_peaks(self, staff_peaks, peak_heights) -> List[int]:
        # Checks the gaps for a first (title) or last (legend) outlier.
        gaps = staff_peaks[1:] - staff_peaks[:-1]
        if len(gaps) > 0:
            average_gap = sum(gaps) / len(gaps)
            if gaps[0] > 2 * average_gap:
                return staff_peaks[1:]
            elif gaps[-1] > 2 * average_gap:
                return staff_peaks[:-1]
        # Drops the looser, i.e. smallest peak.
        sorted_peaks = sorted(
            zip(staff_peaks, peak_heights),
            key=lambda x: x[1],
            reverse=True
        )
        staff_peaks = [staff_peak for staff_peak, _ in sorted_peaks]
        return staff_peaks[:-1]

    def transform(self, orig_image: MatLike) -> Tuple[float, MatLike]:
        image = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2GRAY)
        h, w = image.shape
        scale = self.width / w
        image = cv2.resize(
            image, (self.width, int(h * scale)), interpolation=cv2.INTER_AREA
        )
        image = self.denoise(image)
        rotation_angle, image = self.deskew(image)
        # Some of the operations above de-binarize the images.
        image = self.denoise(image)
        return float(rotation_angle), image

    def decode_page(self, orig_image: MatLike, pageno: int) -> Page:
        image_rotation, orig_image = self.transform(orig_image)
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
        staff_peaks, properties = find_peaks(
            y_proj, width=20, distance=50, height=40_000)

        if len(staff_peaks) % 2 != 0:
            staff_peaks = self.filter_staff_peaks(
                staff_peaks, properties['peak_heights'].tolist()
            )

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

        if self.do_plot:
            plt.subplot()
            plt.plot(y_lines, 'b', label='y_lines')
            plt.plot(y_proj, 'b', label='y_lines', linestyle="dotted")
            plt.vlines(staff_peaks, ymin=0, ymax=300000, colors='r')
            plt.vlines(staff_lines, ymin=0, ymax=100000, colors='g')
            plt.show()

        if len(staff_lines) % 10 != 0:
            # Trims off excess lines
            length_10 = 10 * (len(staff_lines) // 10)
            staff_lines = staff_lines[:length_10]
            print(f"Trimming off {len(staff_lines) -
                  length_10} bottom staff lines.")

        page = Staffer.Page(
            page_number=pageno,
            image_width=orig_image.shape[1],
            image_height=orig_image.shape[0],
            staves=list(),
            validated=False,
            image_rotation=image_rotation,
        )
        positions = [
            (staff_lines[ridx], staff_lines[lidx]) for ridx, lidx in self.line_indices(staff_lines)
        ]

        # Left and right are top and last offsets of vertical lines.
        image = cv2.bitwise_not(orig_image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        for rh_top, lh_bot in positions:
            bars = self.find_bars(image[rh_top:lh_bot, :])
            page.staves.append(Staffer.Staff(int(rh_top), int(lh_bot), bars))

        return page

    def decode_images(self):
        images = self.load_images_from_pdf(-1)

        pages = [
            self.decode_page(np.array(image), pageno) for pageno, image in enumerate(images)
        ]

        self.data = tuple(
            (self.apply_page_transforms(np.array(image), page), page)
            for image, page in zip(images, pages)
        )

    # Rescales and rotates the image according to the given page.
    def apply_page_transforms(self, image: MatLike, page: Page) -> MatLike:
        h, w = image.shape[:-1]
        scale = page.image_width / w
        image = cv2.resize(
            image, (self.width, int(h * scale)), interpolation=cv2.INTER_AREA
        )

        if abs(page.image_rotation) > 0:
            height, width = image.shape[:-1]
            matrix = cv2.getRotationMatrix2D(
                (width // 2, height//2), page.image_rotation, 1)
            image = cv2.warpAffine(image, matrix, (width, height))

        return image

    def load_images_from_pdf(self, expected_count: int = -1) -> tuple[MatLike, ...]:
        if self.pdf_cache and self.pdf_cache.exists():
            # Uses the cache when available.
            png_files = sorted(self.pdf_cache.glob(
                f"{self.pdf_path.stem}-*.png"))
            if expected_count > 0 and len(png_files) == expected_count:
                logging.info(f"Reading pages from cache: {self.pdf_cache}")
                return tuple(
                    np.array(cv2.imread(png_file.as_posix()))
                    for png_file in png_files
                )
            elif expected_count >= 0:
                # The cache is invalid.
                logging.info(
                    f"Invalidating cache for {self.pdf_path}: "
                    f"got {len(png_files)} instead of {expected_count}"
                )
                shutil.rmtree(self.pdf_cache)

        # Extracts images from pdf and caches them.
        pdf = convert_from_path(self.pdf_path)
        images = tuple(np.array(image) for image in pdf)
        if self.pdf_cache:
            logging.info(f"Caching images for {self.pdf_path.name}")
            for idx, image in enumerate(images):
                file = self.pdf_cache / \
                    f"{self.pdf_path.stem}-{idx:03d}.png"
                file.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(file.as_posix(), image)

        return images

    def load_images(self):
        assert self.pages is not None
        images = self.load_images_from_pdf(len(self.pages))
        self.data = tuple(
            (self.apply_page_transforms(image, page), page)
            for image, page in zip(images, self.pages)
        )

    def save(self, pages: Optional[tuple[Page, ...]] = None):
        if pages is None:
            if self.pages is not None:
                pages = self.pages
            else:
                assert self.data is not None
                pages = tuple(page for _, page in self.data)
        elif self.pages is not None:
            # Update self.pages and save.
            assert len(self.pages) == len(pages), \
                f"Expected {len(self.pages)} pages, got {len(pages)}."
            self.pages = pages
        else:
            assert self.data is not None and len(self.data) == len(pages), \
                f"Expected {len(pages)}."
            self.data = tuple(
                (image, page) for (image, _), page in zip(self.data, pages)
            )
        with open(self.json_path, "w+") as fp:
            json.dump({
                "pdf_path": self.key,
                "pages": [asdict(page) for page in pages],
            }, fp, indent=4)

    def load_pages(self):
        if self.no_cache:
            return False
        if self.json_path.exists():
            with open(self.json_path, "r") as fp:
                obj = json.load(fp)
            self.pages = tuple(Staffer.Page.from_dict(s) for s in obj['pages'])
            assert obj['pdf_path'] == self.key, f"Expecting key {
                self.key} to match .json pdf path {obj['pdf_path']}."
            return True
        return False

    def get_pages(self) -> Optional[Iterable[Page]]:
        if self.data:
            return (page for _, page in self.data)
        elif self.pages or self.load_pages():
            return self.pages
        else:
            return None

    def staff(self) -> tuple[tuple[MatLike, Page], ...]:
        if self.data is None:
            if self.load_pages():
                self.load_images()
            else:
                self.decode_images()
                self.save()
        assert self.data is not None, f"{self.key}: no staff found."
        return self.data

    def unlink_pdf(self):
        # We can't work with this .pdf file, remove it and kill any saved state.
        self.pdf_path.unlink(missing_ok=True)
        self.json_path.unlink(missing_ok=True)
        self.data = None

    def is_validated(self) -> bool:
        if (pages := self.get_pages()) is not None:
            assert self.pages is not None, "load_page() succeeded without pages."
            return all((page.validated for page in pages))
        else:
            return False
