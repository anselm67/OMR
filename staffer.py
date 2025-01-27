

import json
import logging
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from pdf2image import convert_from_path
from scipy.signal import find_peaks


class KernReader:

    lines: List[str]
    bars: Dict[int, int] = {}

    @property
    def bar_count(self) -> int:
        return len(self.bars)

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.load_tokens()

    BAR_RE = re.compile(r'^=+\s*(\d+)?.*$')

    def load_tokens(self):
        with open(self.path.with_suffix(".tokens"), "r") as fp:
            self.lines = [line.strip() for line in fp.readlines()]
        # Constructs the bars index.
        for lineno in range(0, len(self.lines)):
            line = self.lines[lineno]
            if (m := self.BAR_RE.match(line)):
                if m.group(1) is not None:
                    self.bars[int(m.group(1))] = lineno
        logging.info(f"{len(self.bars) - 1} bars in {self.path}")

    def has_bar_zero(self):
        return 0 in self.bars

    def get_text(self, barno: int) -> Optional[List[str]]:
        bos = self.bars.get(barno, -1)
        if bos >= 0:
            # Includes the marker for the next bar, feels more comfortable.
            eos = self.bars.get(barno + 1, -1) + 1
            return self.lines[bos:eos] if eos > 0 else self.lines[bos:]
        else:
            return None

    def header(self):
        return self.lines[:10]


class Staffer:
    """Finds the layout of a score.

        Given a pdf file, Staffer converts each page to a normalized
        image, ready for training a model, and provides the corrdinates
        of staves on each page of the score.
    """
    WIDTH = 1200

    @dataclass
    class Staff:
        rh_top: int
        lh_bot: int
        bars: List[int]

    @dataclass
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

    datadir: Path
    key: str
    width: int
    do_plot: bool
    no_cache: bool
    data: Optional[List[Tuple[MatLike, Page]]]

    @property
    def pdf_path(self) -> Path:
        return (self.datadir / self.key).with_suffix(".pdf")

    @property
    def kern_path(self) -> Path:
        return (self.datadir / self.key).with_suffix(".krn")

    @property
    def json_path(self) -> Path:
        return (self.datadir / self.key).with_suffix(".json")

    def __init__(
        self, datadir: Path, pdf_path: str,
        width: int = WIDTH, do_plot: bool = False, no_cache: bool = False
    ):
        self.datadir = Path(datadir)
        self.key = pdf_path
        self.width = width
        self.do_plot = do_plot
        self.no_cache = no_cache
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

    def load_images(self, pages: List[Page]) -> List[MatLike]:
        pdf = convert_from_path(self.pdf_path)

        return [
            self.apply_page_transforms(np.array(image), page)
            for image, page in zip(pdf, pages)
        ]

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

    def decode_images(self) -> List[Tuple[MatLike, Page]]:
        pdf = convert_from_path(self.pdf_path)

        pages = [
            self.decode_page(np.array(image), pageno) for pageno, image in enumerate(pdf)
        ]

        return [
            (self.apply_page_transforms(np.array(image), page), page)
            for image, page in zip(pdf, pages)
        ]

    def save(self):
        assert self.data is not None, "Can't save empty data."
        with open(self.json_path, "w+") as fp:
            json.dump({
                "pdf_path": self.key,
                "pages": [asdict(page) for _, page in self.data],
            }, fp, indent=4)

    def load_if_exists(self):
        if self.no_cache:
            return False
        if self.json_path.exists():
            with open(self.json_path, "r") as fp:
                obj = json.load(fp)
            pages = [Staffer.Page.from_dict(s) for s in obj['pages']]
            assert obj['pdf_path'] == self.key, f"Expecting key {
                self.key} to match .json pdf path {obj['pdf_path']}."
            self.data = list(zip(self.load_images(pages), pages))
            return True
        return False

    def staff(self) -> List[Tuple[MatLike, Page]]:
        if self.data is None:
            if not self.load_if_exists():
                self.data = self.decode_images()
                self.save()
        assert self.data is not None, f"{self.key}: no staff found."
        return self.data

    def unlink_pdf(self):
        # We can't work with this .pdf file, remove it and kill any saved state.
        self.pdf_path.unlink(missing_ok=True)
        self.json_path.unlink(missing_ok=True)
        self.data = None

    def draw_page(
        self, image: MatLike, page: Page, bar_offset: int,
        selected_staff: int = -1,
        selected_bar: int = -1,
        thickness: int = 2
    ) -> MatLike:
        BLUE = (255, 0, 0)
        RED = (0, 0, 255)
        GREEN = (2, 7*16+1, 4*16+8)
        style, selected_style = (BLUE, 2), (RED, 4)
        if page.validated:
            style = (GREEN, 2)
        rgb_image = image.copy()
        if len(page.staves) == 0:
            # That's fine let the user know.
            cv2.putText(
                rgb_image, "Page Validated" if page.validated else "Validate that page.",
                (400, 100),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=style[0], thickness=style[1]
            )

        for staffno, staff in enumerate(page.staves):
            if len(staff.bars) == 0:
                width = rgb_image.shape[1]
                color, thickness = selected_style if (
                    staffno == selected_staff) else style
                cv2.line(
                    rgb_image,
                    (0, staff.rh_top),
                    (width, staff.rh_top),
                    color, thickness
                )
                # Left hand staff
                cv2.line(
                    rgb_image,
                    (0, staff.lh_bot),
                    (width, staff.lh_bot),
                    color,
                    thickness
                )
                continue
            # Draws the bars.
            for barno, bar in enumerate(staff.bars):
                color, thickness = selected_style if (staffno == selected_staff) and (
                    barno == selected_bar) else style
                cv2.line(
                    rgb_image,
                    (bar, staff.rh_top),
                    (bar, staff.lh_bot),
                    color, thickness
                )
                # Renders the bar number only if not last of staff.
                if barno != len(staff.bars) - 1:
                    cv2.putText(
                        rgb_image, str(bar_offset + barno),
                        (bar+3, staff.rh_top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=color, thickness=thickness)
            bar_offset += len(staff.bars) - 1
            # Draws the top most and bottom most staff lines:
            color, thickness = selected_style if (
                staffno == selected_staff) else style
            cv2.line(
                rgb_image,
                (staff.bars[0], staff.rh_top),
                (staff.bars[-1], staff.rh_top),
                color, thickness
            )
            # Left hand staff
            cv2.line(
                rgb_image,
                (staff.bars[0], staff.lh_bot),
                (staff.bars[-1], staff.lh_bot),
                color,
                thickness
            )
        return rgb_image

    class EditorState:
        position: int = 0
        selected_staff: int = 0
        selected_bar: int = 0
        bar_offset: int = 0
        scale_ratio: float = 1.0

        data: List[Tuple[MatLike, 'Staffer.Page']]
        image: MatLike
        page: 'Staffer.Page'

        def __init__(self, data: List[Tuple[MatLike, 'Staffer.Page']]):
            self.data = data
            self.position = 0
            self.image, self.page = self.data[self.position]
            self.selected_staff = 0
            self.selected_bar = 0
            if len(self.page.staves) <= 0:
                self.selected_staff = -1
                self.selected_bar = -1
            elif len(self.page.staves[0].bars) <= 0:
                self.selected_bar = -1

        def get_bar_offset(self) -> int:
            barno = 0
            for _, page in self.data[:self.position]:
                for staff in page.staves:
                    barno += len(staff.bars) - 1
            return barno

        def get(self) -> Tuple[MatLike, 'Staffer.Page']:
            return self.image, self.page

        def next(self):
            if self.position + 1 >= len(self.data):
                print("End of score.")
                return
            self.position += 1
            self.image, self.page = self.data[self.position]
            self.selected_staff = 0
            if len(self.page.staves) == 0:
                self.selected_staff = -1
            elif len(self.page.staves[0].bars) == 0:
                self.selected_bar = -1
            else:
                self.selected_bar = 0

        def prev(self, select_last: bool = False):
            if self.position - 1 < 0:
                print("Beginning of score.")
                return
            self.position -= 1
            self.image, self.page = self.data[self.position]
            staff_count = len(self.page.staves)
            if staff_count == 0:
                self.selected_staff = -1
            elif select_last:
                self.selected_staff = staff_count - 1
            else:
                self.selected_staff = 0
            bars_len = len(self.page.staves[self.selected_staff].bars)
            if self.selected_staff < 0 or bars_len == 0:
                self.selected_bar = -1
            else:
                self.selected_bar = bars_len - 1 if select_last else 0

        def select_prev_staff(self, select_last: bool = False):
            if self.selected_staff - 1 < 0:
                self.prev(select_last=True)
            else:
                self.selected_staff = self.selected_staff - 1
                self.selected_bar = 0
                bar_count = len(self.page.staves[self.selected_staff].bars)
                if bar_count <= 0:
                    self.selected_bar = -1
                elif select_last:
                    self.selected_bar = bar_count - 1

        def select_next_staff(self):
            if self.selected_staff + 1 >= len(self.page.staves):
                self.next()
            else:
                self.selected_staff = self.selected_staff + 1
                self.selected_bar = 0
                if len(self.page.staves[self.selected_staff].bars) <= 0:
                    self.selected_bar = -1

        def select_next_bar(self):
            bar_count = len(self.page.staves[self.selected_staff].bars)
            if self.selected_bar + 1 >= bar_count:
                self.select_next_staff()
            else:
                self.selected_bar += 1

        def select_prev_bar(self):
            if self.selected_bar - 1 < 0:
                self.select_prev_staff(select_last=True)
            else:
                self.selected_bar -= 1

        def staff_height(self):
            # Tries to make a good guess at staff height.
            heights = [staff.lh_bot -
                       staff.rh_top for staff in self.page.staves]
            if len(heights) > 0:
                return int(sum(heights) / len(heights))
            else:
                return 128

        def add_staff(self):
            height = self.staff_height()
            if self.selected_staff < 0:
                ypos = 100
                self.page.staves.append(Staffer.Staff(
                    rh_top=ypos,
                    lh_bot=ypos + height,
                    bars=list()
                ))
                self.selected_staff = len(self.page.staves) - 1
            else:
                ypos = self.page.staves[self.selected_staff].lh_bot + 10
                self.page.staves.insert(
                    self.selected_staff + 1,
                    Staffer.Staff(
                        rh_top=ypos,
                        lh_bot=ypos + height,
                        bars=list()
                    )
                )
                self.selected_staff += 1

        def add_bar(self, offset: int = -1):
            staff = self.page.staves[self.selected_staff]
            if offset < 0:
                # Adds a bar after the selected one.
                if self.selected_bar < 0:
                    offset = 10
                    staff.bars.append(offset)
                else:
                    offset = staff.bars[self.selected_bar] + 10
                    staff.bars.insert(self.selected_bar + 1, offset)
            else:
                staff.bars.append(offset)
            staff.bars = sorted(staff.bars)
            self.selected_bar = staff.bars.index(offset)

        def delete_selected_bar(self):
            if self.selected_bar < 0:
                return
            del self.page.staves[self.selected_staff].bars[self.selected_bar]
            self.selected_bar = max(0, self.selected_bar - 1)

        def delete_selected_staff(self):
            if self.selected_staff < 0:
                return
            del self.page.staves[self.selected_staff]
            if len(self.page.staves) == 0:
                self.selected_staff = -1
                self.selected_bar = -1
            else:
                self.selected_staff = max(0, self.selected_staff - 1)
                self.selected_bar = 0

    def mouse_callback(self, state: EditorState,  event: int, x: int, y: int) -> bool:
        if event == cv2.EVENT_LBUTTONDOWN:
            # Selects the staff at y, adds a bar at x, and select new bar.
            for pos, staff in enumerate(state.page.staves):
                if staff.rh_top <= y <= staff.lh_bot:
                    state.selected_staff = pos
                    state.add_bar(x)
                    return True
        return False

    STAFFER_WINDOW = "Staffer"

    def edit(self, max_height: int = 992) -> bool:
        """Edits the staff.

        Args:
            max_height (int, optional): The requested editor height. Defaults to 992.

        Returns:
            bool: True if the user wishes to continue editing further documents,
                False otherwise.
        """
        state = Staffer.EditorState(self.staff())
        kern = KernReader(self.kern_path)

        cv2.namedWindow(self.STAFFER_WINDOW)

        def update_ui():
            image = self.draw_page(
                state.image, state.page, bars_offset, state.selected_staff, state.selected_bar
            )
            height, width = image.shape[:2]
            if max_height > 0 and height > max_height:
                state.scale_ratio = height / max_height
                new_width = int(max_height * width / height)
                image = cv2.resize(image, (new_width, max_height))
            else:
                state.scale_ratio = 1.0
            cv2.imshow(self.STAFFER_WINDOW, image)

        def mouse_callback(event: int, x: int, y: int, _1: int, _2: Any):
            x, y = int(x * state.scale_ratio), int(y * state.scale_ratio)
            if self.mouse_callback(state, event, x, y):
                update_ui()

        cv2.setMouseCallback(self.STAFFER_WINDOW, mouse_callback)

        while True:

            bars_offset = state.get_bar_offset()
            if not kern.has_bar_zero():
                bars_offset += 1

            update_ui()

            key = cv2.waitKey()
            if key == ord('q'):         # Quits editing.
                return False
            elif key == ord('s'):
                # Marks all pages as reviewed before saving.
                assert self.data is not None, "Makes the type checker happy."
                self.save()
                print(f"{len(self.data)} pages reviewed and saved to {
                      self.json_path}.")
            elif key == ord('a'):
                state.add_bar()
            elif key == ord('w'):
                state.add_staff()
            elif key == ord('x'):
                state.delete_selected_staff()
            elif key == ord('d'):
                state.delete_selected_bar()
            elif key == ord('n'):   # Moves onto the next document if any.
                return True
            elif key == ord(' '):
                state.next()
            elif key == ord('p'):
                state.prev()
            elif key == ord('i'):
                state.page.staves[state.selected_staff].lh_bot -= 2
                state.page.staves[state.selected_staff].rh_top -= 2
            elif key == ord('m'):
                state.page.staves[state.selected_staff].lh_bot += 2
                state.page.staves[state.selected_staff].rh_top += 2
            elif key == ord('e'):   # Extendss staff height.
                state.page.staves[state.selected_staff].lh_bot += 1
            elif key == ord('r'):   # Reduces staff height.
                state.page.staves[state.selected_staff].lh_bot -= 1
            elif key == ord('j'):    # Moves selected bar left.
                state.page.staves[state.selected_staff].bars[state.selected_bar] -= 2
            elif key == ord('k'):
                # Show kerns matching this bar.
                barno = bars_offset
                for i in range(0, state.selected_staff):
                    barno += len(state.page.staves[i].bars) - 1
                barno += state.selected_bar
                # Clears the terminal and displays the kern tokens:
                print('\033[2J', end='')
                print('\033[H', end='')
                records = kern.get_text(barno)
                if records is None:
                    print(f"No records found for bar {barno}")
                else:
                    print(f"Bar {barno} / {kern.bar_count}:")
                    for record in records:
                        print(record)
            elif key == ord('l'):    # Moves selected bar right.
                state.page.staves[state.selected_staff].bars[state.selected_bar] += 2
            elif key == ord('v'):
                state.page.validated = not state.page.validated
            elif key == 84:     # Key down
                state.select_next_staff()
            elif key == 82:     # Key up
                state.select_prev_staff()
            elif key == 83:     # Key left
                state.select_next_bar()
            elif key == 81:     # Key right
                state.select_prev_bar()
            elif key == ord('1'):
                self.unlink_pdf()
                print(f"{self.key} cleaned-up.")
                return True
            elif key == ord('t'):
                # Clears the terminal and displays the kern tokens:
                print('\033[2J', end='')
                print('\033[H', end='')
                print(f"Header for {self.kern_path} - {kern.bar_count} bars")
                for line in kern.header():
                    print(line)
            elif key == ord('c'):
                # Recomputes the bars of the selected staff
                if state.selected_staff >= 0:
                    staff = state.page.staves[state.selected_staff]
                    image = state.image[staff.rh_top:staff.lh_bot, :].copy()
                    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                    image = cv2.bitwise_not(image)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
                    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
                    staff.bars = self.find_bars(image)
            elif key == ord('h') or key == ord('?'):
                print("""
's'     Save changes.                      
'q'     Quit editor without saving.
'1'     Invalidate .pdf file by deleting it.
'2'     Rotates the image by 1 degree.
'n'     Next page of current score.
'p'     Previous page of current score.
'w'     Adds a staff below the selected one.
'x'     Deletes the selected staff.
'c'     Recomputes the bars of the slected staff.
'b'     Adds a bar to the selected staff.
'd'     Deletes the selected bar.
'i/I'   Moves selected staff up, fast & slow.
'j/J'   Moves selected staff down, fast & slow.
'e'     Extends the staff by lowering the right hand bottom.
'r'     Shrinks the staff by raising the right hand bottom.
'l'/'L' Moves selected bar left, fast & slow.
'j'/'J' Moves selected bar right, fast & slow.
'm'     Toggles on/off current page as reviewed.
Up      Selects staff above.
Down    Selects staff below.
Right   Selects staff below.
Left    Selects staff below.
Click   Adds a bar to the staff under the mouse click.
'h','?' This help.
                      """)
            else:
                print(f"Key: {key}")

    def is_validated(self) -> bool:
        # TODO This duplicates load_if_exists() in a nasty way.
        if self.json_path.exists():
            with open(self.json_path, "r") as fp:
                obj = json.load(fp)
            pages = [Staffer.Page.from_dict(s) for s in obj['pages']]
            assert obj['pdf_path'] == self.key, f"Expecting key {
                self.key} to match .json pdf path {obj['pdf_path']}."
            for page in pages:
                if not page.validated:
                    return False
            return True
        return False
