import logging
from dataclasses import replace
from typing import Any, Callable, List, Tuple

import cv2
import numpy as np
from cv2.typing import MatLike

from kern.kern_reader import KernReader
from staffer import Staffer


class Action:
    key_code: int
    func: Callable[[], None]
    help: str

    def __init__(self, key: int | str, func: Callable[[], None], help: str):
        self.key_code = key if type(key) is int else ord(str(key))
        self.func = func
        self.help = help


class StaffEditor:
    STAFFER_WINDOW = "StaffEditor"

    staffer: Staffer

    # Data being edited.
    data: list[tuple[MatLike, Staffer.Page]]
    kern: KernReader

    # Editor's config
    max_height: int
    actions: dict[int, Action]
    fast_mode: bool

    # Current state of the editor.
    position: int = 0
    staff_position: int = 0
    bar_position: int = 0
    bar_offset: int = 0
    scale_ratio: float = 1.0

    @property
    def image(self):
        assert 0 <= self.position < len(self.data)
        return self.data[self.position][0]

    @property
    def page(self):
        assert 0 <= self.position < len(self.data)
        return self.data[self.position][1]

    def replace_page(self, **kwargs):
        self.data[self.position] = (self.image, replace(self.page, **kwargs))

    @property
    def staff(self):
        assert 0 <= self.staff_position < len(self.page.staves)
        return self.page.staves[self.staff_position]

    def replace_staff(self, **kwargs):
        self.page.staves[self.staff_position] = replace(self.staff, **kwargs)

    def move_bar(self, delta: int):
        bars = self.staff.bars.copy()
        assert 0 <= self.bar_position < len(bars)
        bars[self.bar_position] += delta
        self.replace_staff(bars=bars)

    @property
    def bar_number(self) -> int:
        bar_number = self.bar_offset
        for i in range(0, self.staff_position):
            bar_number += len(self.page.staves[i].bars) - 1
        return bar_number + self.bar_position

    def __init__(self, staffer: Staffer, max_height: int = 992):
        self.staffer = staffer
        self.data = list(staffer.staff())
        self.kern = KernReader(self.staffer.kern_path)
        self.max_height = max_height
        self.position = 0
        self.staff_position = 0
        self.bar_position = 0
        self.fast_mode = False
        if len(self.page.staves) <= 0:
            self.staff_position = -1
            self.bar_position = -1
        elif len(self.page.staves[0].bars) <= 0:
            self.bar_position = -1
        self.update_bar_offset()
        cv2.namedWindow(self.STAFFER_WINDOW)
        self.init_commands()

    def draw_page(
        self, image: MatLike, page: Staffer.Page, bar_offset: int,
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

    def get_bar_offset(self, position: int = -1) -> int:
        if position < 0:
            position = len(self.data)
        bar_number = 0
        for _, page in self.data[:position]:
            for staff in page.staves:
                bar_number += len(staff.bars) - 1
        if not self.kern.has_bar_zero():
            bar_number += 1
        return bar_number

    def update_bar_offset(self):
        self.bar_offset = self.get_bar_offset(self.position)

    def get(self) -> Tuple[MatLike, 'Staffer.Page']:
        return self.image, self.page

    def next(self):
        if self.position + 1 >= len(self.data):
            print("End of score.")
            return
        if self.fast_mode:
            self.replace_page(validated=True)
            self.save()
        self.position += 1
        self.staff_position = 0
        if len(self.page.staves) == 0:
            self.staff_position = -1
        elif len(self.page.staves[0].bars) == 0:
            self.bar_position = -1
        else:
            self.bar_position = 0

    def prev(self, select_last: bool = False):
        if self.position - 1 < 0:
            print("Beginning of score.")
            return
        self.position -= 1
        staff_count = len(self.page.staves)
        if staff_count == 0:
            self.staff_position = -1
        elif select_last:
            self.staff_position = staff_count - 1
        else:
            self.staff_position = 0
        bars_len = len(self.page.staves[self.staff_position].bars)
        if self.staff_position < 0 or bars_len == 0:
            self.bar_position = -1
        else:
            self.bar_position = bars_len - 1 if select_last else 0

    def select_prev_staff(self, select_last: bool = False):
        if self.staff_position - 1 < 0:
            self.prev(select_last=True)
        else:
            self.staff_position = self.staff_position - 1
            self.bar_position = 0
            bar_count = len(self.page.staves[self.staff_position].bars)
            if bar_count <= 0:
                self.bar_position = -1
            elif select_last:
                self.bar_position = bar_count - 1

    def select_next_staff(self):
        if self.staff_position + 1 >= len(self.page.staves):
            self.next()
        else:
            self.staff_position = self.staff_position + 1
            self.bar_position = 0
            if len(self.page.staves[self.staff_position].bars) <= 0:
                self.bar_position = -1

    def select_next_bar(self):
        bar_count = len(self.page.staves[self.staff_position].bars)
        if self.bar_position + 1 >= bar_count:
            self.select_next_staff()
        else:
            self.bar_position += 1

    def select_prev_bar(self):
        if self.bar_position - 1 < 0:
            self.select_prev_staff(select_last=True)
        else:
            self.bar_position -= 1

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
        if self.staff_position < 0:
            ypos = 100
            self.page.staves.append(Staffer.Staff(
                rh_top=ypos,
                lh_bot=ypos + height,
                bars=list()
            ))
            self.staff_position = len(self.page.staves) - 1
        else:
            ypos = self.page.staves[self.staff_position].lh_bot + 10
            self.page.staves.insert(
                self.staff_position + 1,
                Staffer.Staff(
                    rh_top=ypos,
                    lh_bot=ypos + height,
                    bars=list()
                )
            )
            self.staff_position += 1

    def add_bar(self, offset: int = -1):
        bars = self.staff.bars.copy()
        if offset < 0:
            # Adds a bar after the selected one.
            if self.bar_position < 0:
                offset = 10
                bars.append(offset)
            else:
                offset = self.staff.bars[self.bar_position] + 10
                bars.insert(self.bar_position + 1, offset)
        else:
            bars.append(offset)
        bars = sorted(bars)
        self.replace_staff(bars=bars)
        self.bar_position = bars.index(offset)
        self.check_bar_count()

    def delete_selected_bar(self):
        if self.bar_position < 0:
            return
        del self.page.staves[self.staff_position].bars[self.bar_position]
        self.bar_position = max(0, self.bar_position - 1)
        self.check_bar_count()

    def delete_selected_staff(self):
        if self.staff_position < 0:
            return
        del self.page.staves[self.staff_position]
        if len(self.page.staves) == 0:
            self.staff_position = -1
            self.bar_position = -1
        else:
            self.staff_position = max(0, self.staff_position - 1)
            self.bar_position = 0

    def check_bar_count(self):
        bar_count = self.get_bar_offset()
        if not self.kern.has_bar_zero():
            bar_count -= 1
        if bar_count == self.kern.bar_count:
            self.beep()
            print("Bar count matches, victory !")
        else:
            print("Bar count mismtach, more work!")

    def update_ui(self):
        image = self.draw_page(
            self.image, self.page, self.bar_offset, self.staff_position, self.bar_position
        )
        height, width = image.shape[:2]
        if self.max_height > 0 and height > self.max_height:
            self.scale_ratio = height / self.max_height
            new_width = int(self.max_height * width / height)
            image = cv2.resize(image, (new_width, self.max_height))
        else:
            self.scale_ratio = 1.0
        cv2.imshow(self.STAFFER_WINDOW, image)

    def edit(self, fast_mode: bool = False) -> bool:
        """Edits the staff.

        Args:
            max_height (int, optional): The requested editor height. Defaults to 992.

        Returns:
            bool: True if the user wishes to continue editing further documents,
                False otherwise.
        """

        self.fast_mode = fast_mode

        while True:

            self.update_bar_offset()
            self.update_ui()

            key = cv2.waitKey()

            if self.run_command(key):
                continue

            if key == ord('q'):         # Quits editing.
                return False
            elif key == ord('n'):   # Moves onto the next document if any.
                if self.fast_mode:
                    self.replace_page(validated=True)
                    self.save()
                return True
            elif key == ord('1'):
                self.staffer.unlink_pdf()
                print(f"{self.staffer.key} cleaned-up.")
                return True
            else:
                print(f"Unknown key: '{key}', press 'h' for help.")

    def save(self):
        self.staffer.save(tuple(page for _, page in self.data))
        print(f"{len(self.data)} pages reviewed and saved.")

    KEY_NAMES = {
        81: "Left",
        82: "Up",
        83: "Right",
        84: "Down",
    }

    def help(self):
        def key_name(key_code: int) -> str:
            if (name := self.KEY_NAMES.get(key_code, None)):
                return name
            elif 32 <= key_code < 127:
                return f"'{chr(key_code)}'"
            else:
                return "???"

        for key_code, action in self.actions.items():
            print(f"{key_name(key_code):<8}{action.help}")

    def clear(self):
        # Clears the terminal and displays the kern tokens:
        print('\033[2J', end='')
        print('\033[H', end='')

    def beep(self):
        print("\a", end="", flush=True)

    def print_kerns(self):
        self.clear()
        bar_number = self.bar_number
        records = self.kern.get_text(bar_number)
        if records is None:
            print(f"No records found for bar {bar_number}")
        else:
            print(f"Bar {bar_number} / {self.kern.bar_count}:")
            for record in records:
                print(record)

    def title(self):
        self.clear()
        print("Fast move on!" if self.fast_mode else "Fast mode off.")
        print(f"Header for {
            self.staffer.kern_path} - {self.kern.bar_count} bars")
        for line in self.kern.header():
            print(line)

    def recompute_bars(self):
        if self.staff_position >= 0:
            staff = self.page.staves[self.staff_position]
            image = self.image[staff.rh_top:staff.lh_bot, :].copy()
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            image = cv2.bitwise_not(image)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            self.replace_staff(bars=self.staffer.find_bars(image))

    def toggle_fast_mode(self):
        self.fast_mode = not self.fast_mode
        print("Fast move on!" if self.fast_mode else "Fast mode off.")

    def register_actions(self, *actions: Action):
        for a in actions:
            assert self.actions.get(a.key_code) is None, \
                f"Action for {a.key_code} already defined."
            self.actions[a.key_code] = a

    def init_commands(self):
        self.actions = {}
        self.register_actions(
            Action('s', self.save,
                   "Saves works to disk."),
            Action('a', self.add_bar,
                   "Adds a bar after the one currently selected."),
            Action('w', self.add_staff,
                   "adds a pair of staves after the one currently selected."),
            Action('x', self.delete_selected_staff,
                   "Deletes the staff currently selected."),
            Action('d', self.delete_selected_bar,
                   "Deletes the bar currently selected."),
            Action(' ', self.next,
                   "Moves to next page."),
            Action('p', self.prev,
                   "Moves to previous page."),

            Action('j', lambda: self.move_bar(-2),
                   "Moves the selected bar left."),
            Action('l', lambda: self.move_bar(2),
                   "Moves the selected bar right."),
            Action('e', lambda: self.replace_staff(lh_bot=self.staff.lh_bot + 1),
                   "Extends the selected staff down."),
            Action('r', lambda: self.replace_staff(lh_bot=self.staff.lh_bot - 1),
                   "Shrinks the selected staff up."),
            Action('i', lambda: self.replace_staff(
                lh_bot=self.staff.lh_bot - 2,
                rh_top=self.staff.rh_top - 2,
            ), "Moves the selected staff up."),
            Action('m', lambda: self.replace_staff(
                lh_bot=self.staff.lh_bot + 2,
                rh_top=self.staff.rh_top + 2,
            ), "Moves the selected staff down."),

            Action('c', self.recompute_bars,
                   "Recomputes the bars within the selected staves."),

            Action('v', lambda: self.replace_page(validated=not self.page.validated),
                   "Toggles the current page validation flag on or off."),

            Action(81, self.select_prev_bar,
                   "Moves to and selects previous bar."),
            Action(83, self.select_next_bar,
                   "Moves to and selects next bar."),
            Action(84, self.select_next_staff,
                   "Moves to and selects next pair of staves."),
            Action(82, self.select_prev_staff,
                   "Moves to and selects previous pairs of staves."),
            Action('h', self.help,
                   "Displays this help text."),
            Action('?', self.help,
                   "Displays this help text."),


            Action('k', self.print_kerns,
                   "Prints the kern tokens for the selected bar."),
            Action('t', self.title,
                   "Prints misc. infos about the score being edited."),

            Action('f', self.toggle_fast_mode,
                   "Toggles fast mode: automatic save and valid on page jumps."),

            Action('/', self.check_bar_count,
                   "Checks bar count")
        )

    def run_command(self, key_code) -> bool:
        action = self.actions.get(key_code, None)
        if action:
            try:
                action.func()
            except Exception as e:
                logging.exception(f"Command {chr(key_code)} failed:\n{e}", e)
            return True
        return False
