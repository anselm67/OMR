#!/usr/bin/env python3

import os
import pickle
import sys
from pathlib import Path
from typing import List, Tuple, cast

from cv2.typing import MatLike
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from staffer import Page, find_staff


class KernSheetEditor(QWidget):

    kern_files: List[Path] = list([])
    position: int = 0
    text_label: QTextEdit

    def __init__(self, datadir: Path):
        super().__init__()
        self.datadir = Path(datadir)
        self.list()
        self.create_widgets()

    def list(self):
        for root, _, filenames in os.walk(self.datadir):
            for filename in filenames:
                file = Path(root) / filename
                if file.suffix == ".numpy":
                    self.kern_files.append(file.with_suffix(""))

    def load_image(self, numpy_path, pageno: int = 0) -> Tuple[Page, QImage]:
        with open(numpy_path.with_suffix(".numpy"), "rb") as f:
            image = cast(List[MatLike], pickle.load(f))[pageno]
            height, width = image.shape
            return find_staff(image, do_plot=False), QImage(
                image.data, width, height,
                width,
                QImage.Format.Format_Grayscale8
            )

    def create_widgets(self):
        self.setWindowTitle("KernSheet Editor")
        # Create a QLabel to display the image
        self.image_label = QLabel()
        self.image_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        self.text_label = QTextEdit()  # Set initial text
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setReadOnly(True)
        self.text_label.setMinimumHeight(200)  # set a minimum height
        self.text_label.setMaximumHeight(400)  # set a maximum height
        self.text_label.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Maximum
        )

        # Loads thefirst image here, otherwise the widgets don't size properly
        self.cycle()

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(False)
        scroll_area.setWidget(self.image_label)
        scroll_area.setMaximumHeight(400)

        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(scroll_area)
        layout.addWidget(self.text_label)
        self.setLayout(layout)

        self.button = QPushButton("Change Image")
        self.button.clicked.connect(self.cycle)

        layout.addWidget(self.button)

        self.resize(1250, 800)  # set window size

    def render_staff(self, page: Page, pixmap: QPixmap):
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor("red"), 2, Qt.PenStyle.SolidLine))
        for staff in page.staves:
            for bar in staff.bars:
                painter.drawLine(bar, staff.rh_top, bar, staff.lh_bot)
            painter.drawLine(
                staff.bars[0], staff.rh_top, staff.bars[-1], staff.rh_top)
            painter.drawLine(
                staff.bars[0], staff.lh_bot, staff.bars[-1], staff.lh_bot)

    def render_score(self, path: Path) -> bool:
        print(f"render_score {path}")
        # Load the image
        page, image = self.load_image(path)
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():  # Handle invalid image paths
            print(f"Error: Could not load image at {path}")
            return False
        self.render_staff(page, pixmap)
        self.image_label.setPixmap(pixmap)

        kern_text = ""
        with open(path.with_suffix(".krn"), "r") as fp:
            kern_text = fp.readlines()
        self.text_label.setText(
            "<div align='left'>" +
            "".join([f"<p>{line}</p>" for line in kern_text]) +
            "</div>"
        )
        return True

    def cycle(self):
        self.position = (self.position + 1) % len(self.kern_files)
        self.render_score(self.kern_files[self.position])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    datadir = Path("/home/anselm/datasets/KernSheet")
    window = KernSheetEditor(datadir)
    window.render_score(datadir / "chopin/mazurka/mazurka68-1")
    window.show()

    app.exec()
