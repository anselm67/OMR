#!/usr/bin/env python3

import json
import logging
import os
import pickle
import random
import time
from pathlib import Path
from typing import Optional, cast

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from cv2.typing import MatLike
from numpy.typing import NDArray

STAVES_DATASET = Path("/home/anselm/Downloads/staves-dataset")
LOG_PATH = Path("untracked/log_extract_staves.json")
MODEL_PATH = Path("untracked/extract_staves.pt")

# (left, top, right, bottom)
BoundingBox = tuple[int, int, int, int]
PredictedBox = tuple[BoundingBox, float]


def to32(i: int) -> int:
    return 32 * ((i + 31) // 32)


class Dataset:
    page_divider = 2
    max_width = to32(1200 // page_divider)
    max_height = to32(1825 // page_divider)
    batch_size = 16
    max_staves = 8

    data: list[Path]

    def __init__(self, home: Path = STAVES_DATASET):
        self.data = list()
        for root, _, filenames in os.walk(home):
            for filename in filenames:
                file = Path(root) / filename
                if file.suffix == ".pkl":
                    self.data.append(file)

    def __len__(self) -> int:
        return len(self.data)

    def pick_one(self) -> MatLike:
        path = random.choice(self.data)
        tensor, _ = self.load(path)
        return tensor.numpy()

    def load(self, path: Path, transform=False) -> tuple[torch.Tensor, list[BoundingBox]]:
        with open(path, "rb") as fp:
            page, crops = cast(
                tuple[MatLike, list[BoundingBox]],
                pickle.load(fp)
            )
            height, width, _ = page.shape
            page = cv2.resize(
                page, (width // self.page_divider, height // self.page_divider))
            crops = [(
                left // self.page_divider,
                top // self.page_divider,
                right // self.page_divider,
                bottom // self.page_divider
            ) for left, top, right, bottom in crops]
            if transform:
                return self.transform(page), crops
            else:
                return torch.from_numpy(page), crops

    @classmethod
    def transform(cls, page: MatLike) -> torch.Tensor:
        output = torch.full((cls.max_height, cls.max_width), 255)
        height, width, _ = page.shape
        output[:height, :width] = torch.from_numpy(
            cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        )
        output = output.to(torch.float32)
        return (output - output.mean()) / output.std()

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        samples = [self.load(path, transform=True)
                   for path in random.sample(self.data, self.batch_size)]

        def bounding_boxes(pages: list[list[BoundingBox]]) -> torch.Tensor:
            empty = torch.full(
                (self.batch_size, 5 * self.max_staves), 0).to(torch.float32)
            for batchno, bboxes in enumerate(pages):
                x = 0
                for left, top, right, bottom in bboxes:
                    empty[batchno, x] = left / self.max_width
                    empty[batchno, x+1] = top / self.max_height
                    empty[batchno, x+2] = right / self.max_width
                    empty[batchno, x+3] = bottom / self.max_height
                    empty[batchno, x+4] = 1.0
                    x += 5
            return empty

        return (
            torch.stack([page for page, _ in samples]),
            bounding_boxes([page for _, page in samples])
        )

    @classmethod
    def pred2bbox(cls, preds: torch.Tensor, is_logits: bool = True) -> list[PredictedBox]:
        bboxes = list()
        i = 0
        while i < preds.shape[0]:
            confidence = torch.sigmoid(preds[i+4]) if is_logits else preds[i+4]
            bboxes.append(((
                int(preds[i+0] * cls.max_width),
                int(preds[i+1] * cls.max_height),
                int(preds[i+2] * cls.max_width),
                int(preds[i+3] * cls.max_height)),
                confidence
            ))
            i += 5
        return bboxes

    def stats(self):
        max_width, max_height, max_staves = 0, 0, 0
        for path in self.data:
            page, crops = self.load(path)
            height, width, _ = page.shape
            max_width, max_height, max_staves = (
                max(width, max_width),
                max(height, max_height),
                max(len(crops), max_staves)
            )
        print(f"{max_height=}, {max_width=}, {max_staves=}")


class StafferModel(nn.Module):

    def __init__(self):
        super(StafferModel, self).__init__()
        self.net = nn.Sequential(
            # Layer 1 (BS, 1, H, W) -> (BS, 32, H/16, W/16)
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 8 * 5),
        )

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        x = self.gap(x)
        x = self.fc(x)
        # Apply sigmoidto the coordinates only.
        x[:, :4] = torch.sigmoid(x[:, :4])
        return x


def draw(page: MatLike, boxes: list[PredictedBox], wait: bool = True) -> bool:
    print(f"{len(boxes)} bounding boxes:")
    for (left, top, right, bottom), confidence in boxes:
        print(
            f"\t({left}, {top}), ({right}, {bottom}) @ {confidence:.2f}"
        )
        cv2.rectangle(page, (left, top),
                      (right, bottom), (255, 0, 0), 2)
    if wait:
        cv2.imshow("predict", page)
        return cv2.waitKey(0) != ord('q')
    else:
        return True


def do_predict(model: nn.Module, page: MatLike):
    # Loads and prepares the image.
    h, w, _ = page.shape
    scale = Dataset.max_width / w
    page = cv2.resize(
        page, (Dataset.max_width, int(h * scale)), interpolation=cv2.INTER_AREA
    )

    input = Dataset.transform(page)

    yhat = model.forward(input.unsqueeze(0).unsqueeze(0))[0]
    boxes = Dataset.pred2bbox(yhat)
    draw(page, boxes)


class Log:

    path: Path

    mse_losses: list[float]
    bce_losses: list[float]

    def __init__(self, path=LOG_PATH):
        self.path = path
        self.load()

    def load(self):
        if self.path.exists():
            with open(self.path, "r") as fp:
                obj = json.load(fp)
            self.mse_losses = obj["mse_losses"]
            self.bce_losses = obj["bce_losses"]
        else:
            self.bce_losses = list()
            self.mse_losses = list()

    def save(self):
        with open(self.path, "w+") as fp:
            json.dump({
                "mse_losses": self.mse_losses,
                "bce_losses": self.bce_losses
            }, fp, indent=4)

    def log(self, mse_loss: float, bce_loss: float):
        self.mse_losses.append(mse_loss)
        self.bce_losses.append(bce_loss)
        self.save()


@click.command()
@click.option("path", "-o", type=click.Path(file_okay=True, dir_okay=False),
              default=MODEL_PATH)
@click.option("epochs", "-e", type=int, default=64)
def train(path: Path, epochs: int):
    ds = Dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    model = StafferModel().to(device)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    log = Log()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    batch_per_epochs = len(ds) // Dataset.batch_size

    count = 0
    count_per_report = 250
    last_report_count = 0

    # Mask to extract boxes 5-tuples
    bbox_mask = torch.full((Dataset.batch_size, 40), True, dtype=torch.bool)
    bbox_mask[:, 4::5] = False

    start_time = time.time()
    for epoch in range(0, epochs):
        model.train()
        for batchno in range(0, batch_per_epochs):

            opt.zero_grad()
            pages, y = ds.batch()
            pages, y = pages.to(device), y.to(device)
            yhat = model.forward(pages.unsqueeze(1))

            y_prob, yhat_prob = y[:, 4::5], yhat[:, 4::5]
            y_bboxes, yhat_bboxes = y[bbox_mask], yhat[bbox_mask]

            mse_loss = mse(y_bboxes, yhat_bboxes)
            bce_loss = bce(yhat_prob, y_prob)
            loss = 3 * mse_loss + bce_loss
            loss.backward()
            opt.step()

            count += Dataset.batch_size

            if count // count_per_report != last_report_count:
                last_report_count = count // count_per_report
                now = time.time()
                # Logs status
                logging.info(
                    f"Epoch: {epoch:2.2f} " +
                    f"batch {batchno}/{batch_per_epochs}, " +
                    f"{count} samples " +
                    f"in {(now - start_time):.2f}s " +
                    f"loss: {loss.item():2.5f}"
                )

                # Tracks the losses.
                log.log(mse_loss.item(), bce_loss.item())
                start_time = time.time()

        torch.save({
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "count": count,
        }, path)


@click.command()
@click.option("model_path", "-m",
              type=click.Path(file_okay=True, dir_okay=False, exists=True),
              default=MODEL_PATH)
@click.argument("image_path",
                type=click.Path(file_okay=True, dir_okay=False, exists=True),
                required=False)
def predict(model_path: Path, image_path: Optional[Path] = None):
    # Loads the model.
    obj = torch.load(model_path, weights_only=True)
    model = StafferModel()
    model.load_state_dict(obj["state_dict"])

    if image_path is not None:
        do_predict(model, cv2.imread(str(image_path)))
    else:
        ds = Dataset()
        while True:
            page = ds.pick_one()
            do_predict(model, page)


@click.command()
def draw_batch():
    ds = Dataset()
    pages, boxes = ds.batch()
    for i in range(0, len(pages)):
        page_boxes = ds.pred2bbox(boxes[i], is_logits=False)
        if not draw(pages[i].numpy(), page_boxes, wait=True):
            break


def moving_average(y: NDArray[np.float32], window_size: int = 10) -> NDArray[np.float32]:
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')


@click.command()
@click.argument("log_path",
                type=click.Path(file_okay=True, dir_okay=False, exists=True),
                default=LOG_PATH)
@click.option("--smooth/--no-smooth", default=True,
              help="Smooth the curves before plotting them.")
def plot(log_path: Path = LOG_PATH, smooth: bool = True):
    log = Log(Path(log_path))
    # State and function to quit the tracking loop.
    quit: bool = False

    def on_key(event):
        nonlocal quit
        quit = (event.key == 'q')

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_key)
    mse_plot, = ax.plot([], [], 'b', label='MSE loss.')
    bce_plot, = ax.plot([], [], 'r', label='BCE loss.')
    print("Press 'q' to quit.")

    while not quit:
        log.load()
        if smooth:
            mse_losses = moving_average(
                np.array(log.mse_losses, dtype=np.float32))
            bce_losses = moving_average(
                np.array(log.bce_losses, dtype=np.float32))
        else:
            mse_losses = log.mse_losses
            bce_losses = log.bce_losses
        mse_plot.set_xdata(range(0, len(mse_losses)))
        mse_plot.set_ydata(mse_losses)
        bce_plot.set_xdata(range(0, len(bce_losses)))
        bce_plot.set_ydata(bce_losses)
        ax.relim()
        ax.autoscale_view()
        ax.legend()
        fig.canvas.draw_idle()
        plt.pause(1)


@click.group()
def cli():
    pass


cli.add_command(train)
cli.add_command(predict)
cli.add_command(draw_batch)
cli.add_command(plot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
