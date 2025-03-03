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
import torchvision
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
    no_box = -1     # Represents the absence of a box.
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

    def pick_one(self) -> tuple[torch.Tensor, torch.Tensor]:
        pages, boxes = self.batch(1, transform=False)
        return pages[0], boxes[0]

    def load(self, path: Path, transform=False) -> tuple[torch.Tensor, torch.Tensor]:
        with open(path, "rb") as fp:
            page, crops = cast(
                tuple[MatLike, list[BoundingBox]],
                pickle.load(fp)
            )
            height, width, _ = page.shape
            target_height, target_width = height // self.page_divider, width // self.page_divider
            page = cv2.resize(page, (target_width, target_height))
            crops = torch.stack([torch.Tensor((
                left / width,
                top / height,
                right / width,
                bottom / height
            )).to(torch.float32) for left, top, right, bottom in crops])
            if transform:
                return cast(tuple[torch.Tensor, torch.Tensor], self.transform(page, crops))
            else:
                return torch.from_numpy(page), crops

    @classmethod
    def transform(
        cls, page: MatLike, boxes: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pads the image.
        output = torch.full((cls.max_height, cls.max_width), 255)
        height, width, _ = page.shape
        output[:height, :width] = torch.from_numpy(
            cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        )
        output = output.to(torch.float32)

        # Scales the boxes accordingly.
        if boxes is not None:
            boxes[:, ::2] = width * boxes[:, ::2] / cls.max_width
            boxes[:, 1::2] = height * boxes[:, 1::2] / cls.max_height

        return (output - output.mean()) / output.std(), boxes

    def batch(
        self, batch_size=batch_size, transform: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        samples = [self.load(path, transform=transform)
                   for path in random.sample(self.data, batch_size)]

        def bounding_boxes(pages: list[torch.Tensor]) -> torch.Tensor:
            empty = torch.full(
                (batch_size, self.max_staves, 5), self.no_box
            ).to(torch.float32)
            for batchno, boxes in enumerate(pages):
                empty[batchno, :len(boxes), 0:4] = boxes
                empty[batchno, :len(boxes), 4] = 1.0
                empty[batchno:, len(boxes):, 4] = 0
            return empty

        return (
            torch.stack([page for page, _ in samples]),
            bounding_boxes([boxes for _, boxes in samples])
        )

    @classmethod
    def pred2bbox(
        cls, shape: torch.Size, preds: torch.Tensor, is_logits: bool = True
    ) -> list[PredictedBox]:
        bboxes = list()
        i = 0
        height, width = shape[0:2]
        for i in range(0, preds.shape[0]):
            confidence = torch.sigmoid(
                preds[i, 4]) if is_logits else preds[i, 4]
            bboxes.append(((
                int(preds[i, 0] * width),
                int(preds[i, 1] * height),
                int(preds[i, 2] * width),
                int(preds[i, 3] * height)),
                confidence
            ))
        return bboxes

    def stats(self):
        staves = 0
        max_width, max_height, max_staves = 0, 0, 0
        for path in self.data:
            page, crops = self.load(path)
            height, width, _ = page.shape
            max_width, max_height, max_staves = (
                max(width, max_width),
                max(height, max_height),
                max(len(crops), max_staves)
            )
            staves += len(crops)
        print(
            f"{max_height=}, {max_width=}, {max_staves=}, "
            f"avg_staves={staves / len(self.data):.2f}"
        )


class StafferModel(nn.Module):

    def __init__(self):
        super(StafferModel, self).__init__()
        self.net = nn.Sequential(
            # Layer 1 (BS, 1, H, W) -> (BS, 32, H/16, W/16)
            nn.Conv2d(1, 16, kernel_size=12, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 2
            nn.Conv2d(16, 32, kernel_size=6, stride=1, padding=1),
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
            nn.Linear(64, Dataset.max_staves * 5),
        )

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        x = self.gap(x)
        x = self.fc(x)
        # Apply sigmoid to the coordinates only.
        x = x.view((-1, 8, 5))
        x[:, :, :4] = torch.sigmoid(x[:, :, :4])
        return x.view((-1, 8, 5))


def draw(page: MatLike,
         boxes: torch.Tensor,
         true_boxes: Optional[torch.Tensor] = None,
         wait: bool = True) -> bool:
    print(f"{len(boxes)} bounding boxes:")
    for (left, top, right, bottom), confidence in Dataset.pred2bbox(page.shape, boxes):
        print(
            f"\t({left}, {top}), ({right}, {bottom}) @ {confidence:.2f}"
        )
        if confidence > 0.5:
            cv2.rectangle(page, (left, top),
                          (right, bottom), (255, 0, 0), 2)
    if true_boxes is not None:
        for (left, top, right, bottom), confidence in Dataset.pred2bbox(page.shape, true_boxes):
            if confidence > 0:
                cv2.rectangle(page, (left, top),
                              (right, bottom), (0, 255, 0), 2)
    if wait:
        cv2.imshow("predict", page)
        return cv2.waitKey(0) != ord('q')
    else:
        return True


def do_predict(model: nn.Module,
               page: MatLike,
               true_boxes: Optional[torch.Tensor] = None) -> bool:
    input, _ = Dataset.transform(page)

    yhat = model.forward(input.unsqueeze(0).unsqueeze(0))[0]
    return draw(page, yhat, true_boxes)


class Log:

    path: Path

    iou_losses: list[float]
    bce_losses: list[float]

    def __init__(self, path=LOG_PATH):
        self.path = path
        self.load()

    def load(self):
        if self.path.exists():
            with open(self.path, "r") as fp:
                obj = json.load(fp)
            self.iou_losses = obj["iou_losses"]
            self.bce_losses = obj["bce_losses"]
        else:
            self.bce_losses = list()
            self.iou_losses = list()

    def save(self):
        with open(self.path, "w+") as fp:
            json.dump({
                "iou_losses": self.iou_losses,
                "bce_losses": self.bce_losses
            }, fp, indent=4)

    def log(self, iou_loss: float, bce_loss: float):
        self.iou_losses.append(iou_loss)
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

    iou = torchvision.ops.distance_box_iou_loss
    bce = nn.BCEWithLogitsLoss()

    log = Log()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    batch_per_epochs = len(ds) // Dataset.batch_size

    count = 0
    count_per_report = 250
    last_report_count = 0

    # Mask to extract boxes 5-tuples
    bbox_mask = torch.full((Dataset.batch_size, 8, 5), True, dtype=torch.bool)
    bbox_mask[:, :, 4::5] = False

    start_time = time.time()
    for epoch in range(0, epochs):
        model.train()
        for batchno in range(0, batch_per_epochs):

            opt.zero_grad()
            pages, y = ds.batch()
            pages, y = pages.to(device), y.to(device)
            yhat = model.forward(pages.unsqueeze(1))

            # Extracts the confidence from the batch and the model output.
            y_prob, yhat_prob = y[:, :, 4::5], yhat[:, :, 4::5]
            y_bboxes, yhat_bboxes = y[bbox_mask], yhat[bbox_mask]

            mask = (y_bboxes != ds.no_box)
            y_bboxes, yhat_bboxes = y_bboxes[mask], yhat_bboxes[mask]

            iou_loss = iou(
                yhat_bboxes.view((-1, 4)),
                y_bboxes.view((-1, 4)),
                reduction="mean")
            bce_loss = bce(yhat_prob, y_prob)
            loss = iou_loss + bce_loss
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
                    f"iou_loss: {iou_loss.item():2.5f}, "
                    f"bce_loss: {bce_loss.item():2.5f}, "
                    f"loss: {loss.item():2.5f}"
                )

                # Tracks the losses.
                log.log(iou_loss.item(), bce_loss.item())
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
    model.eval()

    if image_path is not None:
        do_predict(model, cv2.imread(str(image_path)))
    else:
        ds = Dataset()
        while True:
            page, boxes = ds.pick_one()
            # draw(page.numpy(), boxes)
            if not do_predict(model, page.numpy(), true_boxes=boxes):
                break


@click.command()
def draw_batch():
    ds = Dataset()
    pages, boxes = ds.batch()
    for i in range(0, len(pages)):
        if not draw(pages[i].numpy(), boxes[i], wait=True):
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
    iou_plot, = ax.plot([], [], 'b', label='IOU loss.')
    bce_plot, = ax.plot([], [], 'r', label='BCE loss.')
    print("Press 'q' to quit.")

    while not quit:
        log.load()
        if smooth:
            iou_losses = moving_average(
                np.array(log.iou_losses, dtype=np.float32))
            bce_losses = moving_average(
                np.array(log.bce_losses, dtype=np.float32))
        else:
            iou_losses = log.iou_losses
            bce_losses = log.bce_losses
        iou_plot.set_xdata(range(0, len(iou_losses)))
        iou_plot.set_ydata(iou_losses)
        bce_plot.set_xdata(range(0, len(bce_losses)))
        bce_plot.set_ydata(bce_losses)
        ax.relim()
        ax.autoscale_view()
        ax.legend()
        fig.canvas.draw_idle()
        plt.pause(1)


@click.command()
def stats():
    ds = Dataset()
    ds.stats()


@click.group()
def cli():
    pass


cli.add_command(train)
cli.add_command(predict)
cli.add_command(draw_batch)
cli.add_command(plot)
cli.add_command(stats)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
