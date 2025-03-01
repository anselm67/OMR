#!/usr/bin/env python3

import logging
import os
import pickle
import random
import time
from pathlib import Path
from typing import Optional, cast

import click
import cv2
import numpy as np
import torch
import torch.nn as nn
from cv2.typing import MatLike


def to32(i: int) -> int:
    return 32 * ((i + 31) // 32)


class Dataset:
    page_divider = 2
    max_width = to32(1200 // page_divider)
    max_height = to32(1825 // page_divider)
    batch_size = 16
    max_staves = 8

    data: list[Path]

    def __init__(self, home: Path):
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

    def load(self, path: Path, transform=False) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
        with open(path, "rb") as fp:
            page, crops = cast(
                tuple[MatLike, list[tuple[int, int, int, int]]],
                pickle.load(fp)
            )
            height, width, _ = page.shape
            page = cv2.resize(
                page, (width // self.page_divider, height // self.page_divider))
            if transform:
                return self.transform(page), crops
            else:
                return torch.from_numpy(page), crops

    @staticmethod
    def transform(page: MatLike) -> torch.Tensor:
        output = torch.full((Dataset.max_height, Dataset.max_width), 255)
        height, width, _ = page.shape
        output[:height, :width] = torch.from_numpy(
            cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        )
        output = output.to(torch.float32)
        return (output - output.mean()) / output.std()

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        samples = [self.load(path, transform=True)
                   for path in random.sample(self.data, self.batch_size)]

        def bounding_boxes(pages) -> torch.Tensor:
            empty = torch.full(
                (self.batch_size, 5 * self.max_staves), 0).to(torch.float32)
            for batchno, bboxes in enumerate(pages):
                x = 0
                for bbox in bboxes:
                    empty[batchno, x] = bbox[0] / self.max_width
                    empty[batchno, x+1] = bbox[1] / self.max_height
                    empty[batchno, x+2] = bbox[2] / self.max_width
                    empty[batchno, x+3] = bbox[3] / self.max_height
                    empty[batchno, x+4] = 1.0
                    x += 5
            return empty

        return (
            torch.stack([page for page, _ in samples]),
            bounding_boxes([page for _, page in samples])
        )

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
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 40))

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        x = self.gap(x)
        x = self.fc(x)
        # Apply sigmoidto the coordinates only.
        x[:, :4] = torch.sigmoid(x[:, :4])
        return x


def do_predict(model: nn.Module, page: MatLike):
    # Loads and prepares the image.
    h, w, _ = page.shape
    scale = Dataset.max_width / w
    page = cv2.resize(
        page, (Dataset.max_width, int(h * scale)), interpolation=cv2.INTER_AREA
    )

    input = Dataset.transform(page)

    yhat = model.forward(input.unsqueeze(0).unsqueeze(0))[0]
    bboxes = list()
    i = 0
    while i < yhat.shape[0]:
        bboxes.append((
            int(yhat[i] * Dataset.max_width),
            int(yhat[i+1] * Dataset.max_height),
            int(yhat[i+2] * Dataset.max_width),
            int(yhat[i+3] * Dataset.max_height),
            yhat[i+4]       # Confidence
        ))
        i += 5
    print(f"Found {len(bboxes)} bounding boxes:")
    for bbox in bboxes:
        print(
            f"\t({bbox[0]}, {bbox[1]}), ({bbox[2]}, {bbox[3]}) @ {bbox[5]:.2f}"
        )
        cv2.rectangle(page, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), (255, 0, 0), 2)

    # target_height = 990
    # height, width, _ = page.shape
    # scale = target_height / height
    # page = cv2.resize(
    #     page, (int(width * scale), int(height*scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow("predict", page)
    cv2.waitKey(0)


@click.command()
@click.option("path", "-o", type=click.Path(file_okay=True, dir_okay=False),
              default=Path("untracked/extract_staves.pt"))
@click.option("epochs", "-e", type=int, default=64)
def train(path: Path, epochs: int):
    ds = Dataset(Path("/home/anselm/Downloads/staves-dataset"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    model = StafferModel().to(device)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    opt = torch.optim.Adam(
        model.parameters(),
        lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )

    batch_per_epochs = len(ds) // Dataset.batch_size

    count = 0
    count_per_report = 500
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

            loss = mse(y_bboxes, yhat_bboxes) + bce(yhat_prob, y_prob)
            loss.backward()
            opt.step()

            count += Dataset.batch_size

            if count // count_per_report != last_report_count:
                last_report_count = count // count_per_report
                now = time.time()
                logging.info(
                    f"Epoch: {epoch:2.2f} " +
                    f"batch {batchno}/{batch_per_epochs}, " +
                    f"{count} samples " +
                    f"in {(now - start_time):.2f}s " +
                    f"loss: {loss.item():2.5f}"
                )
                start_time = time.time()

        torch.save({
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "count": count,
        }, path)


@click.command()
@click.option("model_path", "-m",
              type=click.Path(file_okay=True, dir_okay=False, exists=True),
              default=Path("untracked/extract_staves.pt"))
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
        ds = Dataset(Path("/home/anselm/Downloads/staves-dataset"))
        while True:
            page = ds.pick_one()
            do_predict(model, page)


@click.group()
def cli():
    pass


cli.add_command(train)
cli.add_command(predict)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
