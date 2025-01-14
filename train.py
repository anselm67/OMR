
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from client import Model
from grandpiano import DatasetName, GrandPiano
from model import Config, Translator
from utils import DeviceType, current_commit, get_model_device


class Train(Model):

    outdir: Path
    gp: GrandPiano

    @staticmethod
    def get_train_log_path(outdir: Path, name: str) -> Path:
        return outdir / f"{name}_log.json"

    @property
    def epoch(self) -> float:
        return float(self.training_samples) / self.gp.len("train")

    @property
    def model_path(self) -> Path:
        return self.get_model_path(self.outdir, self.name)

    @property
    def optimizer_path(self) -> Path:
        return self.outdir / f"{self.name}_optimizer.pt"

    @property
    def log_path(self) -> Path:
        return self.get_train_log_path(self.outdir, self.name)

    def __init__(self, config: Config, dataset: GrandPiano, outdir: Path, name: str):
        super(Train, self).__init__(config, outdir, name, create=True)
        self.outdir = outdir
        self.gp = dataset

    def save(self):
        torch.save({
            "state_dict": self.model.state_dict(),
            "training_samples": self.training_samples,
            "git_hash": self.git_hash,
        }, self.model_path)

    def load_optimizer(self, device: DeviceType) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.0001, betas=(0.9, 0.98), eps=1e-9
        )
        if self.optimizer_path.exists():
            obj = torch.load(self.optimizer_path, weights_only=True)
            if obj["git_hash"] != self.git_hash:
                raise ValueError(f"Git hash for {self.model_path} and {
                                 self.optimizer_path} don't match.")
            optimizer.load_state_dict(obj["state_dict"])
        # Moves the optimizer to the requested device:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        return optimizer

    def save_optimizer(self, optimizer: torch.optim.Adam):
        torch.save({
            "state_dict": optimizer.state_dict(),
            "git_hash": self.git_hash
        }, self.optimizer_path)

    class Log:

        path: Path
        losses: List[float]
        valid_losses: List[float]

        def __init__(self, path: Path):
            self.path = path
            self.losses = list([])
            self.valid_losses = list([])
            self.load()

        def load(self):
            if self.path.exists():
                with open(self.path, "r") as fp:
                    obj = json.load(fp)
                self.losses = obj["losses"]
                self.valid_losses = obj["valid_losses"]

        def save(self):
            with open(self.path, "w+") as fp:
                json.dump({
                    "losses": self.losses,
                    "valid_losses": self.valid_losses,
                }, fp, indent=4)

        def log(self, loss: float, valid_loss: float):
            self.losses.append(loss)
            self.valid_losses.append(valid_loss)
            self.save()

    def load_log(self):
        return Train.Log(self.log_path)

    def load_batch(
        self,
        dataset_name: DatasetName,
        batch_size: int,
        device: DeviceType
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = []
        for _ in range(batch_size):
            samples.append(self.gp.next(dataset_name, pad=True, device=device))
        return (
            torch.stack([sample[0] for sample in samples]).to(device),
            torch.stack([sample[2] for sample in samples]).to(device),
        )

    def evaluate(
        self,
        loss_fn: nn.CrossEntropyLoss,
        batch_size: int,
        device: DeviceType,
        count: int = 5,
    ):
        self.model.eval()
        valid_losses = []

        for _ in range(count):
            X, y = self.load_batch("valid", batch_size, device)
            y_input = y[:, :-1, :]
            y_expected = y[:, 1:, :]

            target_mask = self.get_target_mask(y_input.shape[1], device=device)

            logits = self.model(X, y_input, target_mask)

            loss = loss_fn(
                logits.reshape(-1, self.gp.vocab_size),
                y_expected.flatten()
            )

            valid_losses.append(loss.item())
        self.model.train()
        return torch.tensor(valid_losses).mean().item()

    def train(
        self,
        epoch_count: int,
        batch_size: int,
        device: Optional[DeviceType] = None
    ) -> int:
        """Train the model for an addition count of epochs

        Args:
            epoch_count (int): Number of additional epochs to train the model for.

            device (DeviceType or None): Device to train on, if None best option is chosen.
        Returns:
            int: Total number of samples we've run training for, includes all
                previous training runs.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        opt = self.load_optimizer(device)
        log = self.load_log()

        loss_fn = nn.CrossEntropyLoss(ignore_index=GrandPiano.PAD[0])
        batch_per_epoch = self.gp.len("train") // batch_size
        start_time = time.time()

        tokens_per_report = 150.0
        report_ticks = 0

        for _ in range(epoch_count):

            for b in range(batch_per_epoch):
                X, y = self.load_batch("train", batch_size, device)
                # Shifts the Ys for training.
                y_input = y[:, :-1, :]
                y_expected = y[:, 1:, :]

                target_mask = self.get_target_mask(
                    y_input.shape[1], device=device)

                opt.zero_grad()
                logits = self.model(X, y_input, target_mask)
                loss = loss_fn(
                    logits.reshape(-1, self.gp.vocab_size),
                    y_expected.flatten()
                )
                loss.backward()
                opt.step()

                self.training_samples += batch_size

                if int(self.training_samples / tokens_per_report) != report_ticks:
                    report_ticks = int(
                        self.training_samples / tokens_per_report)
                    now = time.time()
                    valid_loss = self.evaluate(loss_fn, batch_size, device)
                    print(
                        f"Epoch {self.epoch:2.2f} " +
                        f"batch {b}/{batch_per_epoch:,}, " +
                        f"{self.training_samples / 1000:,.2f}k samples" +
                        f" in {(now-start_time):.2f}s " +
                        f"loss: {loss.item():2.2f}, vloss: {valid_loss:2.2f}"
                    )
                    log.log(loss=loss.item(), valid_loss=valid_loss)
                    start_time = time.time()

                    if report_ticks % 10 == 0:
                        # Checkpoints the model and the optimizer state.
                        self.save()
                        self.save_optimizer(opt)
            print(f"Epoch {_} fnished.")
        return self.training_samples


@click.command()
@click.argument("epoch_count", type=int, required=True)
@click.option("-batch-size", "-b", "batch_size", type=int, default=16)
@click.pass_context
def train(ctx, epoch_count: int, batch_size: int):
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    context.require_train().train(epoch_count, batch_size, device)
