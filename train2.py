#!/usr/bin/env python3

import logging
from pathlib import Path

import click
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor, nn, optim, utils

from dataset import Factory, Vocab, init_dataset
from logger import SimpleLogger, plot
from model import Config, Translator


class Trainer2(L.LightningModule):

    config: Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = Translator(config)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=Vocab.PAD)

    target_mask_cache: dict[int, Tensor] = {}

    def get_target_mask(self, size: int) -> Tensor:
        mask = self.target_mask_cache.get(size, None)
        if mask is None:
            mask = torch.triu(
                torch.ones(size, size), diagonal=1).to(torch.bool).to(self.device)
            self.target_mask_cache[size] = mask
        return mask

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        X, y = batch

        y_input = y[:, :-1, :]
        y_expected = y[:, 1:, :]
        target_mask = self.get_target_mask(y_input.shape[1])

        logits = self.model(X, y_input, target_mask)

        loss = self.loss_fn(
            logits.reshape(-1, self.config.vocab_size),
            y_expected.flatten()
        )

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        X, y = batch

        y_input = y[:, :-1, :]
        y_expected = y[:, 1:, :]
        target_mask = self.get_target_mask(y_input.shape[1])

        logits = self.model(X, y_input, target_mask)

        loss = self.loss_fn(
            logits.reshape(-1, self.config.vocab_size),
            y_expected.flatten()
        )
        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=0.0001, betas=(0.9, 0.98), eps=1e-9
        )


@click.command()
@click.option("epochs", "-e", type=int, default=32)
def train(epochs: int):
    home = Path("/home/anselm/datasets/GrandPiano")
    root = Path("untracked/train")

    factory = Factory(home)
    config = factory.config
    config.save(root / "config.json")

    # Prepares the train and valid loaders.
    train_ds, valid_ds = factory.datasets(valid_split=0.15)
    train_loader = utils.data.DataLoader(
        train_ds, num_workers=8, batch_size=config.batch_size, shuffle=True
    )
    valid_loader = utils.data.DataLoader(
        valid_ds, num_workers=8, batch_size=config.batch_size
    )

    # Prepares the trainer.
    translator = Trainer2(config)
    trainer = L.Trainer(
        default_root_dir=root,
        max_epochs=epochs, limit_val_batches=10,
        logger=SimpleLogger(
            root / "train_logs.json",
            "default-model"
        ),
        log_every_n_steps=25,
        callbacks=[
            ModelCheckpoint(dirpath=root, save_last=True)
        ]
    )
    trainer.fit(translator, train_loader, valid_loader)


@click.group()
def cli():
    pass


cli.add_command(train)
cli.add_command(plot)
cli.add_command(init_dataset)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("high")
    cli()
