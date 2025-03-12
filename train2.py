#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import Any, cast

import click
import cv2
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor, nn, optim, utils

from dataset import Factory, Vocab
from logger import SimpleLogger, plot
from model import Config, Translator
from utils import compare_sequences


class LitTranslator(L.LightningModule):

    config: Config
    model: Translator
    loss_fn: nn.CrossEntropyLoss

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

    def predict_step(self, source: Tensor) -> Tensor:
        c = self.config
        source = source.unsqueeze(0)
        source_key_padding_mask = (source == Vocab.PAD)[:, :, 0]
        memory = self.model.encode(
            source,
            src_key_padding_mask=source_key_padding_mask.to(self.device)
        )
        yhat = torch.full(
            (c.spad_len, c.max_chord), fill_value=Vocab.PAD
        ).to(self.device)
        yhat[0, :] = Vocab.SOS
        for idx in range(1, c.spad_len-1):
            target_mask = self.get_target_mask(idx)
            out = self.model.decode(
                yhat[:idx, :].unsqueeze(0), memory, target_mask)
            out = out.transpose(0, 1)
            prob = self.model.generator(out[:, -1])
            # As we're not runing through softmax this isn't really a probability,
            # still fine as we're only interested in argmax.
            prob = prob.view(-1, c.max_chord, c.vocab_size)
            token = torch.argmax(prob[-1:, :, :], dim=2)
            yhat[idx, :] = token
            if token[0, 0] == Vocab.EOS:
                break
        return yhat


@click.command()
@click.option("epochs", "-e", type=int, default=32)
def train(epochs: int):
    home = Path("/home/anselm/datasets/GrandPiano")
    root = Path("untracked/train")

    # Checks if we're resuming or starting from fresh.
    if root.exists() and (root / "config.json").exists():
        config = Config.load(root / "config.json")
        factory = Factory(home, config)
        ckpt_path = "last"
        logging.info("Resuming training from existing root.")
    else:
        root.mkdir(exist_ok=True, parents=True)
        config = Config()
        factory = Factory(home, config)
        ckpt_path = None
        config.save(root / "config.json")
        logging.info("Creating fresh training root.")

    # Prepares the train and valid loaders.
    train_ds, valid_ds = factory.datasets(valid_split=0.15)
    train_loader = utils.data.DataLoader(
        train_ds, num_workers=8, batch_size=config.batch_size, shuffle=True
    )
    valid_loader = utils.data.DataLoader(
        valid_ds, num_workers=8, batch_size=config.batch_size
    )

    # Prepares the trainer.
    translator = LitTranslator(config)
    trainer = L.Trainer(
        default_root_dir=root,
        max_epochs=epochs, limit_val_batches=10,
        logger=SimpleLogger(root / "train_logs.json", "train"),
        log_every_n_steps=25,
        callbacks=[
            ModelCheckpoint(dirpath=root, save_last=True)
        ]
    )
    trainer.fit(translator, train_loader, valid_loader, ckpt_path=ckpt_path)


def chord_repr(vocab: Vocab, chord: Tensor) -> str:
    # Otherwise, displays anything but PAD.
    if any([id != Vocab.PAD for id in chord]):
        texts = vocab.i2tok([
            int(id.item()) for id in chord if id != Vocab.SIL
        ])
        return " ".join([text for text in texts if text])
    else:
        return ""


def display(vocab: Vocab, yhat: Tensor, gt: Tensor):
    for chord_hat, chord_gt in zip(yhat, gt):
        # Skips SOS and EOS.
        if all([id.item() == Vocab.SOS for id in chord_hat]):
            continue
        # if all([id.item() == Vocab.EOS for id in chord]):
        #     return
        print(
            f"{chord_repr(vocab, chord_gt):<40}{chord_repr(vocab, chord_hat)}"
        )


@click.command()
def test(
    home: Path = Path("/home/anselm/datasets/GrandPiano"),
    root: Path = Path("untracked/train")
):
    config = Config.create(root / "config.json")
    factory = Factory(home, config)
    _, valid_ds = factory.datasets(valid_split=0.15)
    loader = utils.data.DataLoader(valid_ds)
    model = LitTranslator.load_from_checkpoint(
        root / "last.ckpt", config=config)
    trainer = L.Trainer(
        default_root_dir=root,
        logger=SimpleLogger(root / "predict_logs.json", "predict"),
    )

    for images, gts in loader:
        yhats = cast(list[Tensor], trainer.predict(model, images))
        for image, yhat, gt in zip(images.unbind(0), yhats, gts):
            print("\033[2J\033[H", end="")
            print(f"edist: {compare_sequences(yhat, gt)}")
            display(factory.vocab, yhat, gt)
            cv2.imshow("window", image.transpose(1, 0).cpu().numpy())
            if cv2.waitKey(0) == ord('q'):
                return


@click.group()
def cli():
    pass


cli.add_command(train)
cli.add_command(plot)
cli.add_command(test)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("high")
    cli()
