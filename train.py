#!/usr/bin/env python3

import contextlib
from pathlib import Path
from typing import Literal, cast

import click
import cv2
import lightning as L
import torch
from pytorch_lightning.utilities.model_summary.model_summary import summarize
from torch import Tensor, nn, optim, utils

from model import Config, Translator
from sequence import compare_sequences, display_sequence
from vocab import Vocab


class LitTranslator(L.LightningModule):

    config: Config
    model: Translator
    loss_fn: nn.CrossEntropyLoss

    _decoding_method: Literal["greedy", "beam"] = "greedy"

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = Translator(config)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=Vocab.PAD)

    def _get_target_mask(self, size: int) -> Tensor:
        return torch.triu(
            torch.ones(size, size), diagonal=1
        ).to(torch.bool).to(self.device)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        X, y = batch

        y_input = y[:, :-1, :]
        y_expected = y[:, 1:, :]
        target_mask = self._get_target_mask(y_input.shape[1])

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
        target_mask = self._get_target_mask(y_input.shape[1])

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

    def _init_sequence(self, source: Tensor) -> tuple[Tensor, Tensor]:
        c, v = self.config, Vocab
        source_key_padding_mask = (source == v.PAD)[:, :, 0]
        memory = self.model.encode(
            source,
            src_key_padding_mask=source_key_padding_mask.to(self.device)
        )
        seq = torch.full(
            (c.spad_len, c.max_chord), fill_value=v.PAD
        ).to(self.device)
        seq[0, :] = v.SOS
        return memory, seq

    def _logits(self, c: Config, memory: Tensor, input: Tensor, idx: int) -> Tensor:
        target_mask = self._get_target_mask(idx)
        out = self.model.decode(
            input[:idx, :].unsqueeze(0), memory, target_mask)
        out = out.transpose(0, 1)
        out = self.model.generator(out[:, -1])
        out = out.view(-1, c.max_chord, c.vocab_size)
        return out

    def _greedy_decode(self, source: Tensor) -> Tensor:
        c, v = self.config, Vocab
        # memory, seq = self._init_sequence(source.unsqueeze(0)) FIXME
        memory, seq = self._init_sequence(source)
        for idx in range(1, c.spad_len-1):
            logits = self._logits(c, memory, seq, idx)
            token = torch.argmax(logits[-1:, :, :], dim=2)
            seq[idx, :] = token
            if token[0, 0] == v.EOS:
                break
        return seq

    def _beam_decode(self, source: Tensor) -> Tensor:
        c, v = self.config, Vocab
        memory, seq = self._init_sequence(source.unsqueeze(0))

        top_k = 3
        beam_width = 6
        beams: list[tuple[Tensor, float]] = [(seq, 0.0)]
        done:  list[tuple[Tensor, float]] = []
        for idx in range(1, c.spad_len-1):
            candidates:  list[tuple[Tensor, float]] = []

            for seq, score in beams:
                logits = self._logits(c, memory, seq, idx)
                prob = torch.nn.functional.log_softmax(logits, dim=-1)

                log_probs, tokens = torch.topk(prob[-1, :, :], top_k)
                for i in range(top_k):
                    candidate = seq.clone()
                    candidate[idx, :] = tokens[:, i]
                    log_prob = torch.sum(log_probs[:, i]).item()
                    if tokens[i, 0] == v.EOS:
                        done.append((candidate, score + log_prob))
                    else:
                        candidates.append(
                            (candidate, score + log_prob))
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)
            beams = beams[:beam_width]

            if len(done) >= beam_width:
                break

        seq = max(done, key=lambda x: x[1])[0] if done else beams[0][0]
        return seq.to(self.device)

    @contextlib.contextmanager
    def use(self, decoding_method: Literal["beam", "greedy"]):
        method = self._decoding_method
        self._decoding_method = decoding_method
        yield
        self._decoding_method = method

    def forward(self, image: Tensor) -> Tensor:
        """Forwards one image tensor, and decodes the model's output.

        The default decoding method is "greedy", but it can be switched 
        to "beam" with the use() method.

        Args:
            image (Tensor): Input image (height, width)

        Returns:
            Tensor: The decoded sequence of chords (spad_len, max_chords)
        """
        match self._decoding_method:
            case "beam":
                return self._beam_decode(image.unsqueeze(0))
            case "greedy":
                return self._greedy_decode(image.unsqueeze(0))

    def predict_step(self, source: Tensor) -> Tensor:
        match self._decoding_method:
            case "beam":
                return self._beam_decode(source)
            case "greedy":
                return self._greedy_decode(source)


@click.command()
@click.option("epochs", "-e", type=int, default=32)
@click.pass_context
def train(ctx, epochs: int):
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    factory = context.require_factory()
    config = factory.config

    model, trainer = context.require_trainer(
        max_epochs=epochs,
        limit_val_batches=10,
        log_every_n_steps=25,
    )

    # Prepares the train and valid loaders.
    train_ds, valid_ds = factory.datasets(valid_split=0.15)
    train_loader = utils.data.DataLoader(
        train_ds, num_workers=8, batch_size=config.batch_size, shuffle=True
    )
    valid_loader = utils.data.DataLoader(
        valid_ds, num_workers=8, batch_size=config.batch_size
    )

    ckpt_path = "last" if (context.model_directory /
                           "last.ckpt").exists() else None
    trainer.fit(
        model,
        train_loader, valid_loader,
        ckpt_path=ckpt_path
    )


@click.command()
@click.option(
    "--use-decoding", "-u",
    type=click.Choice(["greedy", "beam"], case_sensitive=False),
    default="greedy",
    help="Select decoding method: 'greedy' (default) or 'beam'."
)
@click.pass_context
def test(ctx, use_decoding: Literal["beam", "greedy"]):
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    factory = context.require_factory()

    _, valid_ds = factory.datasets(valid_split=0.15)
    loader = utils.data.DataLoader(valid_ds, batch_size=1, shuffle=True)

    model = context.require_model()

    for image, gt in loader:
        image, gt = (
            image.squeeze(0).to(model.device),
            gt.squeeze(0).to(model.device)
        )
        with model.use(use_decoding):
            yhat = model(image)
        print("\033[2J\033[H", end="")
        print(f"edist: {compare_sequences(yhat, gt)}")
        print(display_sequence(factory.vocab, yhat, gt))
        cv2.imshow("window", image.transpose(1, 0).cpu().numpy())
        if cv2.waitKey(0) == ord('q'):
            return


@click.command()
@click.argument("path", type=click.Path(file_okay=True))
@click.option("--accuracy/--do-accuracy", "do_accuracy", default=True,
              help="Computes accuracy against .tokens file.")
@click.option("--display/--no-display", "do_display", default=True,
              help="Displays / don't display the computed sequence.")
@click.option(
    "--use-decoding", "-u",
    type=click.Choice(["greedy", "beam"], case_sensitive=False),
    default="greedy",
    help="Select decoding method: 'greedy' (default) or 'beam'."
)
@click.pass_context
def predict(
    ctx, path: Path, do_display: bool, do_accuracy: bool, use_decoding: Literal["beam", "greedy"]
):
    """Translates the given PATH image file into kern-like notation.

    Args:
        path (Path): Path of the image to decode.

    Raises:
        FileNotFoundError: If the image PATH is not found or can't be loaded, e.g.
            because it's too wide; Or if accuracy is requested and a matching .tokens
            file couldn't be loaded.
    """
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    factory = context.require_factory()
    model = context.require_model()

    path = Path(path)

    # Loads both the image and the sequence.
    source = factory.load_image(path).to(model.device)
    if do_accuracy:
        seq = factory.load_sequence(
            path.with_suffix(".tokens")).to(model.device)

    with model.use(use_decoding):
        yhat = model(source)
    if do_accuracy:
        assert seq is not None, "load() failed to check target wasn't None."
        accuracy = compare_sequences(yhat.to("cuda"), seq)
        print(f"Accuracy: {100.0 * accuracy:2.2f}")
    if do_display:
        print(display_sequence(factory.vocab, yhat))


@click.command()
@click.option(
    "-v", "--verbose",
    count=True, default=1,
    help="Level of details, use -v, -vv, -vvv."
)
@click.pass_context
def summary(ctx, verbose: int):
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    model = context.require_model()
    match verbose:
        case 1:
            max_depth = 1
        case 2:
            max_depth = 2
        case 3:
            max_depth = -1
    print(summarize(model, max_depth=max_depth))    # type: ignore
