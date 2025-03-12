import logging
from pathlib import Path
from typing import Optional, cast

import click
import torch
from torch import utils
from torchinfo import summary

from config import Config
from dataset import Vocab
from model import Translator
from utils import DeviceType, compare_sequences, current_commit


class Client:

    config: Config
    name: str
    device: DeviceType
    model: Translator

    training_samples: int
    git_hash: str

    def __init__(
        self, config: Config,
        ckpt_path: Path,
        device: Optional[DeviceType] = None
    ):
        self.config = config
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        # Checks the git hash and loads the model.
        if self.config.git_hash != current_commit():
            logging.warning("Git hash mismatch: model may not load.")
        self.load(ckpt_path)

    def load(self, path: Path):
        self.model = Translator(self.config).to(self.device)
        ckpt = torch.load(path, weights_only=True)
        self.training_samples = ckpt["global_step"] * self.config.batch_size
        self.model.load_state_dict({
            key.replace("model.", ""): value
            for key, value in ckpt["state_dict"].items()
        })

    target_mask_cache: dict[int, torch.Tensor] = {}

    def get_target_mask(self, size: int) -> torch.Tensor:
        mask = self.target_mask_cache.get(size, None)
        if mask is None:
            mask = torch.triu(
                torch.ones(size, size), diagonal=1).to(torch.bool).to(self.device)
            self.target_mask_cache[size] = mask
        return mask

    def beam_decode(self, source: torch.Tensor) -> torch.Tensor:
        c, v = self.config, Vocab
        top_k = 3
        beam_width = 6

        self.model.eval()
        source_key_padding_mask = (source == v.PAD)[:, :, 0]
        memory = self.model.encode(
            source,
            src_key_padding_mask=source_key_padding_mask.to(self.device)
        )
        ys = torch.full(
            (c.spad_len, c.max_chord), fill_value=v.PAD
        ).to(self.device)
        ys[0, :] = v.SOS

        beams: list[tuple[torch.Tensor, float]] = [(ys, 0.0)]
        done:  list[tuple[torch.Tensor, float]] = []
        for idx in range(1, c.spad_len-1):
            candidates:  list[tuple[torch.Tensor, float]] = []

            for seq, score in beams:

                target_mask = self.get_target_mask(idx)
                out = self.model.decode(
                    seq[:idx, :].unsqueeze(0), memory, target_mask)
                out = out.transpose(0, 1)
                out = self.model.generator(out[:, -1])
                out = out.view(-1, c.max_chord, c.vocab_size)
                prob = torch.nn.functional.log_softmax(out, dim=-1)

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

        return max(done, key=lambda x: x[1])[0] if done else beams[0][0]

    def greedy_decode(self, source: torch.Tensor) -> torch.Tensor:
        c, v = self.config, Vocab
        self.model.eval()
        source_key_padding_mask = (source == v.PAD)[:, :, 0]
        memory = self.model.encode(
            source,
            src_key_padding_mask=source_key_padding_mask.to(self.device)
        )
        yhat = torch.full(
            (c.spad_len, c.max_chord), fill_value=v.PAD
        ).to(self.device)
        yhat[0, :] = v.SOS
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
            if token[0, 0] == v.EOS:
                break
        return yhat

    def full_decoder(self, source: torch.Tensor) -> torch.Tensor:
        c, v = self.config, Vocab
        # Let's call it 50 pixels / token; 25% overlap = 256 pixels ~ 5 tokens.
        chunk_size = int(3 * c.ipad_shape[1] / 4)   # 25% overlap
        context_window_size = 5
        width = source.size(1)

        yhat = torch.full(
            (2 * c.spad_len, c.max_chord), fill_value=v.PAD
        ).to(self.device)
        yhat[0, :] = v.SOS
        yhat_pos = 1
        context_window_start = 0  # Beginning of the context window size
        for offset in range(1 + width // chunk_size):
            start_offset = chunk_size * offset
            end_offset = min(start_offset + c.ipad_shape[1], width)
            print(f"stitch: {start_offset}:{end_offset} yhat_pos: {yhat_pos}")
            chunk = source[:, start_offset:end_offset, :]
            source_key_padding_mask = (chunk == v.PAD)[:, :, 0]
            memory = self.model.encode(
                chunk,
                src_key_padding_mask=source_key_padding_mask.to(self.device)
            )

            while yhat_pos < yhat.size(0) and yhat_pos - context_window_size < c.spad_len:
                target_mask = self.get_target_mask(
                    yhat_pos - context_window_start)
                out = self.model.decode(
                    yhat[context_window_start:yhat_pos, :].unsqueeze(0), memory, target_mask)
                out = out.transpose(0, 1)
                prob = self.model.generator(out[:, -1])
                prob = prob.view(-1, c.max_chord, c.vocab_size)
                token = torch.argmax(prob[-1:, :, :], dim=2)
                # Don't add EOS
                if token[0, 0] == v.EOS:
                    break
                yhat[yhat_pos, :] = token
                yhat_pos += 1

            context_window_start = yhat_pos - context_window_size

        # Do add the final EOS.
        yhat[yhat_pos, :] = v.EOS
        return yhat

    def predict(
        self, source: torch.Tensor, use_beam: bool = False, use_full: bool = False
    ) -> torch.Tensor:
        self.model.eval()
        if use_full:
            yhat = self.full_decoder(source.unsqueeze(0))
        elif use_beam:
            yhat = self.beam_decode(source.unsqueeze(0))
        else:
            yhat = self.greedy_decode(source.unsqueeze(0))
        return yhat


@click.command()
@click.argument("count", type=int, default=50)
@click.option("--use-beam", "use_beam", is_flag=True, default=False,
              help="Use beam decoding rather than greedy.")
@click.pass_context
def random_check(ctx, count: int, use_beam: bool):
    """Test model accuracy over randomly picked samples from the validation dataset.

    Args:
        count (int): Number of checks to run.
        do_display (bool): Displays the translations as we go.
    """
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    _, valid_ds = context.require_factory().datasets(valid_split=0.15)
    loader = utils.data.DataLoader(valid_ds, batch_size=1, shuffle=True)
    client = context.require_client()
    accuracies: list[float] = []
    with torch.no_grad():
        for image, seq in loader:
            image, seq = image.squeeze(0).to(
                client.device), seq.squeeze(0).to(client.device)
            yhat = client.predict(image, use_beam=use_beam)
            accuracy = compare_sequences(yhat, seq)
            accuracies.append(accuracy)
            print(f"{100.0 * accuracy:<8.2f}")
            count -= 1
            if count <= 0:
                return
    print(f"Average: {100.0 * sum(accuracies) / len(accuracies):2.2f}")


@click.command()
@click.argument("path", type=click.Path(file_okay=True))
@click.option("--accuracy/--do-accuracy", "do_accuracy", default=True,
              help="Computes accuracy against .tokens file.")
@click.option("--display/--no-display", "do_display", default=True,
              help="Displays / don't display the computed sequence.")
@click.option("--use-beam", "use_beam", is_flag=True, default=False,
              help="Use beam decoding rather than greedy.")
@click.option("--use-full", "use_full", is_flag=True, default=False,
              help="Enables decoding of images larger than padding length.")
@click.pass_context
def predict(
    ctx,
    path: Path, do_display: bool, do_accuracy: bool, use_beam: bool, use_full: bool
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
    client = context.require_client()

    path = Path(path)

    # Loads both the image and the sequence.
    source = factory.load_image(path).to(client.device)
    if do_accuracy:
        seq = factory.load_sequence(
            path.with_suffix(".tokens")).to(client.device)

    with torch.no_grad():
        yhat = client.predict(source, use_beam=use_beam, use_full=use_full)
        if do_accuracy:
            assert seq is not None, "load() failed to check target wasn't None."
            accuracy = compare_sequences(yhat, seq)
            print(f"Accuracy: {100.0 * accuracy:2.2f}")
        # if do_display:
        #     gp.display(yhat)


@click.command()
@click.option("--summary/--no-summary", "do_summary", default=True,
              help="Displays / don't display the full model summary.")
@click.option("--config/--no-config", "do_config", default=True,
              help="Displays / don't display the full model config.")
@click.pass_context
def infos(ctx, do_config: bool, do_summary: bool):
    """Display infos and summary about the model.

    Displays the model compatible git commit hash and the number
    of training tokens it's gone through.

    Args:
        do_config (bool): Also displays the config.
        do_summary (bool): Also displays the full model summary.
    """
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    client = context.require_client()
    print(f"git hash: {client.config.git_hash}")
    print(f"Training samples: {client.training_samples:,}")
    if do_config:
        print("Config:")
        for key, value in client.config.__dict__.items():
            print(f"\t{key:<20}: {value}")
    if do_summary:
        summary(client.model)
