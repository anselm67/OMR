from pathlib import Path
from typing import Dict, Optional, cast

import click
import torch
from torchinfo import summary

from grandpiano import GrandPiano
from model import Config, Translator
from utils import DeviceType, current_commit, get_model_device


class Model:

    config: Config
    name: str
    model: Translator

    training_samples: int
    git_hash: str

    @staticmethod
    def get_model_path(outdir: Path, name: str) -> Path:
        return Path(outdir) / f"{name}_model.pt"

    def __init__(self, config: Config, outdir: Path, name: str, create: bool = False):
        self.config = config
        self.name = name
        self.load(self.get_model_path(outdir, name), create)

    def load(self, path: Path, create: bool = False):
        self.model = Translator(self.config)
        if path.exists():
            obj = torch.load(path, weights_only=True)
            self.model.load_state_dict(obj["state_dict"])
            self.training_samples = obj["training_samples"]
            self.git_hash = obj["git_hash"]
        elif create:
            self.training_samples = 0
            self.git_hash = current_commit()
        else:
            raise FileNotFoundError(
                f"Model file {path} and not found."
            )

    target_mask_cache: Dict[int, torch.Tensor] = {}

    def get_target_mask(self, size: int, device: Optional[DeviceType]) -> torch.Tensor:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        mask = self.target_mask_cache.get(size, None)
        if mask is None:
            mask = torch.triu(
                torch.ones(size, size), diagonal=1).to(torch.bool).to(device)
            self.target_mask_cache[size] = mask
        return mask

    def greedy_decode(self, gp: GrandPiano, source: torch.Tensor) -> torch.Tensor:
        device = get_model_device(self.model) or "cpu"
        source_key_padding_mask = (source == GrandPiano.PAD[0])[:, :, 0]
        memory = self.model.encode(
            source,
            src_key_padding_mask=source_key_padding_mask
        ).to(device)
        ys = torch.full(
            (1, gp.Stats.max_chord),
            fill_value=GrandPiano.SOS[0]
        ).to(device)
        for i in range(gp.spad_len-1):
            target_mask = self.get_target_mask(ys.shape[0], device=device)
            out = self.model.decode(ys.unsqueeze(0), memory, target_mask)
            out = out.transpose(0, 1)
            prob = self.model.generator(out[:, -1])
            prob = prob.view(-1, gp.Stats.max_chord, gp.vocab_size)
            token = torch.argmax(prob[-1:, :, :], dim=2)
            ys = torch.cat([ys, token], dim=0)
            if token[0, 0] == gp.EOS[0]:
                break
        return ys

    def predict(self, gp: GrandPiano, path: Path):
        r"""Translates the given PATH image file into kern-ike notation.
        """
        source, _ = gp.load_image(path, pad=True)
        if source is None:
            raise FileNotFoundError(f"File {path} not found, likely too wide.")

        chords = self.greedy_decode(gp, source.unsqueeze(0))
        for chord in chords:
            texts = gp.decode([
                int(id.item()) for id in chord if id != GrandPiano.SIL[0]
            ])
            print("\t".join(texts))


@click.command()
@click.argument("path", type=click.Path(file_okay=True))
@click.pass_context
def predict(ctx, path: Path):
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    """Translates the given PATH image file into kern-ike notation.
    """
    context.require_client().predict(context.require_dataset(), path)


@click.command()
@click.option("--summary/--no-summary", "do_summary", default=True,
              help="Displays / don't display  the full model summary.")
@click.option("--config/--no-config", "do_config", default=True,
              help="Displays / don't display the full model config.")
@click.pass_context
def infos(ctx, do_config: bool, do_summary: bool):
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    """Display infos and summary about the model.
    """
    client = context.require_client()
    print(f"git hash: {client.git_hash}")
    print(f"Training samples: {client.training_samples:,}")
    if do_config:
        print("Config:")
        for key, value in client.config.__dict__.items():
            print(f"\t{key:<20}: {value}")
    if do_summary:
        summary(client.model)
