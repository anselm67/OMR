from pathlib import Path
from typing import Dict, List, Optional, cast

import click
import torch
from torchinfo import summary

from grandpiano import GrandPiano
from model import Config, Translator
from utils import DeviceType, current_commit


class Model:

    config: Config
    name: str
    device: DeviceType
    model: Translator

    training_samples: int
    git_hash: str

    @staticmethod
    def get_model_path(outdir: Path, name: str) -> Path:
        return Path(outdir) / f"{name}_model.pt"

    def __init__(
        self, config: Config, outdir: Path, name: str,
        device: Optional[DeviceType] = None, create: bool = False
    ):
        self.config = config
        self.name = name
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.load(self.get_model_path(outdir, name), create)

    def load(self, path: Path, create: bool = False):
        self.model = Translator(self.config).to(self.device)
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

    def get_target_mask(self, size: int) -> torch.Tensor:
        mask = self.target_mask_cache.get(size, None)
        if mask is None:
            mask = torch.triu(
                torch.ones(size, size), diagonal=1).to(torch.bool).to(self.device)
            self.target_mask_cache[size] = mask
        return mask

    def greedy_decode(self, gp: GrandPiano, source: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        source_key_padding_mask = (source == GrandPiano.PAD[0])[:, :, 0]
        memory = self.model.encode(
            source,
            src_key_padding_mask=source_key_padding_mask.to(self.device)
        )
        ys = torch.full(
            (gp.spad_len, gp.Stats.max_chord), fill_value=GrandPiano.PAD[0]
        ).to(self.device)
        ys[0, :] = GrandPiano.SOS[0]
        for idx in range(1, gp.spad_len-1):
            target_mask = self.get_target_mask(idx)
            out = self.model.decode(
                ys[:idx, :].unsqueeze(0), memory, target_mask)
            out = out.transpose(0, 1)
            prob = self.model.generator(out[:, -1])
            prob = prob.view(-1, gp.Stats.max_chord, gp.vocab_size)
            token = torch.argmax(prob[-1:, :, :], dim=2)
            ys[idx, :] = token
            if token[0, 0] == gp.EOS[0]:
                break
        return ys

    def predict(
        self, gp: GrandPiano,
        source: torch.Tensor,
        target: Optional[torch.Tensor],
        do_display: bool = True
    ) -> float:
        """Translated the image given by PATH to kern-like tokens.

        Args:
            gp (GrandPiano): The dataset, to decode the tokens to string.
            source (torch.Tensor): Image to decode.
            target (torch.Tensor, optional): Compute and return accuracy against this 
                given target. Defaults to None.
            do_display (bool, optional): Displays the computed tokens. Defaults to True.
        """
        self.model.eval()
        if source is None:
            raise FileNotFoundError(f"File {path} not found, likely too wide.")

        chords = self.greedy_decode(gp, source.unsqueeze(0))
        accuracy = 0.0
        if target is not None:
            wrong = torch.sum(target != chords).item()
            total = torch.sum(target != gp.PAD[0]).item()
            accuracy = 1.0 - wrong / total
        if do_display:
            for chord in chords:
                if any([id != GrandPiano.PAD[0] for id in chord]):
                    texts = gp.decode([
                        int(id.item()) for id in chord if id != GrandPiano.SIL[0]
                    ])
                    print("\t".join([text for text in texts if text]))
        return accuracy


@click.command()
@click.argument("count", type=int)
@click.option("--display/--no-display", "do_display", default=True,
              help="Displays / don't display the computed sequence.")
@click.pass_context
def random_check(ctx, count: int, do_display: bool):
    """Test model accuracy over randomly picked samples from the validation dataset.

    Args:
        count (int): Number of checks to run.
        do_display (bool): Displays the translations as we go.
    """
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    accuracies: List[float] = []
    with torch.no_grad():
        for _ in range(count):
            path, image, _, sequence, _ = context.require_dataset().next("valid", pad=True)
            accuracy = context.require_client().predict(
                context.require_dataset(), image, sequence, do_display=do_display)
            accuracies.append(accuracy)
            print(f"{100.0 * accuracy:<8.2f}{path}")
    print(f"Average: {100.0 * sum(accuracies) / len(accuracies):2.2f}")


@click.command()
@click.argument("path", type=click.Path(file_okay=True))
@click.option("--accuracy/--do-accuracy", "do_accuracy", default=True,
              help="Computes accuracy against .tokens file.")
@click.option("--display/--no-display", "do_display", default=True,
              help="Displays / don't display the computed sequence.")
@click.pass_context
def predict(ctx, path: Path, do_display: bool, do_accuracy: bool):
    """Translates the given PATH image file into kern-like notation.

    Args:
        path (Path): Path f the image to decode.
        do_display (bool): Displays the sequence of decoded tokens.
        do_accuracy (bool): If a .tokens file is available, display accuracy.

    Raises:
        FileNotFoundError: _description_
    """
    from click_context import ClickContext
    context = cast(ClickContext, ctx.obj)
    gp = context.require_dataset()
    path = Path(path)
    source, _ = gp.load_image(path, pad=True)
    if source is None:
        raise FileNotFoundError(f"File {path} not found, likely too wide.")
    target = None
    if do_accuracy:
        target, _ = gp.load_sequence(path.with_suffix(".tokens"), pad=True)
    with torch.no_grad():
        accuracy = context.require_client().predict(
            context.require_dataset(), source, target, do_display=do_display)
        if do_accuracy:
            print(f"Accuracy: {100.0 * accuracy:2.2f}")


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
    print(f"git hash: {client.git_hash}")
    print(f"Training samples: {client.training_samples:,}")
    if do_config:
        print("Config:")
        for key, value in client.config.__dict__.items():
            print(f"\t{key:<20}: {value}")
    if do_summary:
        summary(client.model)
