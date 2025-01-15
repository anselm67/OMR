from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import click
import torch
from torchinfo import summary

from grandpiano import GrandPiano
from model import Config, Translator
from utils import DeviceType, compare_sequences, current_commit


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

    def beam_decode(self, gp: GrandPiano, source: torch.Tensor) -> torch.Tensor:
        top_k = 3
        beam_width = 6

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

        beams: List[Tuple[torch.Tensor, float]] = [(ys, 0.0)]
        done:  List[Tuple[torch.Tensor, float]] = []
        for idx in range(1, gp.spad_len-1):
            candidates:  List[Tuple[torch.Tensor, float]] = []

            for seq, score in beams:

                target_mask = self.get_target_mask(idx)
                out = self.model.decode(
                    seq[:idx, :].unsqueeze(0), memory, target_mask)
                out = out.transpose(0, 1)
                out = self.model.generator(out[:, -1])
                out = out.view(-1, gp.Stats.max_chord, gp.vocab_size)
                prob = torch.nn.functional.log_softmax(out, dim=-1)

                log_probs, tokens = torch.topk(prob[-1, :, :], top_k)
                for i in range(top_k):
                    candidate = seq.clone()
                    candidate[idx, :] = tokens[:, i]
                    log_prob = torch.sum(log_probs[:, i]).item()
                    if tokens[i, 0] == gp.EOS[0]:
                        done.append((candidate, score + log_prob))
                    else:
                        candidates.append(
                            (candidate, score + log_prob))
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)
            beams = beams[:beam_width]

            if len(done) >= beam_width:
                break

        return max(done, key=lambda x: x[1])[0] if done else beams[0][0]

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
            # As we're not runing through softmax this isn't really a probability,
            # still fine as we're only interested in argmax.
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
        do_display: bool = True,
        use_beam: bool = False
    ) -> float:
        """Translated the image given by PATH to kern-like tokens.

        Args:
            gp (GrandPiano): The dataset, to decode the tokens to string.
            source (torch.Tensor): Image to decode.
            target (torch.Tensor, optional): Compute and return accuracy against this 
                given target. Defaults to None.
            do_display (bool, optional): Displays the computed tokens. Defaults to True.
            use_beam (bool, optional): Use beam decoding rather than greedy. Defaults to False.
        """
        self.model.eval()
        if source is None:
            raise FileNotFoundError(f"File {path} not found, likely too wide.")

        if use_beam:
            chords = self.greedy_decode(gp, source.unsqueeze(0))
        else:
            chords = self.beam_decode(gp, source.unsqueeze(0))
        accuracy = 0.0
        if target is not None:
            accuracy = compare_sequences(chords, target)
        if do_display:
            gp.display(chords)
        return accuracy


@click.command()
@click.argument("count", type=int)
@click.option("--display/--no-display", "do_display", default=True,
              help="Displays / don't display the computed sequence.")
@click.option("--beam-decoding", "use_beam", is_flag=True, default=False,
              help="Use beam decoding rather than greedy.")
@click.pass_context
def random_check(ctx, count: int, do_display: bool, use_beam: bool):
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
                context.require_dataset(),
                image, sequence,
                do_display=do_display, use_beam=use_beam
            )
            accuracies.append(accuracy)
            print(f"{100.0 * accuracy:<8.2f}{path}")
    print(f"Average: {100.0 * sum(accuracies) / len(accuracies):2.2f}")


@click.command()
@click.argument("path", type=click.Path(file_okay=True))
@click.option("--accuracy/--do-accuracy", "do_accuracy", default=True,
              help="Computes accuracy against .tokens file.")
@click.option("--display/--no-display", "do_display", default=True,
              help="Displays / don't display the computed sequence.")
@click.option("--beam-decoding", "use_beam", is_flag=True, default=False,
              help="Use beam decoding rather than greedy.")
@click.pass_context
def predict(ctx, path: Path, do_display: bool, do_accuracy: bool, use_beam: bool):
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
            context.require_dataset(),
            source, target,
            do_display=do_display, use_beam=use_beam
        )
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
