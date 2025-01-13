
import json
import time
from pathlib import Path
from typing import List, Optional

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from grandpiano import Dataset, GrandPiano
from model import Config, Translator

DATADIR = Path("/home/anselm/Downloads/GrandPiano")

gp = GrandPiano(
    DATADIR,
    filter=GrandPiano.Filter(
        max_image_width=1024,
        max_sequence_length=128
    )
)

config = Config(
    image_height=gp.image_height,
    max_image_width=gp.ipad_len,                       # TODO
    max_sequence_height=GrandPiano.CHORD_MAX,
    max_sequence_width=gp.spad_len,                     # TODO
    vocab_size=gp.vocab_size,
)

model = Translator(config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,
                             betas=(0.9, 0.98), eps=1e-9)
start_epoch = 1

MODEL_PATH = Path("untracked") / "checkpoint.pt"
if MODEL_PATH.exists():
    obj = torch.load(MODEL_PATH, weights_only=True)
    start_epoch = obj["epoch"]
    model.load_state_dict(obj["state_dict"])
    optimizer.load_state_dict(obj['optimizer'])


def load_batch(gp: GrandPiano, dataset_name: Dataset, batch_size: int = 8, device: str = "cpu"):
    samples = []
    for _ in range(batch_size):
        samples.append(gp.next(dataset_name, pad=True, device=device))
    return (
        # Images, image lengths, sequences, sequence lengths.
        torch.stack([sample[0] for sample in samples]).to(device),
        torch.stack([sample[2] for sample in samples]).to(device),
    )


cached_mask: Optional[torch.Tensor] = None
cached_mask_size: int = -1


def get_target_mask(size: int, device: str = "cpu") -> torch.Tensor:
    global cached_mask, cached_mask_size
    if cached_mask is not None and size == cached_mask_size:
        return cached_mask
    else:
        cached_mask_size = size
        cached_mask = torch.triu(torch.ones(
            size, size), diagonal=1).to(torch.bool).to(device)
    return cached_mask


def checkpoint(path: Path, epoch: int, model: nn.Module, opt: torch.optim.Adam):
    print(f"Checkpoint to {path}")
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': opt.state_dict()
    }, path)


class TrainLog:

    path: Path
    losses: List[float]
    vlosses: List[float]

    def __init__(self, path: Path):
        self.path = path
        # Loads existing log if available.
        if path.exists():
            with open(self.path, "r") as f:
                obj = json.load(f)
            self.losses = obj["losses"]
            self.vlosses = obj["vlosses"]
        else:
            self.losses = list([])
            self.vlosses = list([])

    def save(self):
        with open(self.path, "w+") as f:
            json.dump({
                "losses": self.losses,
                "vlosses": self.vlosses
            }, f, indent=4)

    def log(self, loss: float, vloss: float):
        self.losses.append(loss)
        self.vlosses.append(vloss)
        self.save()


def evaluate(loss_fn, batch_size: int, count: int = 5, device: str = "cpu"):
    model.eval()
    vlosses = []

    for _ in range(count):
        X, y = load_batch(gp, "valid", batch_size, device=device)

        y_input = y[:, :-1, :]
        y_expected = y[:, 1:, :]

        target_mask = get_target_mask(y_input.shape[1], device=device)

        logits = model(X, y_input, target_mask)

        loss = loss_fn(
            logits.reshape(-1, gp.vocab_size),
            y_expected.flatten()
        )

        vlosses.append(loss.item())
    model.train()
    return torch.tensor(vlosses).mean().item()


# @click.command
# @click.argument("num_epoch", type=int, required=True,)
def train(num_epoch: int, start_epoch: int = 1, compile: bool = False):
    """
        Train the model for NUM_EPOCH epochs, resuming from the last 
        checkpoint when available.
    """
    global model

    log_path = Path("untracked") / "train_log.json"
    checkpoint_path = Path("untracked") / "checkpoint.pt"

    log = TrainLog(log_path)
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    batch_size = 16

    if compile:
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    # Moves the model and optimizer state to the device.
    model = model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Runs the trainin loop.
    batch_per_epoch = gp.len("train") // batch_size
    loss_fn = nn.CrossEntropyLoss(ignore_index=int(gp.PAD[0]))
    start_time = time.time()
    for e in range(start_epoch, start_epoch + num_epoch):
        for b in range(batch_per_epoch):

            X, y = load_batch(
                gp, "train", batch_size=batch_size, device=device)

            y_input = y[:, :-1, :]
            y_expected = y[:, 1:, :]

            target_mask = get_target_mask(y_input.shape[1], device=device)

            optimizer.zero_grad()
            logits = model(X, y_input, target_mask)
            loss = loss_fn(
                logits.reshape(-1, gp.vocab_size),
                y_expected.flatten()
            )
            loss.backward()
            optimizer.step()

            if b % 50 == 0:
                done = 100.0 * b / batch_per_epoch
                now = time.time()
                vloss = evaluate(loss_fn, batch_size, device=device)
                print(f"Epoch {e} - {done:2.2f}% done in {
                      (now-start_time):.2f}s loss: {loss.item():2.2f}, vloss: {vloss:2.2f}")
                log.log(loss=loss.item(), vloss=vloss)
                start_time = time.time()

            if b % 1000 == 0:
                checkpoint(checkpoint_path, e, model, optimizer)


def greedy_decode(
    source: torch.Tensor,
    max_len: int, start_symbol: int,
    device: str = "cpu"
) -> torch.Tensor:
    source_key_padding_mask = (source == GrandPiano.PAD[0])[:, :, 0]
    memory = model.encode(
        source,
        src_key_padding_mask=source_key_padding_mask,
    ).to(device)
    ys = torch.full(
        (1, gp.CHORD_MAX),
        fill_value=start_symbol
    ).to(device)
    for i in range(max_len-1):
        target_mask = get_target_mask(ys.shape[0], device=device)
        out = model.decode(ys.unsqueeze(0), memory, target_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        prob = prob.view(-1, gp.CHORD_MAX, gp.vocab_size)
        token = torch.argmax(prob[-1:, :, :], dim=2)
        ys = torch.cat([ys, token], dim=0)
        if token[0, 0] == gp.EOS[0]:
            break
    return ys


@click.command
@click.argument("path", type=click.Path(file_okay=True))
def predict(path: Path):
    """

        Translates the given PATH image file into kern-ike notation.
    """
    image, _ = gp.load_image(path, pad=True)
    if image is None:
        raise FileNotFoundError(f"File {path} not found, likely too wide.")

    chords = greedy_decode(
        image.unsqueeze(0), gp.spad_len, start_symbol=gp.SOS[0])
    for chord in chords:
        texts = gp.decode([int(id.item())
                          for id in chord if id != GrandPiano.SIL[0]])
        print("\t".join(texts))

    pass


if __name__ == '__main__':
    train(10, start_epoch=start_epoch)
