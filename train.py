
import json
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from grandpiano import GrandPiano
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


def load_batch(gp: GrandPiano, batch_size: int = 8, device: str = "cpu"):
    samples = []
    for _ in range(batch_size):
        samples.append(gp.next(pad=True, device=device))
    return (
        # Images, image lengths, sequences, sequence lengths.
        torch.stack([sample[0] for sample in samples]).to(device),
        torch.stack([sample[2] for sample in samples]).to(device),
    )


cached_mask: Optional[torch.Tensor] = None
cached_mask_size: int = -1


def get_target_mask(size: int) -> torch.Tensor:
    global cached_mask, cached_mask_size
    if cached_mask is not None and size == cached_mask_size:
        return cached_mask
    else:
        cached_mask_size = size
        cached_mask = torch.triu(torch.ones(
            size, size), diagonal=1).to(torch.bool)
    return cached_mask


def decode(logits: torch.Tensor):
    probs = F.softmax(logits, dim=-1)
    tokids = torch.argmax(probs, dim=-1)
    match tokids.shape:
        case (batch, length, count):
            assert count == gp.CHORD_MAX, f"Expecting {
                gp.CHORD_MAX} tokens per tick."
            for b in range(batch):
                print(f"Batch {b}:")
                for chord in tokids[b]:
                    texts = gp.decode([int(tokid.item()) for tokid in chord])
                    print(" ".join(texts))
        case (lengh, tokens):
            print("one")
        case (token):
            print("token")
    pass


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

    def __init__(self, path: Path):
        self.path = path
        self.losses = list([])

    def save(self):
        with open(self.path, "w+") as f:
            json.dump({
                "losses": self.losses
            }, f, indent=4)

    def log(self, loss: float):
        self.losses.append(loss)
        self.save()


def train(num_epoch: int, start_epoch: int = 1, compile: bool = True):
    global model

    log_path = Path("untracked") / "train_log.json"
    checkpoint_path = Path("untracked") / "checkpoint.pt"
    log_path.unlink()
    checkpoint_path.unlink()

    log = TrainLog(log_path)
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    batch_size = 8

    if compile:
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    model = model.to(device)

    batch_per_epoch = gp.len() // batch_size
    loss_fn = nn.CrossEntropyLoss(ignore_index=int(gp.PAD[0]))
    opt = torch.optim.Adam(model.parameters(), lr=0.0001,
                           betas=(0.9, 0.98), eps=1e-9)
    start_time = time.time()
    for e in range(start_epoch, start_epoch + num_epoch):
        for b in range(batch_per_epoch):

            X, y = load_batch(gp, batch_size=batch_size, device=device)

            y_input = y[:, :-1, :]
            y_expected = y[:, 1:, :]

            target_mask = get_target_mask(y_input.shape[1])

            opt.zero_grad()
            logits = model(X, y_input, target_mask.to(device))
            # decode(logits)
            loss = loss_fn(
                logits.reshape(-1, gp.vocab_size),
                y_expected.flatten()
            )
            loss.backward()
            opt.step()

            if b % 50 == 0:
                done = 100.0 * b / batch_per_epoch
                now = time.time()
                print(f"Epoch {e} - {done:2.2f}% done in {
                      (now-start_time):.2f}s loss: {loss.item():2.2f}")
                start_time = now
                log.log(loss=loss.item())
            if b % 1000 == 0:
                checkpoint(checkpoint_path, e, model, opt)


if __name__ == '__main__':
    train(1)
