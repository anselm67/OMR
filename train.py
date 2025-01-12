
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from grandpiano import GrandPiano
from model import Config, Translator

DATADIR = Path("/home/anselm/Downloads/GrandPiano")

sequence_pad_length = 500
image_pad_length = 3092

gp = GrandPiano(
    DATADIR,
    spad_len=sequence_pad_length,
    ipad_len=image_pad_length,
)
config = Config(
    image_height=gp.image_height,
    max_image_width=image_pad_length,                       # TODO
    max_sequence_height=GrandPiano.CHORD_MAX,
    max_sequence_width=sequence_pad_length,                     # TODO
    vocab_size=gp.vocab_size,
)

model = Translator(config)


def load_batch(gp: GrandPiano, batch_size: int = 8):
    samples = []
    for _ in range(batch_size):
        samples.append(gp.next(pad=True))
    return (
        # Images, image lengths, sequences, sequence lengths.
        torch.stack([sample[0] for sample in samples]),
        torch.stack([sample[2] for sample in samples]),
    )


def get_target_mask(size: int) -> torch.Tensor:
    return torch.triu(torch.ones(size, size), diagonal=1).to(torch.bool)


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


def train(num_epoch: int, start_epoch: int = 1):
    global model

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    model = model.to(device)
    batch_count = 1000
    loss_fn = nn.CrossEntropyLoss(ignore_index=int(gp.PAD[0]))
    opt = torch.optim.Adam(model.parameters(), lr=0.0001,
                           betas=(0.9, 0.98), eps=1e-9)

    for _ in range(batch_count):

        X, y = load_batch(gp)
        X, y = X.to(device), y.to(device)

        y_input = y[:, :-1, :]
        y_expected = y[:, 1:, :]

        target_mask = get_target_mask(y_input.shape[1])
        logits = model(X, y_input, target_mask.to(device))
        # decode(logits)
        opt.zero_grad()
        loss = loss_fn(
            logits.view(-1, gp.vocab_size),
            y_expected.flatten()
        )
        loss.backward()
        opt.step()

        print(loss.item())


if __name__ == '__main__':
    train(1)
