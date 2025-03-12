
from io import StringIO

import torch
from torch import Tensor

from vocab import Vocab


def display_sequence(vocab: Vocab, yhat: Tensor, gt: None | Tensor = None) -> str:
    buffer = StringIO()

    def chord_repr(vocab: Vocab, chord: Tensor) -> str:
        # Otherwise, displays anything but PAD.
        if any([id != Vocab.PAD for id in chord]):
            texts = vocab.i2tok([
                int(id.item()) for id in chord if id != Vocab.SIL
            ])
            return " ".join([text for text in texts if text])
        else:
            return ""

    def is_eos(chord):
        return all([id.item() == Vocab.EOS for id in chord])

    if gt is None:
        for chord in yhat:
            # Skips SOS.
            if all([id.item() == Vocab.SOS for id in chord]):
                continue
            # Returns on EOS.
            if all([id.item() == Vocab.EOS for id in chord]):
                break
            buffer.write(f"{chord_repr(vocab, chord)}\n")
    else:
        yhat_done, gt_done = False, False
        for chord_hat, chord_gt in zip(yhat, gt):
            # Skips SOS.
            if all([id.item() == Vocab.SOS for id in chord_hat]):
                continue
            # Returns when both sequences are done.
            yhat_done = yhat_done or is_eos(chord_hat)
            gt_done = gt_done or is_eos(chord_gt)
            if yhat_done and gt_done:
                break
            buffer.write(
                f"{chord_repr(vocab, chord_gt):<40}{chord_repr(vocab, chord_hat)}\n"
            )

    return buffer.getvalue()


def compare_sequences(yhat: Tensor, y: Tensor) -> float:
    from grandpiano import GrandPiano
    if yhat.size(0) != y.size(0):
        padded = torch.full(
            (max(yhat.size(0), y.size(0)), GrandPiano.Stats.max_chord),
            fill_value=GrandPiano.PAD[0])
        if yhat.size(0) > y.size(0):
            padded[0:y.size(0), :] = y
            y = padded.to(y.device)
        else:
            padded[0:yhat.size(0), :] = yhat
            yhat = padded.to(yhat.device)
    wrong = torch.sum(y != yhat).item()
    total = torch.sum(y != GrandPiano.PAD[0]).item()
    return 1.0 - wrong / total
