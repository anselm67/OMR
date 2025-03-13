
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

    def is_tok(chord: Tensor, tok: int):
        return all([id.item() == tok for id in chord])

    if gt is None:
        for chord in yhat:
            # Skips SOS.
            if is_tok(chord, vocab.SOS):
                continue
            # Returns on EOS.
            if is_tok(chord, vocab.EOS):
                break
            if (s := chord_repr(vocab, chord)):
                buffer.write(f"{s}\n")
    else:
        yhat_done, gt_done = False, False
        for chord_hat, chord_gt in zip(yhat, gt):
            if is_tok(chord_hat, vocab.SOS):
                continue
            # Returns when both sequences are done.
            yhat_done = yhat_done or is_tok(chord_hat, vocab.EOS)
            gt_done = gt_done or is_tok(chord_gt, vocab.EOS)
            if yhat_done and gt_done:
                break
            s_gt = chord_repr(vocab, chord_gt)
            s_hat = chord_repr(vocab, chord_hat)
            if s_gt or s_hat:
                buffer.write(f"{s_gt:<40}{s_hat}\n")

    return buffer.getvalue()


def compare_sequences(yhat: Tensor, y: Tensor) -> float:
    assert yhat.size(1) == y.size(1), \
        f"Expecting same chord size {yhat.shape[1]} vs {y.shape[1]}"
    if yhat.size(0) != y.size(0):
        padded = torch.full(
            (max(yhat.size(0), y.size(0)), yhat.size(1)),
            fill_value=Vocab.PAD)
        if yhat.size(0) > y.size(0):
            padded[0:y.size(0), :] = y
            y = padded.to(y.device)
        else:
            padded[0:yhat.size(0), :] = yhat
            yhat = padded.to(yhat.device)
    wrong = torch.sum(y != yhat).item()
    total = torch.sum(y != Vocab.PAD).item()
    return 1.0 - wrong / total
