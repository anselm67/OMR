
from io import StringIO

import numpy as np
import torch
from torch import Tensor

from vocab import Vocab


def display_sequence(vocab: Vocab, yhat: Tensor, gt: None | Tensor = None) -> str:
    buffer = StringIO()

    def chord_repr(vocab: Vocab, chord: Tensor) -> str:
        # Otherwise, displays anything but PAD.
        if any([id != Vocab.PAD for id in chord]):
            texts = vocab.i2tok((
                int(id.item()) for id in chord if id != Vocab.SIL and id != Vocab.EOS
            ))
            return " ".join((text for text in texts if text))
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
                if s_gt != s_hat:
                    buffer.write("\033[31m" f"{s_gt:<40}{s_hat}" "\033[0m\n")
                else:
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


def chord_distance(seq1: Tensor, seq2: Tensor):
    """Compute Levenshtein distance between two sequences (lists of notes)."""
    assert seq1.size(0) == seq2.size(0), \
        f"Expecting same chord size {seq1.shape[1]} vs {seq2.shape[1]}"
    length = seq1.size(0)
    (seq1, _), (seq2, _) = torch.sort(seq1), torch.sort(seq2)
    dp = np.zeros((length + 1, length + 1), dtype=int)

    for i in range(length + 1):
        for j in range(length + 1):
            if i == 0:
                dp[i][j] = j  # Cost of inserting j notes
            elif j == 0:
                dp[i][j] = i  # Cost of deleting i notes
            else:
                cost = 1 if seq1[i - 1] != seq2[j - 1] else 0
                dp[i][j] = min(
                    dp[i - 1][j] + 1,       # Deletion
                    dp[i][j - 1] + 1,       # Insertion
                    dp[i - 1][j - 1] + cost  # Substitution
                )

                if i > 1 and j > 1 and seq1[i - 1] == seq2[j - 2] and seq1[i - 2] == seq2[j - 1]:
                    dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)

    return dp[length][length].item()


def compare_sequences2(seq1: Tensor, seq2: Tensor):
    """Compute edit distance between two seq`uences of chords."""
    ins_del_cost = 1
    len1, len2 = seq1.size(0), seq2.size(0)
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)

    for i in range(len1 + 1):
        for j in range(len2 + 1):
            if i == 0:
                dp[i][j] = j * ins_del_cost
            elif j == 0:
                dp[i][j] = i * ins_del_cost
            else:
                cost = chord_distance(seq1[i - 1], seq2[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j] + ins_del_cost,  # Deletion
                    dp[i][j - 1] + ins_del_cost,  # Insertion
                    dp[i - 1][j - 1] + cost       # Substitution
                )
    return dp[len1][len2].item()
