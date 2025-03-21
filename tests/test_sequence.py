import unittest

import torch
from torch import Tensor

from sequence import chord_distance, compare_sequences2


def create_chord(notes: list[int]) -> Tensor:
    chord = torch.full((12,), 0)
    for idx, note in enumerate(notes):
        chord[idx] = note
    return chord


class TestSequence(unittest.TestCase):

    def test_chord_distance(self):
        chord1 = create_chord([1, 2, 3])
        chord2 = create_chord([1, 2, 3])
        self.assertEqual(0, chord_distance(chord1, chord2))

        chord1 = create_chord([1, 2, 3, 4])
        chord2 = create_chord([1, 3, 2, 4])
        self.assertEqual(1, chord_distance(chord1, chord2))

        chord1 = create_chord([1, 2, 3, 4, 5])
        chord2 = create_chord([1, 2, 3])
        self.assertEqual(2, chord_distance(chord1, chord2))

        chord1 = create_chord([1, 2, 3])
        chord2 = create_chord([1, 2, 1, 3])
        self.assertEqual(2, chord_distance(chord1, chord2))

    def test_sorted_chord_distance(self):
        chord1 = create_chord([1, 2, 3])
        chord2 = create_chord([3, 2, 1])
        self.assertEqual(0, chord_distance(chord1, chord2))

    def test_sequence_distance(self):
        seq1 = torch.stack([
            create_chord([1, 2, 3]),
            create_chord([1, 2, 3]),
            create_chord([1, 2, 3]),
        ])
        seq2 = torch.stack([
            create_chord([1, 2, 3]),
            create_chord([1, 2, 3]),
            create_chord([1, 2, 3]),
        ])
        self.assertEqual(0, compare_sequences2(seq1, seq2))

        seq1 = torch.stack([
            create_chord([1, 2, 3]),
            create_chord([1, 2, 3]),
            create_chord([1, 2, 3]),
        ])
        seq2 = torch.stack([
            create_chord([1, 2, 3]),
            create_chord([1, 2, 3]),
            create_chord([1, 2, 3]),
            create_chord([1, 2, 3]),
        ])
        self.assertEqual(1, compare_sequences2(seq1, seq2))

        seq1 = torch.stack([
            create_chord([1, 2, 3]),
            create_chord([1, 2, 3]),
            create_chord([1, 2, 3]),
        ])
        seq2 = torch.stack([
            create_chord([1, 2, 3]),
            create_chord([1, 2, 3]),
            create_chord([1, 2, 4, 5, 6, 7]),
            create_chord([1, 2, 3]),
        ])
        self.assertEqual(1, compare_sequences2(seq1, seq2))
