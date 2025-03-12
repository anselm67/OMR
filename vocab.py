from typing import Iterable

import torch
from torch import Tensor

from config import Config


class Vocab:
    PAD_T = (0, "PAD")        # Padding for image and sequence length value.
    UNK_T = (1, "UNK")        # Unknown sequence token.
    SOS_T = (2, "SOS")        # Start of sequence token.
    EOS_T = (3, "EOS")        # End of sequence token.
    SIL_T = (4, "SIL")        # Chord padding to max_chord.
    RESERVED_TOKENS = [PAD_T, UNK_T, SOS_T, EOS_T, SIL_T]

    PAD, UNK, SOS, EOS, SIL = map(lambda x: x[0], RESERVED_TOKENS)

    config: Config
    _tok2i: dict[str, int]
    _i2tok: dict[int, str]

    def __init__(self, config: Config, _tok2i: dict[str, int]):
        self.config = config
        self._tok2i = _tok2i
        self._i2tok = {ival: key for key, ival in _tok2i.items()}

    def __len__(self):
        return len(self._tok2i)

    def tok2i(self, tokens: list[str]) -> Tensor:
        c = self.config
        tensor = torch.full((c.max_chord, ), self.SIL)
        for idx, tok in enumerate(tokens):
            tensor[idx] = self._tok2i.get(tok, self.UNK)
        return tensor

    def i2tok(self, ids: Tensor | Iterable[int]) -> list[str]:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        return [self._i2tok.get(id, self.UNK_T[1]) for id in ids]
