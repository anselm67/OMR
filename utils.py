import subprocess
from pathlib import Path
from typing import Iterable, Optional, TypeAlias, Union

import torch

DeviceType = Union[str, torch.device]


def iterable_from_file(path: Union[str, Path]) -> Iterable[str]:
    with open(path, 'r') as file:
        for line in file:
            yield line


def current_commit() -> str:
    """Get the last Git commit hash."""
    try:
        # Use '--short' for a shorter hash if needed
        args = ["git", "rev-parse", "HEAD"]

        # Run the Git command and decode the output
        hash = subprocess.check_output(args).strip().decode("utf-8")
        return hash
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving commit hash: {e}")
        return "unknown-commit"


def compare_sequences(yhat: torch.Tensor, y: torch.Tensor) -> float:
    from grandpiano import GrandPiano
    if yhat.size(0) != y.size(0):
        width = max(yhat.size(0), y.size(0))
        paded = torch.full(
            (width, GrandPiano.Stats.max_chord), fill_value=GrandPiano.PAD[0])
        if yhat.size(0) > y.size(0):
            paded[0:y.size(0), :] = y
            y = paded.to(y.device)
        else:
            paded[0:yhat.size(0), :] = yhat
            yhat = paded.to(yhat.device)
    wrong = torch.sum(y != yhat).item()
    total = torch.sum(y != GrandPiano.PAD[0]).item()
    return 1.0 - wrong / total
