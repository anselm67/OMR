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


def get_model_device(model: torch.nn.Module) -> Optional[DeviceType]:
    """Get the device of a PyTorch model."""
    # Get the first parameter or buffer to determine the device
    for param in model.parameters():
        return param.device
    for buffer in model.buffers():
        return buffer.device
    return None  # If the model has no parameters or buffers
