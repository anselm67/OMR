import subprocess
from pathlib import Path
from typing import Iterable, Union


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
