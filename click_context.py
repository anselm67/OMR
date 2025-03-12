from pathlib import Path
from typing import Optional

from client import Client
from dataset import Factory
from logger import SimpleLogger
from model import Config
from train2 import LitTranslator


class ClickContext:

    config: Config

    dataset_directory: Path
    model_directory: Path

    factory: Optional[Factory] = None
    lit_model: Optional[LitTranslator] = None
    client: Optional[Client] = None

    def __init__(self, dataset_directory: Path, model_directory: Path):
        self.dataset_directory = dataset_directory
        self.model_directory = model_directory
        # We always have a config, no matter what.
        self.config = Config.create(self.model_directory / "config.json")

    def require_factory(self) -> Factory:
        if self.factory is None:
            if self.dataset_directory is None:
                raise ValueError(f"A path to the GrandPiano is required.")
            self.factory = Factory(self.dataset_directory, self.config)
        return self.factory

    def require_client(self) -> Client:
        if self.client is None:
            self.client = Client(
                self.config, self.model_directory / "last.ckpt")
        return self.client
