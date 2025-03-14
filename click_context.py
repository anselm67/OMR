import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional

import lightning as L

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

    def __init__(self, dataset_directory: Path, model_directory: Path):
        self.dataset_directory = dataset_directory
        self.model_directory = model_directory
        # We always have a config, no matter what.
        self.config = Config.create(self.model_directory / "config.json")

    def require_trainer(self, **kwargs) -> tuple[LitTranslator, L.Trainer]:
        if self.model_directory is None:
            raise FileNotFoundError("No model directory, no model.")
        if self.model_directory.exists():
            logging.info(f"Reusing model in {self.model_directory.name}")
            model = LitTranslator(self.config)
        else:
            logging.info(f"Creating model in {self.model_directory.name}")
            factory = self.require_factory()
            self.config = replace(self.config, vocab_size=len(factory.vocab))
            self.model_directory.mkdir(parents=True, exist_ok=True)
            self.config.save(self.model_directory / "config.json")
            model = LitTranslator(self.config)
        trainer = L.Trainer(
            default_root_dir=self.model_directory,
            logger=SimpleLogger(self.model_directory / "train_logs.json"),
            **kwargs
        )
        return model, trainer

    def require_model(self) -> LitTranslator:
        if self.model_directory is None:
            raise FileNotFoundError("No model directory, no model.")
        model = LitTranslator.load_from_checkpoint(
            self.model_directory / "last.ckpt", config=self.config
        )
        return model

    def require_factory(self) -> Factory:
        if self.factory is None:
            if self.dataset_directory is None:
                raise ValueError(f"A path to the dataset is required.")
            self.factory = Factory(self.dataset_directory, self.config)
        return self.factory

    def require_logger(self) -> SimpleLogger:
        if self.model_directory is None:
            raise FileNotFoundError("No model directory, no logger.")
        return SimpleLogger(self.model_directory / "train_logs.json")
