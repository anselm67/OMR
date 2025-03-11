from pathlib import Path
from typing import Optional

from client import Model
from grandpiano import GrandPiano
from model import Config
from train import Train


class ClickContext:

    dataset_path: Optional[Path] = None
    model_path: Optional[Path] = None
    model_name: Optional[str] = None

    gp: Optional[GrandPiano] = None
    train: Optional[Train] = None
    client: Optional[Model] = None

    def __init__(self, dataset_path: Path, model_path: Path, model_name: str):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model_name = model_name

    def get_train_log(self):
        if self.model_path is None or self.model_name is None:
            raise ValueError("A model name and directory is required.")
        return Train.get_train_log_path(self.model_path, self.model_name)

    def require_dataset(self) -> GrandPiano:
        if self.gp is None:
            if self.dataset_path is None:
                raise ValueError(f"A path to the GrandPiano is required.")
            self.gp = GrandPiano(
                self.dataset_path,
                filter=GrandPiano.Filter(
                    max_image_width=1024, max_sequence_length=128
                )
            )
        return self.gp

    def get_config(self) -> Config:
        gp = self.require_dataset()
        return Config(
            ipad_shape=(GrandPiano.Stats.image_height, gp.ipad_len),
            max_chord=GrandPiano.Stats.max_chord,
            spad_len=gp.spad_len,                     # TODO
            vocab_size=gp.vocab_size,
        )

    def require_train(self) -> Train:
        if self.train is None:
            if self.model_path is None or self.model_name is None:
                raise ValueError("A model name and directory is required.")
            self.train = Train(
                self.get_config(),
                self.require_dataset(),
                self.model_path,
                self.model_name)
        return self.train

    def require_client(self) -> Model:
        if self.client is None:
            if self.train is None:
                if self.model_path is None or self.model_name is None:
                    raise ValueError("A model name and directory is required.")
                self.client = Model(self.get_config(),
                                    self.model_path,
                                    self.model_name)
            else:
                self.client = self.train
        return self.client
