from abc import ABC, abstractmethod

import numpy as np


class VideoRenderer(ABC):
    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray) -> dict[str, list]:
        pass

    @abstractmethod
    def generate_all_frames(self, epoch: int) -> None:
        pass

    @abstractmethod
    def generate_frame(self, data: list) -> np.ndarray:
        pass

    @abstractmethod
    def save_model(self) -> None:
        pass

    @abstractmethod
    def load_model(self, path_linear: str, path_convolutional: str) -> None:
        pass
