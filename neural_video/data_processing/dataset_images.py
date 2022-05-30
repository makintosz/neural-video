import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self._x = x.astype(np.float32)
        self._y = y.astype(np.float32)
        self._n_samples = len(self._y)

    def __getitem__(self, item) -> tuple[np.ndarray, np.ndarray]:
        return self._x[item], self._y[item]

    def __len__(self) -> int:
        return self._n_samples
