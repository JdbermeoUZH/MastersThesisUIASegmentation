from typing import Dict
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset

class TTAInterface(ABC):
    
    @abstractmethod
    def tta(self, x: torch.Tensor) -> None:
        pass
    
    @abstractmethod
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor,...]:
        pass
    
    @abstractmethod
    def evaluate(
        self, x_preprocessed: torch.Tensor, y_gt: torch.Tensor,
        preprocessed_pix_size: tuple[float, ...], gt_pix_size: tuple[float, ...]
    ) -> float | Dict[str, float]:
        pass

    @abstractmethod
    def evaluate_dataset(self, ds: Dataset) -> None:
        pass

    @abstractmethod
    def load_state(self, path: str) -> None:
        pass

    @abstractmethod
    def save_state(self, path: str) -> None:
        pass

    @abstractmethod
    def reset_state(self) -> None:
        pass

    @abstractmethod
    def _evaluation_mode(self) -> None:
        pass
