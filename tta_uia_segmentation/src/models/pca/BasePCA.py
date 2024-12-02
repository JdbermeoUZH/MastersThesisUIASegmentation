from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

class BasePCA(ABC):
    @property
    @abstractmethod
    def n_components(self) -> Optional[int]:
        pass

    @n_components.setter
    @abstractmethod
    def n_components(self, value: Optional[int]):
        pass

    @abstractmethod
    def to_pcs(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def from_pcs(self, z: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def num_components_to_keep(self, variance_to_keep: float) -> int:
        pass

    @abstractmethod
    def img_to_pcs(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def img_from_pcs(self, z: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def img_reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def to_device(self, device: torch.device | str):
        pass
    
