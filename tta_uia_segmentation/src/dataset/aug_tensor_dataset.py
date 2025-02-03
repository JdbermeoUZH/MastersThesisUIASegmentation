
from typing import Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset 
import torch.nn.functional as F

from tta_uia_segmentation.src.dataset.augmentation import apply_data_augmentation
from tta_uia_segmentation.src.dataset.utils import (
    transform_orientation,
    ensure_nd,
)
from tta_uia_segmentation.src.utils.utils import default, get_seed


class AgumentedTensorDataset(Dataset):
    def __init__(
            self,
            data: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            aug_params: Optional[dict[str, Any]] = None,
            seed: Optional[int] = None,
    ):
        self._data = data
        self._labels = labels
        self._aug_params = aug_params
        self._augment = aug_params is not None
        self._returns_labels = labels is not None
        self._seed = seed

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        seed = default(self._seed, get_seed())

        image = self._data[idx]  # Convert torch.Tensor to numpy array
        ensure_nd(3, image)

        if self._returns_labels:
            label = self._labels[idx]  # type: ignore
            ensure_nd(3, label)
        else:
            label = None

        if self.augment:
            assert self._aug_params is not None
            image, label = apply_data_augmentation(
                image,
                label,
                rng=np.random.default_rng(seed),
                return_torch=True,
                **self._aug_params,
            )
        
        if label is None:
            label = ["none"] * len(image)

        return image, label

    @property
    def augment(self):
        return self._augment
    
    @augment.setter
    def augment(self, augment: bool):
        self._augment = augment

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed=None):
        self._seed = seed

    @property
    def returns_labels(self):
        return self._returns_labels
