import torch
import numpy as np
from typing import Union

from dataset.dataset_in_memory import DatasetInMemory
from dataset.augmentation import apply_data_augmentation
from utils.utils import get_seed

       
class DatasetInMemoryForDDPM(DatasetInMemory):
    def __init__(
        self,
        concatenate: bool = True,
        axis_to_concatenate: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.concatenate = concatenate
        self.axis_to_concatenate = axis_to_concatenate
        
    def __getitem__(self, index) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        images = self.images[index, ...]
        labels = self.labels[index, ...]
        
        seed = get_seed() if self.seed is None else self.seed
        
        background_mask = self.background_mask[index,...]

        if self.augmentation:
            images, labels, background_mask = apply_data_augmentation(
                images,
                labels,
                background_mask,
                **self.aug_params,
                rng=np.random.default_rng(seed),
            )

        # Normalize label map to [0, 1]
        labels = labels / np.max(self.n_classes - 1)
        
        if not self.concatenate:
            return torch.from_numpy(images).float(), torch.from_numpy(labels).float()
        else:
            return torch.from_numpy(np.concatenate((images, labels), axis=self.axis_to_concatenate)).float()
         