import torch
import numpy as np

from dataset.dataset_all_in_memory import DatasetInMemory
from dataset.augmentation import apply_data_augmentation
from utils.utils import get_seed

       
class DatasetH5ForDDPM(DatasetInMemory):
    def __init__(
        self,
        concatenate: bool = True,
        concatenate_as_channels: bool = False,
        axis_to_concatenate: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.concatenate = concatenate
        self.concatenate_as_channels = concatenate_as_channels
        
    def __getitem__(self, index):

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
            return torch.from_numpy(images), torch.from_numpy(labels)
        else:
            if self.concatenate_as_channels:
                return torch.from_numpy(np.concatenate((images[None, ...], labels[None, ...]), axis=0))
            else:
                return torch.from_numpy(np.concatenate((images, labels), axis=1))
         