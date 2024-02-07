import torch
import numpy as np
from typing import Union
from torch.utils.data import Subset, Dataset

from dataset.dataset_in_memory import DatasetInMemory
from dataset.augmentation import apply_data_augmentation
from utils.utils import get_seed


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def distribute_n_in_m_slots(n, m):
    elements_per_slot = n // m
    slots_with_extra_element = n % m
    equitably_dist_list = slots_with_extra_element * [elements_per_slot + 1] + \
        (m - slots_with_extra_element) * [elements_per_slot]
    np.random.shuffle(equitably_dist_list)

    return equitably_dist_list

def get_datasets(
   splits,
   concatenate_along_channel: bool = False,
   *args,
    **kwargs,
):

    datasets = []

    for split in splits:
        datasets.append(
            DatasetInMemoryForDDPM(split=split, concatenate_along_channel=concatenate_along_channel,
                                   *args, **kwargs)
        )

    return datasets

       
class DatasetInMemoryForDDPM(DatasetInMemory):
    def __init__(
        self,
        concatenate_along_channel: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.concatenate_along_channel = concatenate_along_channel
        self.num_vols = int(self.images.shape[0] / self.dim_proc[-1]) if self.image_size[0] == 1 else self.images.shape[0]
        
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
        
        if self.concatenate_along_channel:
            return torch.from_numpy(np.concatenate([images, labels], axis=-1)).float()
        else:
            return torch.from_numpy(images).float(), torch.from_numpy(labels).float()
    
    
    def sample_slices(self, sample_size: int, range: tuple[float, float] = (0.2, 0.8)) -> Dataset:
        
        # Get the number of slices to sample from each volume
        n_slices_per_volume = distribute_n_in_m_slots(sample_size, self.num_vols)
        
        # Sample the idxs of the slices
        min_slice_idx, max_slice_idx = int(self.dim_proc[-1] * range[0]), int(self.dim_proc[-1] * range[1])
        sampled_slices = [np.random.randint(idx_start * min_slice_idx, (idx_start + 1) * max_slice_idx, n_slices) 
                          for idx_start, n_slices in enumerate(n_slices_per_volume)]
        sampled_slices = list(np.concatenate(sampled_slices))
        
        return Subset(self, sampled_slices)
        
        
        
        
        