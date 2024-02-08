import torch
import numpy as np
from typing import Union
from torch.utils.data import Subset, Dataset

from tta_uia_segmentation.src.dataset.dataset_in_memory import DatasetInMemory
from tta_uia_segmentation.src.dataset.augmentation import apply_data_augmentation
import tta_uia_segmentation.src.dataset.utils as du
from tta_uia_segmentation.src.utils.utils import get_seed


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
        normalize_img: str = 'min_max',
        one_hot_encode: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.concatenate_along_channel = concatenate_along_channel
        self.normalize_img = normalize_img
        self.one_hot_encode = one_hot_encode
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
            
        images = torch.from_numpy(images).float()
        labels = torch.from_numpy(labels).float()

        if self.one_hot_encode:
            labels = du.class_to_onehot(labels, self.n_classes, class_dim=0)

        # Normalize image and label map to [0, 1]
        if self.normalize_img == 'min_max':
            images = (images - images.min())/ (images.max() - images.min())
            
        labels = labels / (self.n_classes - 1)
        
        if self.concatenate_along_channel:
            return torch.concatenate([images, labels], axis=0).float()
        
        else:
            return images, labels    
    
    def sample_slices(self, sample_size: int, range: tuple[float, float] = (0.2, 0.8)) -> Dataset:
        
        # Get the number of slices to sample from each volume
        n_slices_per_volume = distribute_n_in_m_slots(sample_size, self.num_vols)
        
        # Sample the idxs of the slices
        min_slice_idx, max_slice_idx = int(self.dim_proc[-1] * range[0]), int(self.dim_proc[-1] * range[1])
        sampled_slices = [np.random.randint(idx_start * min_slice_idx, (idx_start + 1) * max_slice_idx, n_slices) 
                          for idx_start, n_slices in enumerate(n_slices_per_volume)]
        sampled_slices = list(np.concatenate(sampled_slices))
        
        return Subset(self, sampled_slices)
        
        
        
if __name__ == '__main__':
    import sys
    sys.path.append('/scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/')
    paths = {
        'train': '/scratch_net/biwidl319/jbermeo/logs/brain/preprocessing/hcp_t1/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_20.hdf_normalized_with_nn.h5',
        'test': '/scratch_net/biwidl319/jbermeo/logs/brain/preprocessing/hcp_t1/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_50_to_70.hdf_normalized_with_nn.h5',
        'val': '/scratch_net/biwidl319/jbermeo/logs/brain/preprocessing/hcp_t1/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_20_to_25.hdf_normalized_with_nn.h5'
    }
    
    paths_original = {
        'train': '/scratch_net/biwidl319/jbermeo/data/previous_results_nico/datasets/data_T1_original_depth_256_from_0_to_20.hdf5',
        'test': '/scratch_net/biwidl319/jbermeo/data/previous_results_nico/datasets/data_T1_original_depth_256_from_50_to_70.hdf5',
        'val': '/scratch_net/biwidl319/jbermeo/data/previous_results_nico/datasets/data_T1_original_depth_256_from_20_to_25.hdf5'
    }  
  
    
    ds = DatasetInMemoryForDDPM(
        split          = 'train',
        paths           = paths,
        paths_original  = paths_original,
        image_size      = (1, 256, 256),
        resolution_proc = [0.7, 0.7, 0.7],
        dim_proc        = (256, 256, 256),
        n_classes       = 15,
        aug_params      = None,
        bg_suppression_opts = None,
        deformation     = None,
        load_original   = False,
    )
    
    img, seg = ds[0]
    assert img.shape == (1, 256, 256), 'Image should have shape (1, 256, 256)'
    assert img.max() <= 1, 'Image should be normalized to [0, 1] range'
    assert img.min() >= 0, 'Image should be normalized to [0, 1] range'
    assert seg.shape == (15, 256, 256), 'Segmentation should have shape (15, 256, 256)'
