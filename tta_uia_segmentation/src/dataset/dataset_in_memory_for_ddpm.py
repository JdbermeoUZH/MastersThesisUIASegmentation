import random 
from typing import Optional

import h5py
import torch
import numpy as np
from typing import Union
from torch.utils.data import Subset, Dataset, DataLoader
from torchmetrics.functional.classification import dice

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
   *args,
    **kwargs,
):

    datasets = []

    for split in splits:
        datasets.append(
            DatasetInMemoryForDDPM(split=split, *args, **kwargs)
        )

    return datasets

       
class DatasetInMemoryForDDPM(DatasetInMemory):
    def __init__(
        self,
        norm: Optional[torch.nn.Module],
        use_original_imgs: bool = False,
        concatenate_along_channel: bool = False,
        normalize: str = 'min_max',
        one_hot_encode: bool = True, 
        intensity_value_range: Optional[tuple[float, float]] = None,
        norm_device: str = 'cpu',
        norm_neg_one_to_one: bool = False,
        paths_normalized_h5: Optional[dict[str, str]] = None,
        shard: int = 0,
        num_shards: int = 1,
        *args,
        **kwargs,
    ):
        
        if norm is None and not use_original_imgs:
            if paths_normalized_h5 is None:
                raise ValueError('Either a normalization model or paths to normalized images should be given,'
                                 'if not using original images')
            kwargs['paths'] = paths_normalized_h5
        
        super().__init__(*args, **kwargs)
        
        self.norm = norm
        self.norm_device = norm_device
        if self.norm is not None:
            self.norm.eval()
            self.norm.requires_grad_(False)
            self.norm.to(norm_device)
        
        self.normalized_img_path = paths_normalized_h5[self.split] if paths_normalized_h5 is not None else None
        
        self.concatenate_along_channel = concatenate_along_channel
        self.normalize = normalize
        self.norm_neg_one_to_one = norm_neg_one_to_one
        self.one_hot_encode = one_hot_encode
        self.num_vols = int(self.images.shape[0] / self.dim_proc[0]) if self.image_size[0] == 1 else self.images.shape[0]
        
        if intensity_value_range is not None:
            self.images_min, self.images_max = intensity_value_range
        elif use_original_imgs:
            self.images_min, self.images_max = self._find_min_max_in_original_imgs()
        else:
            self.images_min, self.images_max = self._find_min_max_in_normalized_imgs()
        
        print(f'Min and max values of normalized images: {self.images_min}, {self.images_max}')
        
        # Shard the dataset
        self.images = self.images[shard:][::num_shards]
        self.labels = self.labels[shard:][::num_shards]
        

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
        
        # Use the normalization model to normalize the images, if one is given
        if self.norm is not None:
            with torch.inference_mode():
                images = images.to(self.norm_device)
                labels = labels.to(self.norm_device)
                images = self.norm(images[None, ...]).squeeze(0)
                
        # Update the min and max values of the images, in case a larger value is observed
        img_max, img_min = images.max(), images.min()   
        if img_min < self.images_min:
            self.images_min = img_min
        if img_max > self.images_max:
            self.images_max = img_max

        # Normalize image and label map to [0, 1]
        if self.normalize == 'min_max':
            images = du.normalize(type='min_max', data=images,
                                  max=self.images_max, min=self.images_min,
                                  scale=1)
            if self.one_hot_encode:
                labels = du.class_to_onehot(labels, self.n_classes, class_dim=0)
            else:
                labels = du.normalize(type='min_max', data=labels, 
                                      min=0, max=self.n_classes - 1, 
                                      scale=1)
        
        elif self.one_hot_encode:
            labels = du.class_to_onehot(labels, self.n_classes, class_dim=0) 
        
        else:
            raise ValueError('Only min_max normalization is supported at the moment.')

        # Normalize image to [-1, 1], if specified
        if self.norm_neg_one_to_one:
            images = du.normalize_to_neg_one_to_one(images)
            labels = du.normalize_to_neg_one_to_one(labels.type(torch.int8))
        
        if self.concatenate_along_channel:
            return torch.concatenate([images, labels], axis=0).float()
        
        else:
            return images, labels    
    
    def sample_slices(self, sample_size: int, range_: tuple[float, float] = (0.2, 0.8)) -> Dataset:
        
        # Get the number of slices to sample from each volume
        n_slices_per_volume = distribute_n_in_m_slots(sample_size, self.num_vols)
        
        # Sample the idxs of the slices
        min_slice_idx, max_slice_idx = int(self.dim_proc[0] * range_[0]), int(self.dim_proc[0] * range_[1])
        sampled_slices = [np.random.randint(idx_start * min_slice_idx, (idx_start + 1) * max_slice_idx, n_slices) 
                          for idx_start, n_slices in enumerate(n_slices_per_volume)]
        sampled_slices = list(np.concatenate(sampled_slices))
    
        return Subset(self, sampled_slices)
    
    def _find_min_max_in_normalized_imgs(self):
        if self.normalized_img_path is not None:    
            with h5py.File(self.normalized_img_path, 'r') as data:
                images = data['images'][:]
                images_min = images.min()
                images_max = images.max()
        
        elif self.norm is not None:
            print('Determining min and max values of normalized images from a sample of the dataset')
            self.images_min, self.images_max = np.inf, -np.inf
            sample_dataset = self.sample_slices(min(256, self.__len__()))
            sample_dataset = DataLoader(sample_dataset, batch_size=4, num_workers=2)
            
            for img, _ in sample_dataset:
                img = img.to(self.norm_device)
                img = self.norm(img)
                current_max, current_min = img.max(), img.min()
                if current_min < self.images_min:
                    images_min = current_min
                if current_max > self.images_max:
                    images_max = current_max
                
        else: 
            raise ValueError('No normalization model or normalized images were given, '
                             'thus the min and max values of normalized images cannot be determined')
            
        return images_min, images_max 

    def _find_min_max_in_original_imgs(self):
        return self.images.min(), self.images.max() 
    
    # Create function that samples cuts based on a given volume and cut id
    def get_related_images(
        self, 
        vol_idx: int, z_idx: int,
        mode: str = 'different_same_patient', 
        n: int = 10,
        **kwargs):
        
        valid_modes = [
            'same_patient_very_different_labels', 'same_patient_similar_labels', 
            'different_patient_similar_labels', 'none'] 
        
        assert mode in valid_modes, "Mode of sampling cuts not implemented"

        random.seed(self.seed)
        if mode == 'same_patient_very_different_labels':
            idxs = self._get_same_patient_very_different_labels(
                vol_idx, z_idx, n,
                **kwargs
            )
            
        elif mode == 'same_patient_similar_labels':
            idxs = self._get_same_patient_similar_labels(
                vol_idx, z_idx, n,
                **kwargs
            )
            
        elif mode == 'different_patient_similar_labels':
            idxs = self._get_different_patient_similar_labels(
                vol_idx, z_idx, n,
                **kwargs
            )
            
        elif mode == 'none':
            idxs = [self.vol_and_z_idx_to_idx(vol_idx, z_idx)]
        
        else:
            raise ValueError('Mode not implemented')
         
        return Subset(self, idxs)
    
    def _get_same_patient_very_different_labels(
        self,
        vol_idx: int, z_idx: int,
        n: int,
        min_dist_z_frac: float = 0.2, 
        max_dice_score_threshold: float = 0.3,
    ) -> list[int]:
        
        low_z_lim = max(0, z_idx - int(min_dist_z_frac * self.dim_proc[0]))
        high_z_lim = min(self.dim_proc[0] - 1, z_idx + int(min_dist_z_frac * self.dim_proc[0]))
        
        # Sample positions in the range [high_z_lim, high_z_lim + low_z_lim] 
        idxs_sample = random.sample(range(high_z_lim, self.dim_proc[0] + low_z_lim), n)
        idxs_sample = [idx % self.dim_proc[0] for idx in idxs_sample]
               
        # Exclude all indexes that have a dice score that is too high
        idxs_sample_filtered = []

        _, seg_orig = self[self.vol_and_z_idx_to_idx(vol_idx, z_idx)]
        seg_orig = (seg_orig * (self.n_classes - 1)).int()
        for z_idx in idxs_sample:
            img_idx = self.vol_and_z_idx_to_idx(vol_idx, z_idx)
            _, seg_i = self[img_idx]
            seg_i = (seg_i * (self.n_classes - 1)).int()
            
            if dice(seg_i, seg_orig, ignore_index=0).item() > max_dice_score_threshold:
                continue
            
            idxs_sample_filtered.append(img_idx)
            
        if len(idxs_sample_filtered) < n:
            print(f'WARNING: Could not sample enough slices from volume {vol_idx} \n',
                  f'Sampled {len(idxs_sample_filtered)} images')
        
        return idxs_sample_filtered
            
    def _get_same_patient_similar_labels(
        self,
        vol_idx: int, z_idx: int,
        n: int,
        max_dist_z_frac: float = 0.2,
        min_dice_score_threshold: float = 0.7,
    ) -> list[int]:
        low_z_lim = max(0, z_idx - int(max_dist_z_frac * self.dim_proc[0]))
        high_z_lim = min(self.dim_proc[0] - 1, z_idx + int(max_dist_z_frac * self.dim_proc[0]))
        
        # Sample positions in the range [low_z_lim, high_z_lim] 
        idxs_sample = random.sample(range(low_z_lim, high_z_lim), n)
        
        # Exclude all indexes that have a dice score that is too low
        idxs_sample_filtered = []

        _, seg_orig = self[self.vol_and_z_idx_to_idx(vol_idx, z_idx)]
        seg_orig = (seg_orig * (self.n_classes - 1)).int()
        for z_idx in idxs_sample:
            img_idx = self.vol_and_z_idx_to_idx(vol_idx, z_idx)
            _, seg_i = self[img_idx]
            seg_i = (seg_i * (self.n_classes - 1)).int()
            
            if dice(seg_i, seg_orig) < min_dice_score_threshold:
                print('Dice score too low to use sample: ', dice(seg_i, seg_orig, ignore_index=0))
                continue
            
            idxs_sample_filtered.append(img_idx)
            
        return idxs_sample_filtered
    
    
    def _get_different_patient_similar_labels(
        self,
        vol_idx: int, z_idx: int,
        n: int,
        max_dist_z_frac: float = 0.2,
        min_dice_score_threshold: float = 0.55,
    ) -> list[int]:
        # Define the range of z indexes to sample from
        low_z_lim = max(0, z_idx - int(max_dist_z_frac * self.dim_proc[0]))
        high_z_lim = min(self.dim_proc[0] - 1, z_idx + int(max_dist_z_frac * self.dim_proc[0]))
        
        # Sample other volumes
        #  Get the number of slices to sample from the other volumes
        n_slices_per_volume = distribute_n_in_m_slots(n, self.num_vols - 1)
        vol_idxs_to_sample = list(range(self.num_vols))
        vol_idxs_to_sample.remove(vol_idx)
                
        n_slices_per_volume = {vol_idx: n_slices for vol_idx, n_slices in zip(vol_idxs_to_sample, n_slices_per_volume)}
        
        _, seg_orig = self[self.vol_and_z_idx_to_idx(vol_idx, z_idx)]
        seg_orig = (seg_orig * (self.n_classes - 1)).int()
        idxs_sample_filtered = []
        for vol_idx, n_slices in n_slices_per_volume.items():
            if n_slices == 0:
                continue
            
            sampled = 0 
            for z_idx in range(low_z_lim, high_z_lim + 1):
                img_idx = self.vol_and_z_idx_to_idx(vol_idx, z_idx)
                _, seg_i = self[img_idx]
                seg_i = (seg_i * (self.n_classes - 1)).int()
                
                if dice(seg_i, seg_orig, ignore_index=0) < min_dice_score_threshold:
                    #print('Dice score too low to use sample: ', dice(seg_i, seg_orig, ignore_index=0))
                    continue
                
                idxs_sample_filtered.append(img_idx)
                sampled += 1
                
                if sampled >= n_slices:
                    break
            
            if sampled < n_slices:
                print(f'WARNING: Could not sample enough slices from volume {vol_idx}')
            
        return idxs_sample_filtered
    
    def vol_and_z_idx_to_idx(self, vol_idx: int, z_idx: int) -> int:
        if self.image_size[0] != 1:
            raise ValueError('Indexes of the dataset map to volumes, not images')
        
        idx = vol_idx * self.dim_proc[0] + z_idx
        
        assert idx < self.images.shape[0], f'Index out of bounds, for vol_idx: {vol_idx}, z_idx: {z_idx}, dim_proc: {self.dim_proc}'
        
        return vol_idx * self.dim_proc[0] + z_idx

            
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
        split           = 'train',
        one_hot_encode  = True, 
        normalize       = 'min_max',
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
    
    img, seg = ds[125]
    assert img.shape == (1, 256, 256), 'Image should have shape (1, 256, 256)'
    assert img.max() <= 1, 'Image should be normalized to [0, 1] range'
    assert img.min() >= 0, 'Image should be normalized to [0, 1] range'
    assert seg.max() <= 1, 'Segmentation should be normalized to [0, 1] range'
    assert seg.min() >= 0, 'Segmentation should be normalized to [0, 1] range'
    assert seg.shape == (15, 256, 256), 'Segmentation should have shape (15, 256, 256)'
    
    ds.get_related_images(vol_idx=5, z_idx=120, mode='different_patient_similar_labels', 
                          n=10)
    
