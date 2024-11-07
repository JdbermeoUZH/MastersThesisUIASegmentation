import json
from typing import Optional

import h5py
import numpy as np
from skimage import filters, morphology
from scipy import ndimage
from skimage.transform import rescale
import torch
from torch.utils import data
import nibabel as nib
import nibabel.processing as nibp

from tta_uia_segmentation.src.dataset.augmentation import apply_data_augmentation
from tta_uia_segmentation.src.dataset.deformation import make_noise_masks_3d
from tta_uia_segmentation.src.models.seg.norm_seg.normalization import RBF
from tta_uia_segmentation.src.utils.io import deep_get
from tta_uia_segmentation.src.utils.loss import class_to_onehot
from tta_uia_segmentation.src.utils.utils import get_seed, assert_in, resize_and_resample_nibp


def split_dataset(dataset, ratio):
    ratio = np.array(ratio)
    ratio = np.floor(ratio * len(dataset) / np.sum(ratio)).astype(int)

    n_remaining = len(dataset) - np.sum(ratio)

    remainder = np.ones_like(ratio) * (n_remaining // len(ratio))
    remainder[:n_remaining % len(ratio)] += 1

    ratio += remainder

    assert sum(ratio) == len(dataset)
    
    return data.random_split(dataset, ratio)


def get_datasets(
    splits: list[str],
    h5_filepath: str,
    fold: int,
    n_classes: int,
    rescale_factor: Optional[tuple[float, float, float]] = None,
    image_size: Optional[tuple[int, int, int]] = None,
    voxel_size: Optional[tuple[float, float, float]] = None,
    aug_params: Optional[dict] = None,
    deformation: Optional[dict] = None,
    image_transform: str = 'none',
    image_transform_args: dict = {},
    bg_suppression_opts: dict = {},
    seed: int = None,
):

    datasets = []

    for split in splits:
        datasets.append(
            DatasetH5(
                h5_filepath=h5_filepath,
                fold=fold,
                split=split,
                n_classes=n_classes,
                rescale_factor=rescale_factor,
                image_size=image_size,
                voxel_size=voxel_size,
                aug_params=aug_params,
                deformation=deformation,
                image_transform=image_transform,
                image_transform_args=image_transform_args,
                bg_suppression_opts=bg_suppression_opts,
                seed=seed,
                )
            )

    return datasets


def get_sectors_from_index(relative_index, sector_size=None, n_sectors=None):
    assert any([
        n_sectors is None and sector_size is not None,
        n_sectors is not None and sector_size is None
    ]), 'specify either n_sectors or sector_size and only one of them can be specified'

    if sector_size is None:
        sector_size = 1 / n_sectors

    sectors = torch.div(relative_index, sector_size, rounding_mode='floor')
    return sectors.long()


class AugmentationNetwork(torch.nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels, kernel_size):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, hidden_channels, kernel_size),
            RBF(hidden_channels),
            torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size),
            RBF(hidden_channels),
            torch.nn.Conv2d(hidden_channels, output_channels, kernel_size),
        )

    def reset_parameters(self):
        for m in self.net:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, x):
        return self.net(x)


class DatasetH5(data.Dataset):
    def __init__(
        self,
        h5_filepath,
        fold: int,
        split: list[str],
        n_classes: int,
        rescale_factor: Optional[tuple[float, float, float]] = None,
        image_size: Optional[tuple[int, int, int]] = None,
        voxel_size: Optional[tuple[float, float, float]] = None,
        aug_params: Optional[dict] = None,
        deformation: Optional[dict] = None,
        image_transform: str = 'none',
        image_transform_args: dict = {},
        bg_suppression_opts: dict = {},
        seed: int = None,
    ):

        assert_in(split, 'split', ['train', 'train_dev', 'val_dev', 'test'])
        assert_in(image_transform, 'image_transform', ['none', 'random_net'])
        assert not (rescale_factor is not None and voxel_size is not None), 'Specify either rescale_factor or voxel_size, not both'

        self.h5_filepath = h5_filepath
        self.split = split
        self.fold = fold
        self.rescale_factor = rescale_factor
        self.image_size = image_size
        self.voxel_size = voxel_size
        self.n_classes = n_classes
        self.aug_params = aug_params
        self.deformation = deformation
        self.augmentation = not aug_params is None
        self.image_transform = image_transform
        self.bg_suppression_opts = bg_suppression_opts
        self.seed = seed

        
        # Store the dictionaries of the indexes of the split and fold requested
        with h5py.File(self.h5_filepath, 'r') as h5f:
            ids = h5f['ids'][:]
            indexes_of_ids_in_fold = h5f[f'folds/fold_{self.fold}/{self.split}_idx'][:]
            
            self.index_to_id = {new_index: ids[idx_id] for new_index, idx_id in enumerate(indexes_of_ids_in_fold)}
            self.id_to_index = {id: new_index for new_index, id in self.index_to_id.items()}
            
            self.label_names = json.loads(h5f['label_names'][()])
                    
        # Initialize random net for augmentation 
        if image_transform == 'random_net':
            hidden_channels = deep_get(image_transform_args, 'hidden_channels')
            kernel_size = deep_get(image_transform_args, 'kernel_size')
            self.random_net = AugmentationNetwork(1, 1, hidden_channels, kernel_size)
            self.random_net.eval()

            self.only_foreground = deep_get(image_transform_args, 'only_foreground')


    def scale_to_original_size(self, image, index, interpolation_order=None):
        """
        Do not need this method for the time being. Check original code if needed again
        """
        raise NotImplementedError
        

    def image_transformation(self, images):
        if len(images.unique()) == 1:
            return images

        assert_in(self.image_transform, 'self.image_transform',
                  ['none', 'random_net'])
        if self.image_transform == 'none':
            return images

        elif self.image_transform == 'random_net':
            self.random_net.reset_parameters()
            with torch.no_grad():
                img_aug = self.random_net(images.unsqueeze(1)).reshape(images.shape)
                img_aug -= img_aug.min()
                img_aug /= img_aug.max()

                if self.only_foreground:
                    img_aug[images == 0] = 0

            return img_aug


    def set_augmentation(self, augmentation):
        self.augmentation = augmentation

    def set_seed(self, seed=None):
        self.seed = seed

    def __len__(self):
        return len(self.index_to_id)

    def get_background_mask(self, images, labels):
        """
        Get estimated background mask from images
        
        If the mask_source is 'ground_truth' (from bg_suppression_opts),
        then the background mask is the ground truth segmentation.
        
        Otherwise it is the mask obtained by the specified thresholding method in bg_suppression_opts.
        """
        mask_source = deep_get(self.bg_suppression_opts, 'mask_source', default='ground_truth')
        assert_in(mask_source, 'mask_source', ['ground_truth', 'thresholding'])

        if mask_source == 'ground_truth':
            return labels == 0
        
        assert images.shape == labels.shape, f'Unequal shapes: images.shape: {images.shape}, labels.shape: {labels.shape}'
        shape = images.shape
        images = images.reshape(-1, *shape[-2:])
        labels = labels.reshape(-1, *shape[-2:])
        
        thresholding = deep_get(self.bg_suppression_opts, 'thresholding', default='otsu')
        assert_in(
            thresholding, 'thresholding',
            ['isodata', 'li', 'mean', 'minimum', 'otsu', 'triangle', 'yen'],
        )

        if thresholding == 'isodata':
            threshold = filters.threshold_isodata(images)
        elif thresholding == 'li':
            threshold = filters.threshold_li(images)
        elif thresholding == 'mean':
            threshold = filters.threshold_mean(images)
        elif thresholding == 'minimum':
            threshold = filters.threshold_minimum(images)
        elif thresholding == 'otsu':
            threshold = filters.threshold_otsu(images)
        elif thresholding == 'triangle':
            threshold = filters.threshold_triangle(images)
        elif thresholding == 'yen':
            threshold = filters.threshold_yen(images)

        fg_mask = images > threshold

        hole_filling = deep_get(self.bg_suppression_opts, 'hole_filling', default=True)
        if hole_filling:
            for i in range(fg_mask.shape[0]):
                fg_mask[i] = morphology.binary_dilation(fg_mask[i])
                fg_mask[i] = ndimage.binary_fill_holes(fg_mask[i])

        fg_mask = fg_mask.reshape(shape)

        bg_mask = ~fg_mask
        return bg_mask


    def __getitem__(self, index):
        with h5py.File(self.h5_filepath, 'r') as h5f:
            img_group = h5f['data'][self.index_to_id[index]]
            image = img_group['tof'][:]
            labels = img_group['seg'][:]
            px = img_group['px'][()]
            py = img_group['py'][()]
            pz = img_group['pz'][()]
            
            orig_voxel_size = (pz, px, py)       # Images are assumed as stored in DHW 
            target_voxel_size = None

        seed = get_seed() if self.seed is None else self.seed
               
        if self.rescale_factor is not None:
            target_voxel_size = np.array(orig_voxel_size) * (1 / np.array(self.rescale_factor))
        
        elif self.voxel_size is not None:
            target_voxel_size = self.voxel_size
                
        if target_voxel_size is not None or self.image_size is not None:
            # Resize and resmple image and labels, if needed
            target_voxel_size = orig_voxel_size if target_voxel_size is None else target_voxel_size
            target_size = image.shape if self.image_size is None else self.image_size
            image = resize_and_resample_nibp(image, target_size, orig_voxel_size, target_voxel_size, order=3)
            labels = resize_and_resample_nibp(labels, target_size, orig_voxel_size, target_voxel_size, order=0)
                
        assert image.shape == labels.shape, 'Image and label shape not matching'
        
        background_mask = self.get_background_mask(image, labels)
                     
        if self.augmentation:
            image, labels, background_mask = apply_data_augmentation(
                image,
                labels,
                background_mask,
                **self.aug_params,
                rng=np.random.default_rng(seed),
            )
            
        background_mask = (background_mask == 1)

        if self.deformation is not None:
            labels_deformed = make_noise_masks_3d(
                self.deformation['mask_type'],
                self.deformation['mask_radius'],
                self.deformation['mask_squares'],
                self.n_classes,
                labels,
                self.deformation['is_num_masks_fixed'],
                self.deformation['is_size_masks_fixed'],
            )
        else:
            labels_deformed = torch.tensor([])

        image = torch.from_numpy(image)
        image = self.image_transformation(image)

        labels = torch.from_numpy(labels)
        labels = class_to_onehot(labels, self.n_classes, class_dim=0)

        if self.deformation is not None:
            labels_deformed = torch.from_numpy(labels_deformed)
            labels_deformed = class_to_onehot(labels_deformed, self.n_classes, class_dim=0)

        return image, labels, labels_deformed, index, background_mask

    def get_label_name(self, label_idx):
        return self.label_names[label_idx]