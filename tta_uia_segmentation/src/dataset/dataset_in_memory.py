import os
from typing import Optional

import h5py
import numpy as np
from scipy import ndimage
from skimage import filters, morphology
from skimage.transform import rescale
import torch
import torch.nn.functional as F
from torch.utils import data

from tta_uia_segmentation.src.dataset.augmentation import apply_data_augmentation
from tta_uia_segmentation.src.dataset.deformation import make_noise_masks_3d
from tta_uia_segmentation.src.dataset.utils import transform_orientation
from tta_uia_segmentation.src.models.normalization import RBF
from tta_uia_segmentation.src.utils.io import deep_get
from tta_uia_segmentation.src.utils.loss import class_to_onehot
from tta_uia_segmentation.src.utils.utils import crop_or_pad_slice_to_size, get_seed, assert_in


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
    paths,
    paths_original,
    splits,
    image_size,
    resolution_proc,
    dim_proc,
    n_classes,
    rescale_factor=None,
    aug_params=None,
    deformation=None,
    load_original=False,
    image_transform='none',
    image_transform_args=None,
    bg_suppression_opts=None,
    seed=None,
):

    datasets = []

    for split in splits:
        datasets.append(DatasetInMemory(
            paths,
            paths_original,
            split,
            image_size,
            resolution_proc,
            dim_proc,
            n_classes,
            rescale_factor,
            aug_params,
            deformation,
            load_original,
            image_transform,
            image_transform_args,
            bg_suppression_opts=bg_suppression_opts,
            seed=seed,
        ))

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


class DatasetInMemory(data.Dataset):
    def __init__(
        self,
        paths,
        paths_original,
        split,
        image_size,
        resolution_proc,
        dim_proc,
        n_classes,
        rescale_factor=None,
        aug_params=None,
        deformation=None,
        load_original=False,
        image_transform='none',
        image_transform_args={},
        bg_suppression_opts={},
        seed=None,
        label_names: Optional[dict] = None
    ):

        assert_in(split, 'split', ['train', 'val', 'test'])
        assert_in(image_transform, 'image_transform', ['none', 'random_net'])
        self.path = os.path.expanduser(paths[split])
        self.path_original = os.path.expanduser(paths_original[split])
        self.split = split
        self.image_size = image_size
        self.resolution_proc = resolution_proc
        self.rescale_factor = rescale_factor
        self.dim_proc = dim_proc
        self.n_classes = n_classes
        self.aug_params = aug_params
        self.deformation = deformation
        self.augmentation = not aug_params is None
        self.image_transform = image_transform
        self.bg_suppression_opts = bg_suppression_opts
        self.seed = seed

        with h5py.File(self.path, 'r') as data:

            self.images = data['images'][:].astype(np.float32)
            self.labels = data['labels'][:].astype(np.uint8)

            self.images = self.images.reshape(-1, *self.image_size)  # NDHW
            self.labels = self.labels.reshape(-1, *self.image_size)  # NDHW

            # Pixel sizes of original images.
            self.pix_size_original = np.stack([data['px'], data['py'], data['pz']])

        if rescale_factor is not None:
            self.dim_proc = (self.dim_proc * np.array(rescale_factor)).astype(int).tolist()

            self.images = torch.from_numpy(self.images).unsqueeze(0)
            self.labels = torch.from_numpy(self.labels).unsqueeze(0)

            self.images = F.interpolate(self.images, scale_factor=rescale_factor, mode='trilinear')
            self.labels = F.interpolate(self.labels, scale_factor=rescale_factor, mode='nearest')

            self.images = self.images.squeeze(0).numpy()
            self.labels = self.labels.squeeze(0).numpy()

        assert self.images.shape == self.labels.shape, 'Image and label shape not matching'

        if load_original:
            self.load_original_images()
        else:
            self.labels_original = None
            self.images_original = None

        if image_transform == 'random_net':
            hidden_channels = deep_get(image_transform_args, 'hidden_channels')
            kernel_size = deep_get(image_transform_args, 'kernel_size')
            self.random_net = AugmentationNetwork(1, 1, hidden_channels, kernel_size)
            self.random_net.eval()

            self.only_foreground = deep_get(image_transform_args, 'only_foreground')

        self.background_mask = self.get_background_mask(self.images, self.labels)
        
        if label_names is None:
            label_names = {i: str(i) for i in range(n_classes)}
        else:
            assert len(label_names) == n_classes, 'len(label_names) != n_classes'
        
        self.label_names = label_names


    def load_original_images(self):
        with h5py.File(self.path_original, 'r') as data:
            self.images_original = data['images'][:].astype(np.float32)
            self.labels_original = data['labels'][:].astype(np.uint8)
            
            num_volumes = len(data['nx'])
            inplane_shape = self.images_original.shape[-2:]
            self.images_original = self.images_original.reshape(num_volumes, -1, *inplane_shape)  # NDHW
            self.labels_original = self.labels_original.reshape(num_volumes, -1, *inplane_shape)  # NDHW

            # Number of pixels for original images.
            self.n_pix_original = np.stack([data['nx'], data['ny'], data['nz']])

        self.background_mask_original = self.get_background_mask(self.images_original, self.labels_original)


    def get_volume_indices(self):
        n_slices = len(self)
        n_volumes = n_slices // self.dim_proc[0]
        n_slices_per_volume = np.array([self.dim_proc[0]] * n_volumes)

        indices = np.arange(n_slices)

        indices_per_volume = np.split(indices, n_slices_per_volume.cumsum()[:-1])

        return indices_per_volume


    def scale_to_original_size(self, image, index, interpolation_order=None):

        # image has shape ...HW
        shape = image.shape

        nx, ny, _ = self.n_pix_original[:, index]
        px, py, pz = self.pix_size_original[:, index]
        px_proc, py_proc, pz_proc = self.resolution_proc

        scale = [pz_proc / pz, px_proc / px, py_proc / py]
        image_rescaled = F.interpolate(image, scale_factor = scale, mode=interpolation_order)

        image_rescaled = image_rescaled.reshape(-1, *image_rescaled.shape[-2:])
        image_rescaled = crop_or_pad_slice_to_size(image_rescaled, nx, ny)
        image_rescaled = image_rescaled.reshape((*shape[:-3], -1, nx, ny))

        return image_rescaled


    def get_original_images(self, index, as_onehot=True):
        nx, ny, _ = self.n_pix_original[:, index]
        
        if self.images_original is None or self.labels_original is None:
            self.load_original_images()

        images = self.images_original[index, :, 0:nx, 0:ny]
        labels = self.labels_original[index, :, 0:nx, 0:ny]

        bg_mask = self.background_mask_original[index, :, 0:nx, 0:ny]

        images = torch.from_numpy(images).unsqueeze(0)
        labels = torch.from_numpy(labels).unsqueeze(0)

        if as_onehot:
            labels = class_to_onehot(labels, self.n_classes, class_dim=1)

        return images, labels, bg_mask


    def get_original_pixel_size(self, orientation: str = 'DHW'):
        """
        Get the original pixel size in the specified orientation.

        Parameters
        ----------
        orientation : str, optional
            The desired orientation of the pixel size. Default is 'DHW'.

        Returns
        -------
        tuple
            The pixel size in the specified orientation.
        """
        x, y, z = self.pix_size_original

        return transform_orientation(x, y, z, orientation)


    def get_processed_pixel_size(self, orientation: str = 'DHW'):
        """
        Get the processed pixel size in the specified orientation.

        Parameters
        ----------
        orientation : str, optional
            The desired orientation of the pixel size. Default is 'DHW'.

        Returns
        -------
        tuple
            The pixel size in the specified orientation.
        """
        x, y, z = self.resolution_proc

        return transform_orientation(x, y, z, orientation)
       

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


    def set_augmentation(self, augmentation: bool):
        self.augmentation = augmentation
        
    def get_augmentation(self) -> bool:
        return self.augmentation

    def set_seed(self, seed=None):
        self.seed = seed

    def __len__(self):
        return self.images.shape[0]
    
    def idx_to_slice_idx(self, idx):
        idx = (idx / self.dim_proc[-1]) % 1
        return idx


    def get_background_mask(self, images, labels):

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

        images = torch.from_numpy(images)
        images = self.image_transformation(images)

        labels = torch.from_numpy(labels)
        labels = class_to_onehot(labels, self.n_classes, class_dim=0)

        if self.deformation is not None:
            labels_deformed = torch.from_numpy(labels_deformed)
            labels_deformed = class_to_onehot(labels_deformed, self.n_classes, class_dim=0)

        if self.image_size[0] == 1:
            index = self.idx_to_slice_idx(index)

        return images, labels, labels_deformed, index, background_mask

    def get_label_name(self, label):
        return self.label_names[label]