import os
from typing import Literal, Optional

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
from tta_uia_segmentation.src.utils.utils import crop_or_pad_to_size, get_seed, assert_in


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
        self.mode = '2D' if image_size[0] == 1 else '3D'

        with h5py.File(self.path, 'r') as data:

            self.images = data['images'][:].astype(np.float32)
            self.labels = data['labels'][:].astype(np.uint8)

            self.images = self.images.reshape(-1, *self.image_size)  # NDHW or (N*D)CHW
            self.labels = self.labels.reshape(-1, *self.image_size)  # NDHW or (N*D)CHW

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
        # TODO: Most likely remove this function
        # image has shape ...HW
        shape = image.shape

        nx, ny, _ = self.n_pix_original[:, index]
        px, py, pz = self.pix_size_original[:, index]
        px_proc, py_proc, pz_proc = self.resolution_proc

        scale = [pz_proc / pz, px_proc / px, py_proc / py]
        image_rescaled = F.interpolate(image, scale_factor = scale, mode=interpolation_order)

        image_rescaled = image_rescaled.reshape(-1, *image_rescaled.shape[-2:])
        image_rescaled = crop_or_pad_to_size(image_rescaled, (nx, ny))
        image_rescaled = image_rescaled.reshape((*shape[:-3], -1, nx, ny))

        return image_rescaled


    def get_original_images(self, index, as_onehot=True,
                            format: Literal['DCHW', '1CDHW'] = '1CDHW'):
        """
        Get the original image and label volumes at the specified index.

        """
        nx, ny, _ = self.n_pix_original[:, index]
        
        if self.images_original is None or self.labels_original is None:
            self.load_original_images()

        images = self.images_original[index, :, 0:nx, 0:ny]
        labels = self.labels_original[index, :, 0:nx, 0:ny]
        bg_mask = self.background_mask_original[index, :, 0:nx, 0:ny]

        images = torch.from_numpy(images).unsqueeze(0)   # CDHW
        labels = torch.from_numpy(labels).unsqueeze(0)   # CDHW
        bg_mask = torch.from_numpy(bg_mask).unsqueeze(0) # CDHW

        if as_onehot:
            labels = class_to_onehot(labels, self.n_classes, class_dim=1) # 1CDHW

        if format == '1CDHW':
            images = images.unsqueeze(0)
            bg_mask = bg_mask.unsqueeze(0)

            len(images.shape) == 5, f'images.shape: {images.shape}'
            len(labels.shape) == 5, f'labels.shape: {labels.shape}'
            len(bg_mask.shape) == 5, f'bg_mask.shape: {bg_mask.shape}'

        elif format == 'DCHW':
            images = images.permute(1, 0, 2, 3)
            labels = labels.squeeze(0).permute(1, 0, 2, 3)
            bg_mask = bg_mask.permute(1, 0, 2, 3)

            len(images.shape) == 4, f'images.shape: {images.shape}'
            len(labels.shape) == 4, f'labels.shape: {labels.shape}'
            len(bg_mask.shape) == 4, f'bg_mask.shape: {bg_mask.shape}'
        
        else:
            raise ValueError(f'Unknown format: {format}')

        return images, labels, bg_mask

    
    def get_preprocessed_images(self, index, as_onehot=True,
                                format: Literal['DCHW', '1CDHW'] = '1CDHW'):
        """
        Get the preprocessed image and label volumes at the specified index.
        
        """
        
        if self.mode == '2D':
            images = self.images.reshape(-1, *self.dim_proc)[index]
            labels = self.labels.reshape(-1, *self.dim_proc)[index]
            bg_mask = self.background_mask.reshape(-1, *self.dim_proc)[index]

        else:
            images = self.images[index, ...]
            labels = self.labels[index, ...]
            bg_mask = self.background_mask[index, ...]

        images = torch.from_numpy(images).unsqueeze(0)   # CDHW
        labels = torch.from_numpy(labels).unsqueeze(0)   # CDHW
        bg_mask = torch.from_numpy(bg_mask).unsqueeze(0) # CDHW

        if as_onehot:
            labels = class_to_onehot(labels, self.n_classes, class_dim=1) # 1CDHW

        if format == '1CDHW':
            images = images.unsqueeze(0)
            bg_mask = bg_mask.unsqueeze(0)

            len(images.shape) == 5, f'images.shape: {images.shape}'
            len(labels.shape) == 5, f'labels.shape: {labels.shape}'
            len(bg_mask.shape) == 5, f'bg_mask.shape: {bg_mask.shape}'

        elif format == 'DCHW':
            images = images.permute(1, 0, 2, 3)
            labels = labels.squeeze(0).permute(1, 0, 2, 3)
            bg_mask = bg_mask.permute(1, 0, 2, 3)

            len(images.shape) == 4, f'images.shape: {images.shape}'
            len(labels.shape) == 4, f'labels.shape: {labels.shape}'
            len(bg_mask.shape) == 4, f'bg_mask.shape: {bg_mask.shape}'

        else:
            raise ValueError(f'Unknown format: {format}')

        return images, labels, bg_mask


    def get_original_pixel_size(self, index, orientation: str = 'DHW'):
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
        x, y, z = self.pix_size_original[:, index]

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
    


if __name__ == '__main__':
    import sys

    root_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

    sys.path.append(root_dir)
    
    from tta_uia_segmentation.src.utils.io import load_config
    from tta_uia_segmentation.src.utils.utils import resize_volume

    dataset_config = load_config(os.path.join(root_dir, 'config', 'datasets.yaml'))

    split = 'train'
    dataset = 'nuhs_w_synthseg_labels'
    image_size = [1, 256, 256]

    dataset_info = dataset_config[dataset]

    aug_params = {
        'da_ratio': 0.25,
        'sigma': 20,
        'alpha': 0,
        'trans_min': 0,
        'trans_max': 0,
        'rot_min': 0,
        'rot_max': 0,
        'scale_min': 1.0,
        'scale_max': 1.0,
        'gamma_min': 0.5,
        'gamma_max': 2.0,
        'brightness_min': 0.0,
        'brightness_max': 0.1,
        'noise_mean': 0.0,
        'noise_std': 0.1
    }

    bg_suppression_opts = {
        "type": "fixed_value",  # possible values: none, fixed_value, random_value
        "bg_value": -0.5,
        "bg_value_min": -0.5,
        "bg_value_max": 1,
        "mask_source": "thresholding",  # possible values: thresholding, ground_truth
        "thresholding": "otsu",  # possible values: isodata, li, mean, minimum, otsu, triangle, yen
        "hole_filling": True
    }

    ds, = get_datasets(
        paths           = dataset_info['paths_processed'],
        paths_original  = dataset_info['paths_original'],
        splits          = [split],
        image_size      = image_size,
        resolution_proc = dataset_info['resolution_proc'],
        dim_proc        = dataset_info['dim'],
        n_classes       = dataset_info['n_classes'],
        aug_params      = aug_params,
        deformation     = None,
        load_original   = True,
        bg_suppression_opts=bg_suppression_opts,
    )

    vol_index = 0
    x_orig, y_orig, bg_mask = ds.get_original_images(vol_index)
    print(x_orig.shape, y_orig.shape, bg_mask.shape)

    x_proc, y_proc, bg_mask = ds.get_preprocessed_images(vol_index)
    print(x_proc.shape, y_proc.shape, bg_mask.shape)

    print(f'Original pixel size: {ds.get_original_pixel_size(vol_index)}')
    print(f'Processed pixel size: {ds.get_processed_pixel_size()}')

    # Test resizing of preprocessed volume to original size
    x_proc_resized = resize_volume(
        x_proc,
        current_pix_size= ds.get_processed_pixel_size(),
        target_pix_size=ds.get_original_pixel_size(vol_index),
        target_img_size=tuple(x_orig.shape[-3:]),
    )
    print(x_proc_resized.shape)

    # Test resizing of label volume to preprocessed size
    y_proc_resized = resize_volume(
        y_proc.float(),
        current_pix_size= ds.get_processed_pixel_size(), 
        target_pix_size=ds.get_original_pixel_size(vol_index),
        target_img_size=tuple(y_orig.shape[-3:])
    )
    print(y_proc_resized.shape)


    # Test resizing of original volume to preprocessed size
    x_orig_resized = resize_volume(
        x_orig, current_pix_size=ds.get_original_pixel_size(vol_index), 
        target_pix_size=ds.get_processed_pixel_size(),
        target_img_size=ds.dim_proc 
        )
    print(x_orig_resized.shape)

    # Test resizing of label volume to preprocessed size
    y_orig_resized = resize_volume(
        y_orig.float(), 
        ds.get_original_pixel_size(vol_index), 
        ds.get_processed_pixel_size(),
        ds.dim_proc)
    print(y_orig_resized.shape)

    print('hallo')