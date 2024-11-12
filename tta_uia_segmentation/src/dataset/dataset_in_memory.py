import os
import math
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
from tta_uia_segmentation.src.models.seg.norm_seg.normalization import RBF
from tta_uia_segmentation.src.utils.io import deep_get
from tta_uia_segmentation.src.utils.loss import class_to_onehot
from tta_uia_segmentation.src.utils.utils import (
    crop_or_pad_to_size,
    get_seed,
    assert_in,
    resize_volume,
)


def split_dataset(dataset, ratio):
    ratio = np.array(ratio)
    ratio = np.floor(ratio * len(dataset) / np.sum(ratio)).astype(int)

    n_remaining = len(dataset) - np.sum(ratio)

    remainder = np.ones_like(ratio) * (n_remaining // len(ratio))
    remainder[: n_remaining % len(ratio)] += 1

    ratio += remainder

    assert sum(ratio) == len(dataset)

    return data.random_split(dataset, ratio)


def get_datasets(splits, *args, **kwargs):

    return [DatasetInMemory(split=split, *args, **kwargs) for split in splits]


def get_sectors_from_index(relative_index, sector_size=None, n_sectors=None):
    assert any(
        [
            n_sectors is None and sector_size is not None,
            n_sectors is not None and sector_size is None,
        ]
    ), "specify either n_sectors or sector_size and only one of them can be specified"

    if sector_size is None:
        sector_size = 1 / n_sectors

    sectors = torch.div(relative_index, sector_size, rounding_mode="floor")
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
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(self, x):
        return self.net(x)


class DatasetInMemory(data.Dataset):
    def __init__(
        self,
        dataset_name,
        paths,
        paths_original,
        split,
        resolution_proc,
        dim_proc,
        n_classes,
        image_size: Optional[tuple[int]] = None,
        rescale_factor=None,
        aug_params=None,
        deformation=None,
        load_original=False,
        image_transform=None,
        image_transform_args={},
        seed=None,
        label_names: Optional[dict] = None,
    ):

        assert_in(split, "split", ["train", "val", "test"])
        assert_in(image_transform, "image_transform", [None, "random_net"])
        self.dataset_name = dataset_name
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
        self.apply_deformation = deformation is not None
        self.apply_augmentation = aug_params is not None
        self.image_transform = image_transform
        self.apply_image_transform = (
            image_transform != "none" and image_transform is not None
        )
        self.seed = seed
        self.mode = "2D" if image_size is None or image_size[-3] == 1 else "3D"

        with h5py.File(self.path, "r") as data:

            self.images = data["images"][:].astype(np.float32)
            self.labels = data["labels"][:].astype(np.uint8)

            if self.image_size is not None:
                self.images = self.images.reshape(-1, *self.image_size)  # NDHW or (N*D)CHW
                self.labels = self.labels.reshape(-1, *self.image_size)  # NDHW or (N*D)CHW

            # Pixel sizes of original images.
            self.pix_size_original = np.stack([data["px"], data["py"], data["pz"]])

        if rescale_factor is not None:
            self.dim_proc = (
                (self.dim_proc * np.array(rescale_factor)).astype(int).tolist()
            )

            self.images = torch.from_numpy(self.images).unsqueeze(0)
            self.labels = torch.from_numpy(self.labels).unsqueeze(0)

            self.images = F.interpolate(
                self.images, scale_factor=rescale_factor, mode="trilinear"
            )
            self.labels = F.interpolate(
                self.labels, scale_factor=rescale_factor, mode="nearest"
            )

            self.images = self.images.squeeze(0).numpy()
            self.labels = self.labels.squeeze(0).numpy()

        if load_original:
            self.load_original_images()
        else:
            self.labels_original = None
            self.images_original = None
            self.n_pix_original = None

        if image_transform == "random_net":
            hidden_channels = deep_get(image_transform_args, "hidden_channels")
            kernel_size = deep_get(image_transform_args, "kernel_size")
            self.random_net = AugmentationNetwork(1, 1, hidden_channels, kernel_size)
            self.random_net.eval()

            self.only_foreground = deep_get(image_transform_args, "only_foreground")

        if label_names is None:
            num_zeros_pad = math.ceil(math.log10(n_classes + 1))
            label_names = {i: str(i).zfill(num_zeros_pad) for i in range(n_classes)}
        else:
            assert len(label_names) == n_classes, "len(label_names) != n_classes"

        self.label_names = label_names

    def load_original_images(self):
        with h5py.File(self.path_original, "r") as data:
            self.images_original = data["images"][:].astype(np.float32)
            self.labels_original = data["labels"][:].astype(np.uint8)

            num_volumes = len(data["nx"])
            inplane_shape = self.images_original.shape[-2:]
            self.images_original = self.images_original.reshape(
                num_volumes, -1, *inplane_shape
            )  # NDHW
            self.labels_original = self.labels_original.reshape(
                num_volumes, -1, *inplane_shape
            )  # NDHW

            # Number of pixels for original images.
            self.n_pix_original = np.stack([data["nx"], data["ny"], data["nz"]])

    def get_volume_indices(self) -> list[np.ndarray]:
        """
        Get the indices for each volume in the dataset.

        Returns
        -------
        List[np.ndarray]
            A list of numpy arrays, where each array contains the indices for a
            single volume.
        """
        n_slices = len(self)
        n_volumes = n_slices // self.dim_proc[0]
        n_slices_per_volume = np.array([self.dim_proc[0]] * n_volumes)

        indices = np.arange(n_slices)

        indices_per_volume = np.split(indices, n_slices_per_volume.cumsum()[:-1])

        return indices_per_volume

    def get_idxs_for_volume(self, vol_idx: int) -> np.ndarray:
        """
        Get the indices for the specified volume.

        Parameters
        ----------
        volume_idx : int
            The index of the volume to retrieve.

        Returns
        -------
        np.ndarray
            The indices for the specified volume.
        """
        if self.mode == "2D":
            return self.get_volume_indices()[vol_idx]
        else:
            return np.array([vol_idx])

    def resize_to_original_vol(
        self,
        volume: torch.Tensor,
        index: int,
        match_size: bool = False,
        only_inplane_resample: bool = True,
    ) -> torch.Tensor:
        """
        Resize the processed volume to the original size.

        Parameters
        ----------
        volume : torch.Tensor
            The volume to resize.
        index : int
            The index of the volume in the dataset.
        match_size : bool, optional
            If True, match the size of the original volume exactly.
            Default is False.
        only_inplane_resample : bool, optional
            If True, only resample in-plane. Default is True.

        Returns
        -------
        torch.Tensor
            The resized volume.
        """
        target_size = self.get_original_image_size(index) if match_size else None

        return resize_volume(
            volume,
            current_pix_size=self.get_processed_pixel_size(),
            target_pix_size=self.get_original_pixel_size(index),
            target_img_size=target_size,
            mode="trilinear",
            only_inplane_resample=only_inplane_resample,
        )

    def resize_to_processed_vol(
        self,
        volume: torch.Tensor,
        index: int,
        match_size: bool = False,
        only_inplane_resample: bool = True,
    ) -> torch.Tensor:
        """
        Resample the original images to the processed size.

        Parameters
        ----------
        volume : torch.Tensor
            The volume to resize.
        index : int
            The index of the volume in the dataset.
        match_size : bool, optional
            If True, match the size of the processed volume exactly.
            Default is False.
        only_inplane_resample : bool, optional
            If True, only resample in-plane. Default is True.

        Returns
        -------
        torch.Tensor
            The resized volume.
        """
        target_size = self.dim_proc if match_size else None

        return resize_volume(
            volume,
            current_pix_size=self.get_original_pixel_size(index),
            target_pix_size=self.get_processed_pixel_size(),
            target_img_size=target_size,
            mode="trilinear",
            only_inplane_resample=only_inplane_resample,
        )

    def get_original_images(
        self,
        index: int,
        as_onehot: bool = True,
        format: Literal["DCHW", "1CDHW"] = "1CDHW",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the original image and label volumes at the specified index.

        Parameters
        ----------
        index : int
            The index of the volume to retrieve.
        as_onehot : bool, optional
            If True, return labels in one-hot encoding. Default is True.
        format : {'DCHW', '1CDHW'}, optional
            The desired format of the returned tensors. Default is '1CDHW'.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the image, labels

        Raises
        ------
        ValueError
            If an unknown format is specified.
        """
        nx, ny, _ = self.n_pix_original[:, index]

        if self.images_original is None or self.labels_original is None:
            self.load_original_images()

        images = self.images_original[index, :, 0:nx, 0:ny]
        labels = self.labels_original[index, :, 0:nx, 0:ny]

        images = torch.from_numpy(images).unsqueeze(0)  # CDHW
        labels = torch.from_numpy(labels).unsqueeze(0)  # CDHW

        if as_onehot:
            labels = class_to_onehot(labels, self.n_classes, class_dim=1)  # 1CDHW

        if format == "1CDHW":
            images = images.unsqueeze(0)

            assert len(images.shape) == 5, f"images.shape: {images.shape}"
            assert len(labels.shape) == 5, f"labels.shape: {labels.shape}"

        elif format == "DCHW":
            images = images.permute(1, 0, 2, 3)
            labels = labels.squeeze(0).permute(1, 0, 2, 3)

            assert len(images.shape) == 4, f"images.shape: {images.shape}"
            assert len(labels.shape) == 4, f"labels.shape: {labels.shape}"

        else:
            raise ValueError(f"Unknown format: {format}")

        return images, labels

    def get_preprocessed_images(
        self,
        index: int,
        as_onehot: bool = True,
        format: Literal["DCHW", "1CDHW"] = "1CDHW",
        same_position_as_original: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the preprocessed image and label volumes at the specified index.

        Parameters
        ----------
        index : int
            The index of the volume to retrieve.
        as_onehot : bool, optional
            If True, return labels in one-hot encoding. Default is True.
        format : {'DCHW', '1CDHW'}, optional
            The desired format of the returned tensors. Default is '1CDHW'.
        same_position_as_original : bool, optional
            If True, return the preprocessed images in the same position as the
            original. Default is False.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the preprocessed image, labels
            mask tensors.

        Raises
        ------
        ValueError
            If an unknown format is specified.
        """
        if not same_position_as_original:
            if self.mode == "2D":
                images = self.images.reshape(-1, *self.dim_proc)[index]
                labels = self.labels.reshape(-1, *self.dim_proc)[index]
            else:
                images = self.images[index, ...]
                labels = self.labels[index, ...]

            images = torch.from_numpy(images).unsqueeze(0)  # CDHW
            labels = torch.from_numpy(labels).unsqueeze(0)  # CDHW

            if as_onehot:
                labels = class_to_onehot(labels, self.n_classes, class_dim=1)  # 1CDHW

            if format == "1CDHW":
                images = images.unsqueeze(0)

                assert len(images.shape) == 5, f"images.shape: {images.shape}"
                assert len(labels.shape) == 5, f"labels.shape: {labels.shape}"

            elif format == "DCHW":
                images = images.permute(1, 0, 2, 3)
                labels = labels.squeeze(0).permute(1, 0, 2, 3)

                assert len(images.shape) == 4, f"images.shape: {images.shape}"
                assert len(labels.shape) == 4, f"labels.shape: {labels.shape}"

            else:
                raise ValueError(f"Unknown format: {format}")

        else:
            # Get Original volume image, labels
            #  We do not have the info on the original position, so we have to replicate
            #  the downsampling process done in the original preprocessing (it is the only difference btw. the two anyways)

            images, labels = self.get_original_images(
                index, as_onehot=as_onehot, format=format
            )

            images = self.resize_to_processed_vol(
                images, index, match_size=False, only_inplane_resample=True
            )
            labels = self.resize_to_processed_vol(
                labels.float(), index, match_size=False, only_inplane_resample=True
            )

        return (
            images,
            labels
        )

    def get_original_pixel_size(self, index, orientation: str = "DHW"):
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

    def get_processed_pixel_size(self, orientation: str = "DHW"):
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

    def get_original_image_size(self, index, orientation: str = "DHW"):
        """
        Get the original image size in the specified orientation.

        Parameters
        ----------
        orientation : str, optional
            The desired orientation of the image size. Default is 'DHW'.

        Returns
        -------
        tuple
            The image size in the specified orientation.
        """
        nx, ny, nz = self.n_pix_original[:, index]

        return transform_orientation(nx, ny, nz, orientation)

    def image_transformation(self, images):
        if len(images.unique()) == 1:
            return images

        assert_in(self.image_transform, "self.image_transform", ["none", "random_net"])
        if self.image_transform == "none":
            return images

        elif self.image_transform == "random_net":
            self.random_net.reset_parameters()
            with torch.no_grad():
                img_aug = self.random_net(images.unsqueeze(1)).reshape(images.shape)
                img_aug -= img_aug.min()
                img_aug /= img_aug.max()

                if self.only_foreground:
                    img_aug[images == 0] = 0

            return img_aug

    def set_augmentation(self, augmentation: bool):
        self.apply_augmentation = augmentation

    def get_augmentation(self) -> bool:
        return self.apply_augmentation

    def set_seed(self, seed=None):
        self.seed = seed

    def __len__(self):
        return self.images.shape[0]

    def idx_to_slice_idx(self, idx):
        idx = (idx / self.dim_proc[-1]) % 1
        return idx

    def __getitem__(self, index):

        images = self.images[index, ...]
        labels = self.labels[index, ...]

        seed = get_seed() if self.seed is None else self.seed

        if self.apply_augmentation:
            images, labels = apply_data_augmentation(
                images,
                labels,
                **self.aug_params,
                rng=np.random.default_rng(seed),
            )

        if self.apply_deformation:
            labels_deformed = make_noise_masks_3d(
                self.deformation["mask_type"],
                self.deformation["mask_radius"],
                self.deformation["mask_squares"],
                self.n_classes,
                labels,
                self.deformation["is_num_masks_fixed"],
                self.deformation["is_size_masks_fixed"],
            )
        else:
            labels_deformed = torch.tensor([])

        images = torch.from_numpy(images)

        if self.apply_deformation:
            images = self.image_transformation(images)

        labels = torch.from_numpy(labels)
        labels = class_to_onehot(labels, self.n_classes, class_dim=0)

        if self.deformation is not None:
            labels_deformed = torch.from_numpy(labels_deformed)
            labels_deformed = class_to_onehot(
                labels_deformed, self.n_classes, class_dim=0
            )

        if self.image_size[0] == 1:
            index = self.idx_to_slice_idx(index)

        if self.deformation is not None:
            return images, labels, labels_deformed, index
        else:
            return images, labels

    def get_label_name(self, label):
        return self.label_names[label]


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt

    root_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    sys.path.append(root_dir)

    from tta_uia_segmentation.src.utils.io import load_config
    from tta_uia_segmentation.src.utils.utils import resize_volume
    from tta_uia_segmentation.src.utils.loss import dice_score

    dice_score_fn_mean = lambda y_pred, y_gt: dice_score(
        y_pred, y_gt, soft=False, reduction="mean", smooth=1e-5, foreground_only=True
    )
    dice_score_fn = lambda y_pred, y_gt: dice_score(
        y_pred, y_gt, soft=False, reduction="none", smooth=1e-5, foreground_only=True
    )

    dataset_config = load_config(os.path.join(root_dir, "config", "datasets.yaml"))

    def current_evaluation_approach(y, orig_pix_size, preprocessed_pix_size):
        scale_factor = torch.tensor(orig_pix_size) / torch.tensor(preprocessed_pix_size)
        output_size = (torch.tensor(y.shape[2:]) * scale_factor).round().int().tolist()

        _, _, D, H, W = y.shape

        # Downsize to preprocessed pixel size
        y_resized = F.interpolate(y.float(), size=output_size, mode="trilinear")

        # Upsample to original dimensions
        y_resized = F.interpolate(y_resized, size=(D, H, W), mode="trilinear")

        print(
            "Dice score current evaluation approach: ", dice_score_fn_mean(y_resized, y)
        )
        print(
            "Dice score current evaluation approach (each class): ",
            dice_score_fn(y_resized, y),
        )

    split = "train"
    dataset = "vu_w_synthseg_labels"
    image_size = [1, 256, 256]

    dataset_info = dataset_config[dataset]

    aug_params = {
        "da_ratio": 0.25,
        "sigma": 20,
        "alpha": 0,
        "trans_min": 0,
        "trans_max": 0,
        "rot_min": 0,
        "rot_max": 0,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "gamma_min": 0.5,
        "gamma_max": 2.0,
        "brightness_min": 0.0,
        "brightness_max": 0.1,
        "noise_mean": 0.0,
        "noise_std": 0.1,
    }

    (ds,) = get_datasets(
        dataset_name=dataset,
        paths=dataset_info["paths_processed"],
        paths_original=dataset_info["paths_original"],
        splits=[split],
        image_size=image_size,
        resolution_proc=dataset_info["resolution_proc"],
        dim_proc=dataset_info["dim"],
        n_classes=dataset_info["n_classes"],
        aug_params=aug_params,
        deformation=None,
        load_original=True,
    )

    vol_index = 0
    x_orig, y_orig = ds.get_original_images(vol_index)
    print(x_orig.shape, y_orig.shape)

    x_proc, y_proc = ds.get_preprocessed_images(
        vol_index, same_position_as_original=True
    )
    print(x_proc.shape, y_proc.shape)

    print(f"Original pixel size: {ds.get_original_pixel_size(vol_index)}")
    print(f"Processed pixel size: {ds.get_processed_pixel_size()}")

    print("Resizing preprocessed volume to original size")
    # Test resizing of preprocessed volume to original size
    x_proc_resized = resize_volume(
        x_proc,
        current_pix_size=ds.get_processed_pixel_size(),
        target_pix_size=ds.get_original_pixel_size(vol_index),
        target_img_size=tuple(x_orig.shape[-3:]),
        mode="trilinear",
        only_inplane_resample=True,
    )
    print(x_proc_resized.shape)

    # Test resizing of label volume to preprocessed size
    y_proc_resized = resize_volume(
        y_proc.float(),
        current_pix_size=ds.get_processed_pixel_size(),
        target_pix_size=ds.get_original_pixel_size(vol_index),
        target_img_size=tuple(y_orig.shape[-3:]),
    )
    print(y_proc_resized.shape)

    print(
        f"Dice betweeen y_orig and y_proc_resized: {dice_score_fn_mean(y_proc_resized, y_orig)}"
    )
    print(
        f"Dice betweeen y_orig and y_proc_resized (each class): {dice_score_fn(y_proc_resized, y_orig)}\n"
    )

    current_evaluation_approach(
        y_orig, ds.get_original_pixel_size(vol_index), ds.get_processed_pixel_size()
    )
