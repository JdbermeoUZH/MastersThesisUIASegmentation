import os
import math
from collections import namedtuple
from typing import Literal, Optional

import h5py
import numpy as np
from skimage.transform import rescale
import torch
from torch.utils import data
import torch.nn.functional as F

from tta_uia_segmentation.src.dataset.augmentation import apply_data_augmentation
from tta_uia_segmentation.src.dataset.deformation import make_noise_masks_3d
from tta_uia_segmentation.src.dataset.utils import transform_orientation
from tta_uia_segmentation.src.utils.loss import class_to_onehot
from tta_uia_segmentation.src.utils.utils import (
    default,
    get_seed,
    assert_in,
    resize_volume,
)


extra_inputs = namedtuple("extra_inputs", ["labels_deformed"])


def get_datasets(splits, *args, **kwargs):

    return [Dataset(split=split, *args, **kwargs) for split in splits]


class Dataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        paths: str,
        paths_original: str,
        split: str,
        resolution_proc: tuple[float, float, float],
        dim_proc: tuple[int, int, int],
        n_classes: int,
        mode: Literal["2D", "3D"] = "2D",
        rescale_factor: Optional[tuple[float, float, float]] = None,
        reshape_to_dim_proc: bool = True,
        aug_params: Optional[dict] = None,
        deformation_params: Optional[dict] = None,
        load_original: bool = False,
        load_in_memory: bool = True,
        seed: int = None,
        label_names: Optional[dict] = None,
        check_dims_proc_and_orig_match: bool = True,
    ):

        assert_in(split, "split", ["train", "val", "test"])
        self._dataset_name = dataset_name
        self._path_preprocessed = os.path.expanduser(paths[split])
        self._path_original = os.path.expanduser(paths_original[split])
        self._split = split
        self._load_in_memory = load_in_memory
        self._resolution_proc = resolution_proc
        self._rescale_factor = rescale_factor
        self._reshape_to_dim_proc = reshape_to_dim_proc
        self._dim_proc = dim_proc
        self._n_classes = n_classes
        self._mode = mode

        self._aug_params = aug_params
        self._augment = aug_params is not None
        self._deformation_params = deformation_params
        self._deform = deformation_params is not None
        self._load_original = load_original
        self._seed = seed

        # If rescaling, adjust dim_proc
        if rescale_factor is not None:
            self._dim_proc = (
                (self._dim_proc * np.array(rescale_factor)).astype(int).tolist()
            )

        # Load original and preprocessed to memory (if specified) images and labels
        # :=========================================================================:
        self._h5f_preprocessed = None
        self._images_preprocessed = None
        self._labels_preprocessed = None

        self._h5f_original = None
        self._labels_original = None
        self._images_original = None
        self._n_pix_original = None

        if self._load_in_memory:
            self.load_all_images_to_memory()

        # Define dataset length
        # :=========================================================================:
        if self._mode == "2D":
            with h5py.File(self._path_preprocessed, "r") as data:
                self._length = data["images"].shape[0]

            if load_original and check_dims_proc_and_orig_match:
                with h5py.File(self._path_original, "r") as data:
                    original_length = data["images"].shape[0] * data["images"].shape[1]

                assert (
                    self._length == original_length
                ), "Length of preprocessed and original data do not match"

        elif self._mode == "3D":
            with h5py.File(self._path_preprocessed, "r") as data:
                self._length = data["images"].shape[0] // self._dim_proc[0]

            if load_original and check_dims_proc_and_orig_match:
                with h5py.File(self._path_original, "r") as data:
                    original_length = data["images"].shape[0]

                assert (
                    self._length == original_length
                ), "Length of preprocessed and original data do not match"

        # Define output names of output classes
        # :=========================================================================:
        if label_names is None:
            num_zeros_pad = math.ceil(math.log10(n_classes + 1))
            label_names = {i: str(i).zfill(num_zeros_pad) for i in range(n_classes)}
        else:
            assert len(label_names) == n_classes, "len(label_names) != n_classes"

        self.label_names = label_names

    def __getitem__(self, index):

        # Open connection to h5 file of preprocessed images
        # :=========================================================================:
        if not self._load_in_memory and self._h5f_original is None:
            self._h5f_preprocessed = h5py.File(self._path_preprocessed, "r")
            self._images_preprocessed = self._h5f_preprocessed["images"]
            self._labels_preprocessed = self._h5f_preprocessed["labels"]

        # Load and reshape images and labels if they are 2D or 3D
        # :=========================================================================:
        slice_index = self._get_slice_indexes(index, self._images_preprocessed.shape)

        images = self._images_preprocessed[slice_index].astype(np.float32)
        labels = self._labels_preprocessed[slice_index].astype(np.uint8)

        breakpoint()
        if self._reshape_to_dim_proc is not None:
            dims_to_reshape = 3 if self._mode == "3D" else 2
            images = images.reshape(-1, *self._dim_proc[-dims_to_reshape:])
            labels = labels.reshape(-1, *self._dim_proc[-dims_to_reshape:])

        seed = default(self._seed, get_seed())

        # Apply augmentations if specified
        # :=========================================================================:
        if self._augment:
            images, labels = apply_data_augmentation(
                images,
                labels,
                **self._aug_params,
                rng=np.random.default_rng(seed),
            )

        # Apply deformations if specified
        # :=========================================================================:
        if self._deform:
            labels_deformed = make_noise_masks_3d(
                self._deformation_params["mask_type"],
                self._deformation_params["mask_radius"],
                self._deformation_params["mask_squares"],
                self._n_classes,
                labels,
                self._deformation_params["is_num_masks_fixed"],
                self._deformation_params["is_size_masks_fixed"],
            )
            labels_deformed = torch.from_numpy(labels_deformed)
            labels_deformed = class_to_onehot(
                labels_deformed, self._n_classes, class_dim=0
            )
        else:
            extra_inputs.labels_deformed = torch.tensor([])

        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)
        labels = class_to_onehot(labels, self._n_classes, class_dim=0)

        # Apply rescale factor if specified and not already applied
        # :=========================================================================:
        if self._rescale_factor is not None and not self._load_in_memory:
            tensors_to_rescale = [images, labels, labels_deformed]

            batch_format = "1CDHW" if self._mode == "3D" else "DCHW"

            images, labels, labels_deformed = [
                (
                    resize_volume(
                        tensor,
                        current_pix_size=self._rescale_factor,
                        target_pix_size=(1, 1, 1),
                        target_img_size=self._dim_proc,
                        mode="trilinear",
                        only_inplane_resample=False,
                        input_format=batch_format,
                        output_format=batch_format,
                    )
                    if tensor != torch.tensor([])
                    else tensor
                )
                for tensor in tensors_to_rescale
            ]

        return images, labels, labels_deformed

    def _get_slice_indexes(self, index: int, tensor_shape: tuple[int, ...]) -> tuple:
        # N=N_vol * D, H, W, typically the case for preprocessed data
        if len(tensor_shape) == 3:
            if self._mode == "2D":
                slice_index = (slice(index, index + 1), Ellipsis)
            elif self._mode == "3D":
                n_volumes = len(self)
                batch_idx = self._get_z_idxs_for_volumes()[index]
                slice_index = (batch_idx, Ellipsis)

        # N_vol, D, H, W, typically the case for original data
        elif len(tensor_shape) == 4:
            if self._mode == "2D":
                vol_idx = index // tensor_shape[1]
                z_idx = index % tensor_shape[1]
                slice_index = (
                    slice(vol_idx, vol_idx + 1),
                    slice(z_idx, z_idx + 1),
                    Ellipsis,
                )
            elif self._mode == "3D":
                slice_index = (slice(index, index + 1), Ellipsis)

        return slice_index

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
        if not self._load_original:
            self._load_original_images()

        # Open connection to h5 file of original images
        if not self._load_in_memory and self._h5f_original is None:
            self._h5f_original = h5py.File(self._path_original, "r")
            self._images_original = self._h5f_original["images"]
            self._labels_original = self._h5f_original["labels"]

        if self._mode == "3D":
            index = self.get_z_idxs_for_volume(index)

        # Get the original images and labels
        images = self._images_original[index].astype(np.float32)
        labels = self._labels_original[index].astype(np.uint8)

        # Slice the correct image size
        nx, ny, _ = self._n_pix_original[:, index]
        images = images[..., 0:nx, 0:ny]
        labels = labels[..., 0:nx, 0:ny]

        # Convert to torch tensors
        images = torch.from_numpy(images).unsqueeze(0)  # CDHW
        labels = torch.from_numpy(labels).unsqueeze(0)  # CDHW

        if as_onehot:
            labels = class_to_onehot(labels, self._n_classes, class_dim=1)  # 1CDHW

        # Format the tensors to the correct shape
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

    def __del__(self):
        if self._h5f_preprocessed is not None:
            self._h5f_preprocessed.close()
        if self._h5f_original is not None:
            self._h5f_original.close()

    def _load_original_images(self):
        # Number of pixels for original images.
        with h5py.File(self._path_original, "r") as h5f:
            self._n_pix_original = np.stack([h5f["nx"], h5f["ny"], h5f["nz"]])

        # Pixel sizes of original images.
        with h5py.File(self._path_preprocessed, "r") as data:
            self._pix_size_original = np.stack([data["px"], data["py"], data["pz"]])

        if self._load_in_memory:
            self._load_all_original_volumes_in_memory()

    def _load_all_original_volumes_in_memory(self):
        with h5py.File(self._path_original, "r") as h5f:
            self._images_original = h5f["images"][:]
            self._labels_original = h5f["labels"][:]

            # inplane_shape = self._images_original.shape[-2:]

            # if self._mode == "2D":
            #     self._images_original = self._images_original.reshape(
            #         -1, *inplane_shape
            #     )  # (N_vol*D)HW
            #     self._labels_original = self._labels_original.reshape(
            #         -1, *inplane_shape
            #     )  # (N_vol*D)HW

            # elif self._mode == "3D":
            #     num_volumes = len(h5f["nx"])
            #     self._images_original = self._images_original.reshape(
            #         num_volumes, -1, *inplane_shape
            #     )  # N_volDHW
            #     self._labels_original = self._labels_original.reshape(
            #         num_volumes, -1, *inplane_shape
            #     )  # N_volDHW

    def _load_all_preprocessed_images_to_memory(self):
        """
        Load all images and labels to memory.
        """
        with h5py.File(self._path_preprocessed, "r") as data:
            # Load images and labels to memory if specified.
            #  If not, a connection to the file is opened once __getitem__ is called.
            self._images_preprocessed = data["images"][:]
            self._labels_preprocessed = data["labels"][:]

        if self._rescale_factor is not None:
            self._images_preprocessed = torch.from_numpy(
                self._images_preprocessed
            ).unsqueeze(0)
            self._labels_preprocessed = torch.from_numpy(
                self._labels_preprocessed
            ).unsqueeze(0)

            self._images_preprocessed = F.interpolate(
                self._images_preprocessed,
                scale_factor=self._rescale_factor,
                mode="trilinear",
            )
            self._labels_preprocessed = F.interpolate(
                self._labels_preprocessed,
                scale_factor=self._rescale_factor,
                mode="nearest",
            )

            self._images_preprocessed = self._images_preprocessed.squeeze(0).numpy()
            self._labels_preprocessed = self._labels_preprocessed.squeeze(0).numpy()

    def load_all_images_to_memory(self, reload: bool = True):
        if self._load_in_memory and not reload:
            return

        self._load_in_memory = True
        self._load_all_preprocessed_images_to_memory()

        if self._load_original:
            self._load_all_original_volumes_in_memory()

    def _get_z_idxs_for_volumes(self) -> list[np.ndarray]:
        """
        Get the indices for each volume in the dataset.

        Returns
        -------
        List[np.ndarray]
            A list of numpy arrays, where each array contains the indices for a
            single volume.
        """
        n_slices = len(self)
        n_volumes = n_slices // self._dim_proc[0]
        n_slices_per_volume = np.array([self._dim_proc[0]] * n_volumes)

        indices = np.arange(n_slices)

        indices_per_volume = np.split(indices, n_slices_per_volume.cumsum()[:-1])

        return indices_per_volume

    def get_z_idxs_for_volume(self, vol_idx: int) -> np.ndarray:
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
        if self._mode == "2D":
            return self._get_z_idxs_for_volumes()[vol_idx]
        else:
            return np.array([vol_idx])

    def resize_processed_vol_to_original_vol(
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

    def resize_original_vol_to_processed_vol(
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
        target_size = self._dim_proc if match_size else None

        return resize_volume(
            volume,
            current_pix_size=self.get_original_pixel_size(index),
            target_pix_size=self.get_processed_pixel_size(),
            target_img_size=target_size,
            mode="trilinear",
            only_inplane_resample=only_inplane_resample,
        )

    def get_preprocessed_original_image(
        self,
        index: int,
        as_onehot: bool = True,
        format: Literal["DCHW", "1CDHW"] = "1CDHW",
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
        # Replicate preprocessing (it is only downsampling) on original data
        # we have to replicate it because there is a shift between the original and preprocessed data

        images, labels = self.get_original_images(
            index, as_onehot=as_onehot, format=format
        )

        images = self.resize_original_vol_to_processed_vol(
            images, index, match_size=False, only_inplane_resample=True
        )
        labels = self.resize_original_vol_to_processed_vol(
            labels.float(), index, match_size=False, only_inplane_resample=True
        )

        return images, labels

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
        x, y, z = self._pix_size_original[:, index]

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
        x, y, z = self._resolution_proc

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
        nx, ny, nz = self._n_pix_original[:, index]

        return transform_orientation(nx, ny, nz, orientation)

    @property
    def augment(self) -> bool:
        return self._augment

    @augment.setter
    def augment(self, augmentation: bool):
        self._augment = augmentation

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def set_seed(self, seed=None):
        self._seed = seed

    def __len__(self):
        return self._length

    def idx_to_slice_idx(self, idx):
        idx = (idx / self._dim_proc[-1]) % 1
        return idx

    def get_label_name(self, label):
        return self.label_names[label]


if __name__ == "__main__":
    import sys

    root_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    breakpoint()
    print(root_dir)

    sys.path.append(root_dir)

    from tta_uia_segmentation.src.utils.io import load_config
    from tta_uia_segmentation.src.utils.utils import resize_volume
    from tta_uia_segmentation.src.utils.loss import dice_score

    dice_score_fn_mean = lambda y_pred, y_gt: dice_score(
        y_pred, y_gt, soft=False, reduction="mean", foreground_only=True
    )
    dice_score_fn = lambda y_pred, y_gt: dice_score(
        y_pred, y_gt, soft=False, reduction="none", foreground_only=True
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
