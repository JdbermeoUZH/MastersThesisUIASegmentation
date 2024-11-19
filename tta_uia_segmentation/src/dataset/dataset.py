import os
import math
import copy
import random
from abc import ABC, abstractmethod
from typing import Literal, Optional, Any

import h5py
import numpy as np
from skimage.transform import rescale
import torch
from torch.utils import data
import torch.nn.functional as F

from tta_uia_segmentation.src.dataset.augmentation import apply_data_augmentation
from tta_uia_segmentation.src.dataset.deformation import make_noise_masks_3d
from tta_uia_segmentation.src.dataset.utils import (
    transform_orientation,
    ensure_nd,
)
from tta_uia_segmentation.src.utils.loss import class_to_onehot
from tta_uia_segmentation.src.utils.utils import (
    default,
    get_seed,
    assert_in,
    resize_image,
    resize_volume,
)


class ExtraInputs(ABC):
    @abstractmethod
    def get_extra_inputs_with_same_spatial_size(self) -> list[torch.Tensor]:
        pass

    @abstractmethod
    def set_extra_inputs_with_same_spatial_size(self, extra_inputs: list[torch.Tensor]):
        pass

    @abstractmethod
    def get_all_extra_inputs(self) -> list[Any]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def to_dict():
        pass


class ExtraInputsEmpty(ExtraInputs):
    def get_extra_inputs_with_same_spatial_size(self) -> list[torch.Tensor]:
        return list()

    def get_all_extra_inputs(self) -> list[Any]:
        return list()

    def set_extra_inputs_with_same_spatial_size(
        self, extra_inputs: list[torch.Tensor]
    ) -> None:
        pass

    def __len__(self) -> int:
        return 0

    def to_dict(self):
        return dict()


def get_datasets(splits, *args, **kwargs):

    return [Dataset(split=split, *args, **kwargs) for split in splits]


class Dataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        paths_preprocessed: str,
        paths_original: str,
        split: str,
        resolution_proc: tuple[float, float, float],
        dim_proc: tuple[int, int, int],
        n_classes: int,
        mode: Literal["2D", "3D"] = "2D",
        orientation: Optional[Literal["depth", "height", "width", "any"]] = "depth",
        rescale_factor: Optional[tuple[float, float, float]] = None,
        rescale_mode: Literal["trilinear", "bilinear", "nearest"] = "bilinear",
        rescale_only_inplane: bool = False,
        aug_params: Optional[dict] = None,
        load_original: bool = False,
        load_in_memory: bool = True,
        seed: int = None,
        label_names: Optional[dict] = None,
        check_dims_proc_and_orig_match: bool = False,
    ):

        assert_in(split, "split", ["train", "val", "test"])
        self._dataset_name = dataset_name
        self._path_preprocessed = os.path.expanduser(paths_preprocessed[split])
        self._path_original = os.path.expanduser(paths_original[split])
        self._split = split
        self._load_in_memory = load_in_memory
        self._resolution_proc = resolution_proc
        self._dim_proc = dim_proc
        self._n_classes = n_classes
        self._mode = mode
        self._orientation = orientation

        self._rescale_factor = rescale_factor
        self._rescale_mode = rescale_mode
        self._rescale_only_inplane = rescale_only_inplane

        self._aug_params = aug_params
        self._augment = aug_params is not None
        self._load_original = load_original
        self._seed = seed

        # Load original and preprocessed to memory (if specified) images and labels
        # :=========================================================================:
        self._h5f_preprocessed = None
        self._images_preprocessed = None
        self._labels_preprocessed = None

        if self._load_in_memory:
            self._load_all_preprocessed_images_to_memory()

        self._h5f_original = None
        self._labels_original = None
        self._images_original = None
        self._n_pix_original = None

        if load_original:
            self._load_original_images(load_in_memory=load_in_memory)

        # Define dataset length
        # :=========================================================================:
        self._define_dataset_length(check_dims_proc_and_orig_match)

        # Define output names of output classes
        # :=========================================================================:
        if label_names is None:
            num_zeros_pad = math.ceil(math.log10(n_classes + 1))
            label_names = {i: str(i).zfill(num_zeros_pad) for i in range(n_classes)}
        else:
            assert len(label_names) == n_classes, "len(label_names) != n_classes"

        self.label_names = label_names

    def _define_dataset_length(
        self, check_dims_proc_and_orig_match: bool = False
    ) -> None:
        if self._mode == "2D":
            with h5py.File(self._path_preprocessed, "r") as data:
                self._length = data["images"].shape[0]
                self._length *= (
                    self._rescale_factor[-1] if self._rescale_factor is not None else 1
                )

            if self._load_original and check_dims_proc_and_orig_match:
                with h5py.File(self._path_original, "r") as data:
                    original_length = data["images"].shape[0] * data["images"].shape[1]

                assert (
                    self._length == original_length
                ), "Length of preprocessed and original data do not match"

        elif self._mode == "3D":
            with h5py.File(self._path_preprocessed, "r") as data:
                self._length = data["images"].shape[0] // self._dim_proc[2]

            if self._load_original and check_dims_proc_and_orig_match:
                with h5py.File(self._path_original, "r") as data:
                    original_length = data["images"].shape[0]

                assert (
                    self._length == original_length
                ), "Length of preprocessed and original data do not match"

    def _get_images_and_labels(
        self,
        index: int,
        item_type: Literal["original", "preprocessed"],
        orientation: Literal["depth", "height", "width"] = "depth",
    ) -> tuple[np.ndarray, np.ndarray]:
        # Define the images and labels items from which to slice/load the data
        # :=========================================================================:
        if item_type == "original":
            # Open connection to h5 file of preprocessed images. if not already opened
            if not self._load_in_memory and self._h5f_original is None:
                self._open_connection_to_original_h5_file()

            images = self._images_original
            labels = self._labels_original

        elif item_type == "preprocessed":
            # Open connection to h5 file of preprocessed images. if not already opened
            if not self._load_in_memory and self._h5f_preprocessed is None:
                self._open_connection_to_preprocessed_h5_file()

            images = self._images_preprocessed
            labels = self._labels_preprocessed

        else:
            raise ValueError(f"Unknown item_type: {item_type}")

        # Load/Slice depending on the orientation and if the mode is 2D or 3D
        # :=========================================================================:
        slice_index = self._get_slice_indexes(index, images.shape, orientation)

        images = images[slice_index].astype(np.float32)
        labels = labels[slice_index].astype(np.uint8)

        # Slice image to the correct size, if necessary
        if item_type == "original":
            vol_idx = self._get_vol_idx_for_img_idx(index, orientation)
            nx, ny, nz = self.get_original_image_size_w_orientation(
                vol_idx, orientation="HWD"
            )
            slice_index = [Ellipsis, slice(0, nz), slice(0, nx), slice(0, ny)]
            slice_index = tuple(
                [
                    slice_index[i] if dim_i > 1 else slice(None)
                    for i, dim_i in enumerate(images.shape)
                ]
            )

            images = images[slice_index]
            labels = labels[slice_index]

        # If in 2D mode, squeeze and add batch and channel dimensions
        # :=========================================================================:
        if self._mode == "2D":
            images = images.squeeze()
            labels = labels.squeeze()

            images, labels = ensure_nd(3, images, labels)

        # If mode is 3D, reorient volume to requested orientation
        # :=========================================================================:
        elif self._mode == "3D":
            images, labels = ensure_nd(4, images, labels)

            if orientation in ["height", "width"]:
                if orientation == "height":
                    source = 1  # D
                    destination = 2  # H

                elif orientation == "width":
                    source = 1  # D
                    destination = 3  # W

                images = np.moveaxis(images, source, destination)
                labels = np.moveaxis(labels, source, destination)

        return images, labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, ExtraInputs]]:
        seed = default(self._seed, get_seed())

        # Get the orientation
        orientation = (
            random.choice(["depth", "height", "width"])
            if self._orientation == "any"
            else self._orientation
        )

        # Get the images and labels
        # :=========================================================================:
        images, labels = self._get_images_and_labels(
            index, "preprocessed", orientation=orientation
        )

        # Apply augmentations if specified
        # :=========================================================================:
        if self._augment:
            images, labels = apply_data_augmentation(
                images,
                labels,
                **self._aug_params,
                rng=np.random.default_rng(seed),
            )

        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)

        labels = class_to_onehot(labels, self._n_classes, class_dim=0)

        extra_inputs = self._calculate_extra_inputs(index, images, labels)

        # Apply rescale factor if specified and not already applied
        # :=========================================================================:
        if self._rescale_factor is not None and not self._load_in_memory:
            tensors_to_rescale = [images, labels]
            tensors_to_rescale += extra_inputs.get_extra_inputs_with_same_spatial_size()

            if orientation == "depth":
                rescale_factor = self.get_rescale_factor_w_orientation("DHW")

            elif orientation == "height":
                rescale_factor = self.get_rescale_factor_w_orientation("HDW")

            elif orientation == "width":
                rescale_factor = self.get_rescale_factor_w_orientation("WDH")

            if self._mode == "3D":
                images, labels, *extra_inputs_rescaled = [
                    resize_volume(
                        tensor.float().unsqueeze(0), # add batch dimension
                        current_pix_size=rescale_factor,
                        target_pix_size=(1, 1, 1),
                        mode=self._rescale_mode,
                        only_inplane_resample=self._rescale_only_inplane,
                        input_format="NCDHW",
                        output_format="NCDHW",
                    )
                    for tensor in tensors_to_rescale
                ]

                extra_inputs.set_extra_inputs_with_same_spatial_size(
                    extra_inputs_rescaled
                )

            elif self._mode == "2D":
                images, labels, *extra_inputs_rescaled = [
                    resize_image(
                        tensor.unsqueeze(0), # add batch dimension
                        current_pix_size=rescale_factor[-2:],
                        target_pix_size=(1, 1),
                        mode="bilinear",
                        img_format="NCHW",
                        output_format="NCHW",
                    )
                    for tensor in tensors_to_rescale
                ]

        return images, labels, extra_inputs.to_dict()

    def _calculate_extra_inputs(
        self, index: int, images: torch.Tensor, labels: torch.Tensor
    ) -> ExtraInputs:
        return ExtraInputsEmpty()

    def _get_vol_idx_for_img_idx(
        self, index: int, orientation: Literal["depth", "height", "width"] = "depth"
    ) -> int:
        # Open connection to h5 file of original images if necessary
        if not self._load_in_memory and self._h5f_original is None:
            self._open_connection_to_original_h5_file()

        _, dim_z, dim_x, dim_y = self._images_original.shape

        if orientation == "depth":
            return index // dim_z

        elif orientation == "height":
            return index // dim_x

        elif orientation == "width":
            return index // dim_y

    def _get_slice_indexes(
        self,
        index: int,
        tensor_shape: tuple[int, ...],
        orientation: Optional[Literal["depth", "height", "width"]] = None,
    ) -> tuple:
        dx, dy, dz = self.get_dim_proc_w_orientation("HWD")

        # N=N_vol * D, H, W, typically the case for preprocessed data
        if len(tensor_shape) == 3:
            if self._mode == "2D":
                if orientation == "depth":
                    slice_index = (slice(index, index + 1), Ellipsis)

                elif orientation == "width":
                    vol_idx = index // dy
                    y_idx = index % dy
                    vol_start_idx = vol_idx * dz
                    vol_end_idx = vol_start_idx + dz
                    slice_index = (
                        slice(vol_start_idx, vol_end_idx),
                        slice(None),
                        slice(y_idx, y_idx + 1),
                    )

                elif orientation == "height":
                    vol_idx = index // dx
                    x_idx = index % dx
                    vol_start_idx = vol_idx * dz
                    vol_end_idx = vol_start_idx + dz
                    slice_index = (
                        slice(vol_start_idx, vol_end_idx),
                        slice(x_idx, x_idx + 1),
                        slice(None),
                    )

                else:
                    raise NotImplementedError(
                        f"Orientation {orientation} not implemented for 2D data"
                    )

            elif self._mode == "3D":
                start_idx = index * dz
                end_idx = start_idx + dz
                slice_index = (slice(start_idx, end_idx), Ellipsis)

        # N_vol, D, H, W, typically the case for original data
        elif len(tensor_shape) == 4:
            if self._mode == "2D":
                vol_idx = index // tensor_shape[1]

                if orientation == "depth":
                    z_idx = index % tensor_shape[1]
                    slice_index = (
                        slice(vol_idx, vol_idx + 1),
                        slice(z_idx, z_idx + 1),
                        Ellipsis,
                    )

                elif orientation == "width":
                    y_idx = index % tensor_shape[2]
                    slice_index = (
                        slice(vol_idx, vol_idx + 1),
                        slice(None),
                        slice(None),
                        slice(y_idx, y_idx + 1),
                    )

                elif orientation == "height":
                    x_idx = index % tensor_shape[3]
                    slice_index = (
                        slice(vol_idx, vol_idx + 1),
                        slice(None),
                        slice(x_idx, x_idx + 1),
                        slice(None),
                    )

                else:
                    raise ValueError(f"Unknown orientation: {orientation}")

            elif self._mode == "3D":
                slice_index = (slice(index, index + 1), Ellipsis)

        return slice_index

    def get_original_volume(
        self,
        index: int,
        output_format: Literal["DCHW", "NCDHW"] = "NCDHW",
        as_onehot: bool = True,
        orientation: Optional[Literal["depth", "height", "width"]] = "depth",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the original image and label volumes at the specified index.

        Parameters
        ----------
        index : int
            The index of the item to retrieve.
        output_format : {'CHW', 'NCHW'}, optional
            The desired format of the returned tensors. Default is 'NCHW'.
        as_onehot : bool, optional
            If True, return labels in one-hot encoding. Default is True.
        orientation : {'depth', 'height', 'width'}, optional
            The desired orientation of the image and label items. Default is 'depth'.

        """

        return self._get_original_item(
            index, "3D", output_format, as_onehot, orientation
        )

    def get_original_image(
        self,
        index: int,
        output_format: Literal["CHW", "NCHW"] = "NCHW",
        as_onehot: bool = True,
        orientation: Literal["depth", "height", "width"] = "depth",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the original image and label at the specified index.

        Parameters
        ----------
        index : int
            The index of the item to retrieve.
        output_format : {'CHW', 'NCHW'}, optional
            The desired format of the returned tensors. Default is 'NCHW'.
        as_onehot : bool, optional
            If True, return labels in one-hot encoding. Default is True.
        orientation : {'depth', 'height', 'width'}, optional
            The desired orientation of the image and label items. Default is 'depth'.

        """

        return self._get_original_item(
            index, "2D", output_format, as_onehot, orientation
        )

    def _get_original_item(
        self,
        index: int,
        mode: Literal["2D", "3D"],
        output_format: Literal["CHW", "NCHW", "DCHW", "NCDHW"],
        as_onehot: bool = True,
        orientation: Optional[Literal["depth", "height", "width"]] = "depth",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the original image and label volumes at the specified index.

        Parameters
        ----------
        index : int
            The index of the item to retrieve.
        as_onehot : bool, optional
            If True, return labels in one-hot encoding. Default is True.
        output_format : {'DCHW', 'NCDHW'}, optional
            The desired format of the returned tensors. Default is 'NCDHW'.
        orientation : {'depth', 'height', 'width'}, optional
            The desired orientation of the image and label items. Default is 'depth'.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the image, labels

        Raises
        ------
        ValueError
            If an unknown format is specified.
        """
        if mode == "2D" and output_format in ["DCHW", "NCDHW"]:
            raise ValueError(f"output_format {output_format} not supported for mode 2D")
        if mode == "3D" and output_format in ["CHW", "NCHW"]:
            raise ValueError(f"output_format {output_format} not supported for mode 3D")

        # Get the orientation
        orientation = (
            random.choice(["depth", "height", "width"])
            if self._orientation == "any"
            else self._orientation
        )

        if not self._load_original:
            self._load_original_images()

        # Get images and labels according to the specified mode
        # :=========================================================================:
        original_mode = self._mode

        self._mode = mode
        images, labels = self._get_images_and_labels(index, "original", orientation)

        self._mode = original_mode

        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)

        # Format the tensors to the correct output shape
        # :=========================================================================:
        if as_onehot:
            labels = class_to_onehot(labels, self._n_classes, class_dim=1)

        if output_format in ["NCDHW"]:
            images, labels = ensure_nd(5, images, labels)

        elif output_format == "DCHW":
            images = images.permute(0, 2, 1, 3, 4).squeeze(0)
            labels = labels.permute(0, 2, 1, 3, 4).squeeze(0)

        elif output_format == "NCHW":
            images, labels = ensure_nd(4, images, labels)

        elif output_format == "CHW":
            images, labels = ensure_nd(3, images, labels)

        else:
            raise ValueError(f"Unknown format: {format}")

        assert len(output_format) == len(images.shape) == len(labels.shape), (
            f"images or labels shape do not match output_format: {output_format}"
            + f", images.shape: {images.shape}, "
            + f"labels.shape: {labels.shape}"
        )

        return images, labels

    def __del__(self):
        if self._h5f_preprocessed is not None:
            self._h5f_preprocessed.close()
        if self._h5f_original is not None:
            self._h5f_original.close()

    def _load_original_images(self, load_in_memory: bool = False):

        # Number of pixels for original images.
        with h5py.File(self._path_original, "r") as h5f:
            self._n_pix_original = np.stack([h5f["nx"], h5f["ny"], h5f["nz"]])

        # Pixel sizes of original images.
        with h5py.File(self._path_preprocessed, "r") as h5f:
            self._pix_size_original = np.stack([h5f["px"], h5f["py"], h5f["pz"]])

        if load_in_memory:
            self._load_all_original_volumes_in_memory()

    def _load_all_original_volumes_in_memory(self):
        with h5py.File(self._path_original, "r") as h5f:
            self._images_original = h5f["images"][:]
            self._labels_original = h5f["labels"][:]

    def _load_all_preprocessed_images_to_memory(self):
        """
        Load all images and labels to memory.
        """
        with h5py.File(self._path_preprocessed, "r") as h5f:
            # Load images and labels to memory if specified.
            #  If not, a connection to the file is opened once __getitem__ is called.
            self._images_preprocessed = h5f["images"][:]
            self._labels_preprocessed = h5f["labels"][:]

        if self._rescale_factor is not None:
            # Rescale images and labels with the specified rescale factor
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
        target_size = (
            self.get_original_image_size_w_orientation(index) if match_size else None
        )

        return resize_volume(
            volume,
            current_pix_size=self.get_processed_pixel_size_w_orientation(),
            target_pix_size=self.get_original_pixel_size_w_orientation(index),
            target_img_size=target_size,
            mode="bilinear" if only_inplane_resample else "trilinear",
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
        if self.orientation == "depth":
            target_size = self.get_dim_proc_w_orientation("DHW") if match_size else None
        elif self.orientation == "height":
            target_size = self.get_dim_proc_w_orientation("HDW") if match_size else None
        elif self.orientation == "width":
            target_size = self.get_dim_proc_w_orientation("WDH") if match_size else None

        return resize_volume(
            volume,
            current_pix_size=self.get_original_pixel_size_w_orientation(index),
            target_pix_size=self.get_processed_pixel_size_w_orientation(),
            target_img_size=target_size,
            mode="bilinear" if only_inplane_resample else "trilinear",
            only_inplane_resample=only_inplane_resample,
        )

    def get_preprocessed_original_volume(
        self,
        index: int,
        as_onehot: bool = True,
        output_format: Literal["DCHW", "NCDHW"] = "NCDHW",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the preprocessed image and label volumes at the specified index.

        Parameters
        ----------
        index : int
            The index of the volume to retrieve.
        as_onehot : bool, optional
            If True, return labels in one-hot encoding. Default is True.
        output_format : {'DCHW', 'NCDHW'}, optional
            The desired format of the returned tensors. Default is 'NCDHW'.

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

        images, labels = self.get_original_volume(
            index, as_onehot=as_onehot, output_format=output_format
        )

        images = self.resize_original_vol_to_processed_vol(
            images, index, match_size=False, only_inplane_resample=True
        )
        labels = self.resize_original_vol_to_processed_vol(
            labels.float(), index, match_size=False, only_inplane_resample=True
        )

        return images, labels

    def get_dim_proc_w_orientation(self, orientation: str = "DHW"):
        """
        Get the dimensions of the preprocessed image in the specified orientation.

        Parameters
        ----------
        orientation : str, optional
            The desired orientation of the dimensions. Default is 'DHW'.

        Returns
        -------
        tuple
            The dimensions in the specified orientation.
        """
        nx, ny, nz = self._dim_proc

        return transform_orientation(nx, ny, nz, orientation)

    def get_rescale_factor_w_orientation(self, orientation: str = "DHW"):
        """
        Get the rescale factor in the specified orientation.

        Parameters
        ----------
        orientation : str, optional
            The desired orientation of the rescale factor. Default is 'DHW'.

        Returns
        -------
        tuple
            The rescale factor in the specified orientation.
        """
        x, y, z = self._rescale_factor

        return transform_orientation(x, y, z, orientation)

    def get_original_pixel_size_w_orientation(self, index, orientation: str = "DHW"):
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

    def get_processed_pixel_size_w_orientation(self, orientation: str = "DHW"):
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

    def get_original_image_size_w_orientation(self, index, orientation: str = "DHW"):
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

    def get_num_volumes(self):
        self._open_connection_to_original_h5_file()
        return self._images_original.shape[0]

    def _open_connection_to_preprocessed_h5_file(self):
        if self._h5f_preprocessed is None:
            self._h5f_preprocessed = h5py.File(self._path_preprocessed, "r")
            self._images_preprocessed = self._h5f_preprocessed["images"]
            self._labels_preprocessed = self._h5f_preprocessed["labels"]

    def close_connection_to_preprocessed_h5_file(self):
        if self._h5f_preprocessed is not None:
            self._h5f_preprocessed.close()
            self._h5f_preprocessed = None

    def _open_connection_to_original_h5_file(self):
        if self._h5f_original is None:
            self._h5f_original = h5py.File(self._path_original, "r")
            self._images_original = self._h5f_original["images"]
            self._labels_original = self._h5f_original["labels"]

    def close_connection_to_original_h5_file(self):
        if self._h5f_original is not None:
            self._h5f_original.close()
            self._h5f_original = None

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

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

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation: Literal["depth", "height", "width", "any"]):
        self._orientation = orientation

    @property
    def rescale_factor(self):
        return self._rescale_factor

    @rescale_factor.setter
    def rescale_factor(self, rescale_factor: tuple[float, float, float]):
        self._rescale_factor = rescale_factor

    def __len__(self):
        return self._length

    def idx_to_slice_idx(self, idx):
        idx = (idx / self._dim_proc[-1]) % 1
        return idx

    def get_label_name(self, label):
        return self.label_names[label]
    
    @property
    def path_preprocessed(self):
        return self._path_preprocessed
    
    @property
    def path_original(self):
        return self._path_original

    @property
    def dim_proc(self):
        return self._dim_proc