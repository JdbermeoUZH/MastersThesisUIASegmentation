import os
import copy
import random
from typing import Any, Optional

import h5py
import torch
from torch.utils.data import Dataset

from tta_uia_segmentation.src.dataset.dataset import Dataset

from tta_uia_segmentation.src.utils.loss import class_to_onehot


class DatasetDinoFeatures(Dataset):
    def __init__(
        self,
        split: str,
        paths_preprocessed_dino: dict[str, Any],
        hierarchy_level: int,
        dino_model: str,
        **kwargs,
    ):
        # Handle how arguments should be set for the wrapped Dataset class
        # :====================================================================:
        kwargs_should_be_none = [
            "rescale_factor",
            "aug_params",
        ]

        for kwarg in kwargs_should_be_none:
            if kwarg in kwargs and kwargs[kwarg] is not None:
                raise ValueError(f"{kwarg} is not supported for DinoFeatures dataset")
            else:
                kwargs[kwarg] = None

        if "mode" in kwargs:
            assert (
                kwargs["mode"] == "2D"
            ), "DinoFeatures only precalculated on 2D slices"
        else:
            kwargs["mode"] = "2D"

        if "orientation" in kwargs:
            assert (
                kwargs["orientation"] == "depth"
            ), "DinoFeatures only precalculated on depth-wise slices"
        else:
            kwargs["orientation"] = "depth"

        if "load_in_memory" in kwargs:
            assert not kwargs[
                "load_in_memory"
            ], "DinoFeatures dataset is too large to load in memory"
        else:
            kwargs["load_in_memory"] = False

        super().__init__(split=split, **kwargs)

        # Move data to compute node if path is specified and adjust paths accordingly
        # :=========================================================================:
        self._path_preprocessed_dino = paths_preprocessed_dino[dino_model][split]

        if "node_data_path" in kwargs and kwargs["node_data_path"] is not None:
            breakpoint()
            node_data_path = kwargs["node_data_path"]
            old_path_preprocessed_dino = copy.deepcopy(self._path_preprocessed_dino)
            
            # Remove $DATA_DIR from the beginning of the paths, if necessary 
            if "DATA_DIR" in os.environ:
                data_dir = os.environ['DATA_DIR']

                self._path_preprocessed_dino = self._path_preprocessed_dino[len(data_dir):].lstrip('/')

            self._path_preprocessed_dino = os.path.join(node_data_path, self._path_preprocessed_dino)

            paths = (
                (old_path_preprocessed_dino, self._path_preprocessed_dino),
            )

            # Move data to compute node
            for old_path, new_path in paths:
                breakpoint()
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                os.system(f'rsync -a --inplace {old_path} {new_path} --progress' )

        # Define attributes specific to this class
        # :====================================================================:
        self._hierarchy_level = hierarchy_level

        with h5py.File(self._path_preprocessed_dino, "r") as h5f:
            self._dino_patch_size: int = h5f.attrs["patch_size"]  # type: ignore
            self._dino_emb_dim: int = h5f.attrs["emb_dim"]  # type: ignore
            self._dino_model: str = h5f.attrs["dino_model"]  # type: ignore
            self._n_augmentation_epochs: int = h5f.attrs["n_augmentation_epochs"]  # type: ignore
            self._dataset_original_size: int = h5f.attrs["dataset_original_size"]  # type: ignore
            self._hierarchy_level_available: int = h5f.attrs["hierarchy_level"]  # type: ignore

        assert self._dataset_original_size == self._length, "Dataset length mismatch"

        assert (
            dino_model == self._dino_model
        ), "Dino model mismatch, in h5 file it is: {self._dino_model}"

        assert (
            hierarchy_level <= self._hierarchy_level_available
        ), "Requested hierarchy level is not available in preprocessed Dino features"

        # Define attributes for preprocessed Dino features
        self._h5f_preprocessed_dino: Optional[h5py.File] = None
        self._images_preprocessed_dino: dict[int, Optional[h5py.Dataset]] = {
            hier: None for hier in range(self._hierarchy_level + 1)
        }
        self._labels_preprocessed_dino = None

    def _define_dataset_length(
        self, check_dims_proc_and_orig_match: bool = False
    ) -> None:
        """
        Dataset length is forced to be the same as the number of depth-wise slices
        """
        if self.dataset_name in ['vu', 'vu_w_synthseg_labels']:        
            with h5py.File(self._path_preprocessed, "r") as h5f:
                self._length = h5f["images"].shape[0]   # type: ignore
        else:
            with h5py.File(self._path_original, "r") as h5f:
                self._length = h5f["images"].shape[0] * h5f["images"].shape[1]  # type: ignore

    def __getitem__(
        self, index
    ) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, Any]]:
        """
        Retrieve Dino Features for a given index

        Dino features are also precomputed for precomputed augmentations
        """
        # Open connection to preprocessed Dino features, if necessary
        self._open_connection_to_preprocessed_dino_h5_file()

        if self.augment:
            # Precomputed features for augmented images
            #  are stored in [self.dataset_original_size:]
            #  for self.n_augmentation_epochs
            aug_epoch_to_sample = random.choice(
                range(1, self._n_augmentation_epochs + 1)
            )
            index += aug_epoch_to_sample * self._dataset_original_size

        # Get preprocessed Dino features
        x = list()
        for hier in range(self._hierarchy_level + 1):
            x.append(torch.tensor(self._images_preprocessed_dino[hier][index]))  # type: ignore

        # Get segmentation mask
        y = torch.tensor(self._labels_preprocessed_dino[index])  # type: ignore

        y = class_to_onehot(y, self._n_classes, class_dim=0)

        return x, y, {"output_size": tuple(y.shape[-2:])}

    def get_preprocessed_items(
        self,
        index: int,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor, dict[str, Any]]:
        """
        Get preprocessed slice, typically used for training directly on images
        """

        return super().__getitem__(index)

    def _open_connection_to_preprocessed_dino_h5_file(self):
        if self._h5f_preprocessed_dino is None:
            self._h5f_preprocessed_dino = h5py.File(self._path_preprocessed_dino, "r")

            self._images_preprocessed_dino = {  # type: ignore
                hier: self._h5f_preprocessed_dino[f"images_hier_{hier}"]
                for hier in range(self._hierarchy_level + 1)
            }
            self._labels_preprocessed_dino = self._h5f_preprocessed_dino["labels"]

    def close_connection_to_preprocessed_dino_h5_file(self):
        if self._h5f_preprocessed_dino is not None:
            self._h5f_preprocessed_dino.close()
            self._h5f_preprocessed_dino = None

    def __del__(self):
        self.close_connection_to_preprocessed_dino_h5_file()
        super().__del__()
