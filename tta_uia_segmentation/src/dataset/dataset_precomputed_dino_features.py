import random
from typing import Optional, Literal

import h5py
import torch
import numpy as np
from typing import Union
from torch.utils.data import Subset, Dataset, DataLoader

from tta_uia_segmentation.src.dataset.dataset import Dataset, ExtraInputs, ExtraInputsEmpty
import tta_uia_segmentation.src.dataset.utils as du
from tta_uia_segmentation.src.utils.utils import get_seed
from tta_uia_segmentation.src.utils.loss import dice_score


def get_datasets(
    splits,
    *args,
    **kwargs,
):

    datasets = []

    for split in splits:
        datasets.append(DatasetDinoFeatures(split=split, *args, **kwargs))

    return datasets


class DatasetDinoFeatures(Dataset):
    def __init__(
        self,
        paths_preprocessed_dino: str,
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
            assert kwargs["mode"] == "2D", "DinoFeatures only precalculated on 2D slices"
        else:
            kwargs["mode"] = "2D"

        if "orientation" in kwargs:
            assert kwargs["orientation"] == "depth", "DinoFeatures only precalculated on depth-wise slices"
        else:
            kwargs["orientation"] = "depth"

        if "load_in_memory" in kwargs:
            assert not kwargs["load_in_memory"], "DinoFeatures dataset is too large to load in memory"
        else:
            kwargs["load_in_memory"] = False

        super().__init__(**kwargs)

        # Define attributes specific to this class
        # :====================================================================:
        self._paths_preprocessed_dino = paths_preprocessed_dino
        
        with h5py.File(paths_preprocessed_dino, "r") as data:
            self.n_augmentation_epochs = data["n_augmentation_epochs"][()]
            self.dataset_original_size = data["dataset_original_size"][()]

            assert self.dataset_original_size == self._length, "Dataset length mismatch"

            precomputed_features_attrs = f["precomputed_features"].attrs
        
        self._h5f_preprocessed_dino = None
        self._images_preprocessed_dino = None
        self._labels_preprocessed_dino = None

    def _define_dataset_length(
        self, check_dims_proc_and_orig_match: bool = False
    ) -> None:
        """
        Dataset length is forced to be the same as the number of depth-wise slices
        """
        with h5py.File(self._path_original, "r") as data:
            self._length = data["images"].shape[0] * data["images"].shape[1]

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, dict[str, ExtraInputs]]:
        """
        Retrieve Dino Features for a given index

        Dino features are also precomputed for precomputed augmentations
        """
        if self.augment:    
            # Precomputed features for augmented images
            #  are stored in [self.dataset_original_size:]
            #  for self.n_augmentation_epochs
            aug_epoch_to_sample = random.choice(self.n_augmentation_epochs) + 1
            index += aug_epoch_to_sample * self.dataset_original_size
            print("TODO: Implement non-augmented DinoFeatures")
        else:
            print("TODO: Implement non-augmented DinoFeatures")


    def get_preprocessed_image(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get preprocessed slice, typically used for training directly on images
        """

        return super().__getitem__(index)
    
    def _open_connection_to_preprocessed_dino_h5_file(self):
        if self._h5f_preprocessed is None:
            self._h5f_preprocessed_dino = h5py.File(self._path_preprocessed, "r")
            self._images_preprocessed_dino = self._h5f_preprocessed["images"]
            self._labels_preprocessed_dino = self._h5f_preprocessed["labels"]

    def close_connection_to_preprocessed_dino_h5_file(self):
        if self._h5f_preprocessed is not None:
            self._h5f_preprocessed_dino.close()
            self._h5f_preprocessed_dino = None