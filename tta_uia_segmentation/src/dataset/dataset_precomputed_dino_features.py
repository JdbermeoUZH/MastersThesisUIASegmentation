import random
from typing import Optional, Literal

import h5py
import torch
import numpy as np
from typing import Union
from torch.utils.data import Subset, Dataset, DataLoader

from tta_uia_segmentation.src.dataset.dataset_in_memory import DatasetInMemory
from tta_uia_segmentation.src.dataset.augmentation import apply_data_augmentation
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
        **kwargs,
    ):
        # Assert that wrong kwargs are not passed and set them to None
        kwargs_should_be_none = [
            "rescale_factor",
            "image_size",
            "aug_params",
            "img_transform",
            "deformation",
        ]

        for kwarg in kwargs_should_be_none:
            if kwarg in kwargs and kwargs[kwarg] is not None:
                raise ValueError(f"{kwarg} is not supported for DinoFeatures dataset")
            else:
                kwargs[kwarg] = None

        if "image_transform_args" in kwargs:
            raise ValueError(
                "image_transform_args is not supported for DinoFeatures dataset"
            )
        else:
            kwargs["img_transform"] = {}

        super().__init__(**kwargs)

        with h5py.File(self.path, "r") as data:
            self.n_augmentation_epochs = data["n_augmentation_epochs"][()]
            self.dataset_original_size = data["dataset_original_size"][()]

            precomputed_features_attrs = f["precomputed_features"].attrs

    def get_preprocessed_images(
        self,
        index: int,
        as_onehot: bool = True,
        format: Literal["DCHW", "1CDHW"] = "1CDHW",
        same_position_as_original: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if not same_position_as_original:
            raise ValueError(
                "DinoFeatures can only return processed images before computing"
                + "DinoFeatures with `same_position_as_original=True`"
            )

        return super().get_preprocessed_images(index, as_onehot, format, True)

    def __getitem__(self, index):
        assert index <= self.dataset_original_size, "Index out of bounds"

        if not self.apply_augmentation:
            return super().__getitem__(index)

        else:
            # Precomputed features for augmented images
            #  are stored in [self.dataset_original_size:]
            #  for self.n_augmentation_epochs
            aug_epoch_to_sample = random.choice(self.n_augmentation_epochs) + 1
            index += aug_epoch_to_sample * self.dataset_original_size
            return super().__getitem__(index)

    def __len__(self):
        return self.dataset_original_size
