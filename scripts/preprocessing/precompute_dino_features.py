import os
import sys
import math
import shutil
import argparse

import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(
    os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
        )
    )
)

from tta_uia_segmentation.src.dataset.dataset_in_memory import get_datasets
from tta_uia_segmentation.src.utils.io import (
    load_config,
    dump_config,
    print_config,
    rewrite_config_arguments,
)
from tta_uia_segmentation.src.utils.utils import seed_everything, define_device
from tta_uia_segmentation.src.models.seg.dino.DinoV2FeatureExtractor import (
    DinoV2FeatureExtractor,
)


PREPROCESSING_TYPE = "precalculate_dino_features"


def preprocess_cmd_args() -> argparse.Namespace:
    """_
    Parse command line arguments and return them as a Namespace object.

    All options specified will overwrite whaterver is specified in the config files.

    """
    parser = argparse.ArgumentParser(
        description="Train Segmentation Model (with shallow normalization module)"
    )

    parser.add_argument(
        "dataset_config_file",
        type=str,
        help="Path to yaml config file with parameters that define the dataset.",
    )
    parser.add_argument(
        "preprocessing_config_file",
        type=str,
        help="Path to yaml config file with parameters to preprocess the dataset.",
    )

    # Precomputed feature parameters
    # :==================================================:
    parser.add_argument(
        "--dino_model",
        type=str,
        choices=["small", "base", "large", "giant"],
        help='Size of DINO model to use. Default: "large"',
    )
    parser.add_argument(
        "--hierarchy_levels",
        type=int,
        help="Hierarchy levels to use for feature extraction. Default: 2",
    )

    # Evaluation parameters
    # :==================================================:
    parser.add_argument(
        "--batch_size", type=int, help="Batch size to use for evaluation. Default: 32"
    )

    # Dataset parameters
    # :==================================================:
    parser.add_argument(
        "--dataset", type=str, help="Name of dataset to use for training. Default: USZ"
    )

    # Data augmentation parameters
    # :==================================================:
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs of data augmented examples to store. Default: 20",
    )

    # I/O parameters
    # :==================================================:
    parser.add_argument(
        "--logdir",
        type=str,
        help='Path to directory where logs will be stored. Default: "logs"',
    )

    args = parser.parse_args()

    return args


def get_configuration_arguments() -> tuple[dict, dict]:
    args = preprocess_cmd_args()

    dataset_config = load_config(args.dataset_config_file)
    dataset_config = rewrite_config_arguments(dataset_config, args, "dataset")

    preproc_config = load_config(args.preprocessing_config_file)
    preproc_config = rewrite_config_arguments(preproc_config, args, "preprocessing")

    preproc_config[PREPROCESSING_TYPE] = rewrite_config_arguments(
        preproc_config[PREPROCESSING_TYPE], args, f"preprocessing, {PREPROCESSING_TYPE}"
    )

    preproc_config[PREPROCESSING_TYPE]["augmentation"] = rewrite_config_arguments(
        preproc_config[PREPROCESSING_TYPE]["augmentation"],
        args,
        f"preprocessing, {PREPROCESSING_TYPE}, augmentation",
    )

    return dataset_config, preproc_config


def create_hdf5_datasets(
    h5_path: str,
    image_size: tuple,
    n_classes: int,
    hierarchy_level: int,
    dino_fe: DinoV2FeatureExtractor,
    n_augmentation_epochs: int,
    dataset_original_size: int,
) -> h5py.File:
    hdf5_file = h5py.File(h5_path, "w")

    # Dataset for Dino features
    # :=========================================================================:
    largest_upsample = 2 ** hierarchy_level

    hierarchy_i_img_h = image_size[-2] * largest_upsample
    hierarchy_i_img_w = image_size[-1] * largest_upsample

    feature_spatial_size = (
        dino_fe.emb_dim,
        math.ceil(hierarchy_i_img_h / dino_fe.patch_size),
        math.ceil(hierarchy_i_img_w / dino_fe.patch_size),
    )

    images_dataset = hdf5_file.create_dataset(
        f"images",
        shape=(0, hierarchy_level, *feature_spatial_size),
        maxshape=(None, hierarchy_level, *feature_spatial_size),
        dtype=np.float32,
    )

    # Add attributes to the dataset about the Dino model used
    images_dataset.attrs["patch_size"] = dino_fe.patch_size
    images_dataset.attrs["emb_dim"] = dino_fe.emb_dim
    images_dataset.attrs["dino_model"] = dino_fe.model_name
    images_dataset.attrs["n_augmentation_epochs"] = n_augmentation_epochs
    images_dataset.attrs["dataset_original_size"] = dataset_original_size

    # Dataset for labels that correspond to each image
    # :=========================================================================:
    labels_spatial_size = (
        n_classes,
        image_size[-2],
        image_size[-1],
    )

    hdf5_file.create_dataset(
        "labels",
        shape=(0, *labels_spatial_size),
        maxshape=(None, *labels_spatial_size),
        dtype=np.float32,
    )

    # Image metadata such as field of view and pixel size
    # :=========================================================================:
    per_vol_attributes = ["nx", "ny", "nz", "px", "py", "pz"]
    for attr in per_vol_attributes:
        hdf5_file.create_dataset(attr, shape=(0,), maxshape=(None,), dtype=np.float32)

    return hdf5_file


def append_to_h5dataset(h5file: h5py.File, dataset_name: str, data: np.ndarray) -> None:
    new_rows = data.shape[0]
    dataset_size = hdf5_file[dataset_name].shape[0]

    # Expand dataset by new rows
    h5file[dataset_name].resize(dataset_size + new_rows, axis=0)

    # Append features to the dataset
    h5file[dataset_name][-new_rows:] = data


if __name__ == "__main__":
    print(f"Running {__file__}")

    # Loading general parameters
    # :=========================================================================:

    dataset_config, preproc_config = get_configuration_arguments()

    seed = preproc_config["seed"]
    device = preproc_config["device"]
    logdir = preproc_config["logdir"]

    seed_everything(seed)
    device = define_device(device)

    # Load dataset
    # :=========================================================================:
    print(f"loading splits of dataset {preproc_config['dataset']}")

    dataset_name = preproc_config["dataset"]
    splits = preproc_config["splits"]
    image_size = preproc_config["image_size"]
    n_classes = dataset_config[dataset_name]["n_classes"]

    # Dataset definition
    datasets = get_datasets(
        dataset_name=dataset_name,
        paths=dataset_config[dataset_name]["paths_processed"],
        paths_original=dataset_config[dataset_name]["paths_original"],
        splits=splits,
        image_size=image_size,
        resolution_proc=dataset_config[dataset_name]["resolution_proc"],
        dim_proc=dataset_config[dataset_name]["dim"],
        n_classes=n_classes,
        aug_params=preproc_config[PREPROCESSING_TYPE]["augmentation"],
    )

    dataloaders = {
        split: DataLoader(
            dataset=dataset,
            batch_size=preproc_config["batch_size"],
            shuffle=False,
            num_workers=preproc_config["num_workers"],
            drop_last=False,
        )
        for split, dataset in zip(splits, datasets)
    }

    # Load Dino Feature extractor
    # :=========================================================================:
    dino_fe = DinoV2FeatureExtractor(
        preproc_config[PREPROCESSING_TYPE]["dino_model"]
    ).to(device)

    # Write new hdf5 file with precomputed features
    # :=========================================================================:
    hierarchy_level = preproc_config[PREPROCESSING_TYPE]["hierarchy_levels"]

    for split, dataloader in dataloaders.items():
        print(f"Computing features for {split} split")

        # Create hdf5 file object
        h5path = dataloader.dataset.path.rstrip(".hdf5") + "_dino_features.hdf5"

        hdf5_file = create_hdf5_datasets(
            h5path, image_size, n_classes, hierarchy_level, dino_fe
        )

        feature_spatial_size = hdf5_file["images"].shape[1:]

        # Iterate once over the dataset without data augmentation
        dataloader.dataset.set_augmentation(False)

        pbar = tqdm(
            dataloader,
            desc=f"Computing features without data augmentation",
        )
        
        for image, labels, *_ in pbar:
            image = image.to(device)
            labels = labels.to(device)
            batch_size = image.shape[0]

            # Add Dino features to the hdf5 file
            features_i = []
            for hierarchy_i in range(hierarchy_level):
                features_i = dino_fe.forward(image, hierarchy=hierarchy_i)
                
                features_i_np = np.zeros((batch_size, *feature_spatial_size), dtype=np.float32)
                features_i_np[:, ]
                

            # Add labels to the hdf5 file
            append_to_h5dataset(hdf5_file, "labels", labels)

            # Add per image attributes to the hdf5 file
            for attr in ["nx", "ny", "nz", "px", "py", "pz"]:
                append_to_h5dataset(hdf5_file, attr, getattr(dataloader.dataset, attr))

    # Iterate over the dataset with data augmentation for n epochs
