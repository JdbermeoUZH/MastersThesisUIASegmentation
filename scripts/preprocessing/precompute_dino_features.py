import os
import sys
import math
import argparse
from typing import Literal, Optional

import h5py
import torch
import torch.nn.functional as F
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

from tta_uia_segmentation.src.dataset.io import get_datasets
import tta_uia_segmentation.src.dataset.utils as du
from tta_uia_segmentation.src.utils.io import (
    load_config,
    dump_config,
    rewrite_config_arguments,
)
from tta_uia_segmentation.src.utils.utils import (
    seed_everything,
    define_device,
    torch_to_numpy,
    parse_bool,
)
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

    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers to use for evaluation. Default: 2",
    )

    # Dataset parameters
    # :==================================================:
    parser.add_argument(
        "--dataset", type=str, help="Name of dataset to use for training. Default: USZ"
    )

    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        help="Splits of the dataset to use for training. Default: ['train']",
    )

    parser.add_argument(
        "--compression",
        type=str,
        help="Compression to use for the hdf5 file. Default: None",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Chunk size to use for the hdf5 file. Default: None",
    )

    parser.add_argument(
        "--auto_chunk",
        type=parse_bool,
        help="Whether to use automatic chunking for the hdf5 file. Default: True",
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
        "--out_dir_suffix", 
        type=str,
        help="Suffix to add to the output directory. Default: ''",
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
    hierarchy_level: int,
    dino_fe: DinoV2FeatureExtractor,
    n_augmentation_epochs: int,
    dataset_original_size: int,
    compression: Optional[Literal['gzip', 'lzf'] | int] = None,
    chunk_size: int | bool = True,
) -> h5py.File:
    hdf5_file = h5py.File(h5_path, "w")

    # Dump configuration to hdf5 file

    # Dataset for Dino features
    # :=========================================================================:
    for hierarchy_i in range(hierarchy_level + 1):
        upsample = 2**hierarchy_i

        hierarchy_i_img_h = image_size[0] * upsample
        hierarchy_i_img_w = image_size[1] * upsample

        feature_spatial_size = (
            dino_fe.emb_dim,
            math.ceil(hierarchy_i_img_h / dino_fe.patch_size),
            math.ceil(hierarchy_i_img_w / dino_fe.patch_size),
        )

        if not isinstance(chunk_size, bool) and isinstance(chunk_size, int):
            chunk_size_fe = (1, chunk_size, *feature_spatial_size[1:])
        else:
            chunk_size_fe = True

        # N, dino_fe, H * 2^hier, W*2^hier
        hdf5_file.create_dataset(
            f"images_hier_{hierarchy_i}",
            shape=(0, *feature_spatial_size),
            maxshape=(None, *feature_spatial_size),
            dtype=np.float32,
            compression=compression,
            chunks=chunk_size_fe,
        )

    # Add attributes to the dataset about the Dino model used
    hdf5_file.attrs["patch_size"] = dino_fe.patch_size
    hdf5_file.attrs["emb_dim"] = dino_fe.emb_dim
    hdf5_file.attrs["dino_model"] = dino_fe.model_name
    hdf5_file.attrs["n_augmentation_epochs"] = n_augmentation_epochs
    hdf5_file.attrs["dataset_original_size"] = dataset_original_size
    hdf5_file.attrs["hierarchy_level"] = hierarchy_level
    
    # Dataset for labels that correspond to each image
    # :=========================================================================:
    labels_spatial_size = image_size[:2]

    # N, H, W
    hdf5_file.create_dataset(
        "labels",
        shape=(0, *labels_spatial_size),
        maxshape=(None, *labels_spatial_size),
        dtype=np.float32,
        compression=compression,
        chunks=True,
    )

    return hdf5_file


def append_to_h5dataset(h5file: h5py.File, dataset_name: str, data: np.ndarray) -> None:
    new_rows = data.shape[0]
    dataset_size = hdf5_file[dataset_name].shape[0]

    # Expand dataset by new rows
    h5file[dataset_name].resize(dataset_size + new_rows, axis=0)

    # Append features to the dataset
    h5file[dataset_name][-new_rows:] = data


def add_precomputed_dino_features_to_h5(
    hdf5_file: h5py.File,
    dl: DataLoader,
    dino_fe: DinoV2FeatureExtractor,
    hierarchy_level: int,
    device: torch.device,
    pbar_desc: str = "",
) -> None:
    pbar = tqdm(
        dl,
        desc=pbar_desc,
    )
    for image, labels, *_ in pbar:
        # Expand to NCHW
        image = image.to(device)
        labels = labels.to(device)

        # Convert images to RGB
        if image.shape[1] == 1:
            image = du.grayscale_to_rgb(image)

        # Add Dino features to the hdf5 file
        for hierarchy_i in range(hierarchy_level + 1):
            # get dino features for each hierarchy level
            dino_out = dino_fe.forward(image, hierarchy=hierarchy_i)

            features_i = x = dino_out["patch"].permute(
                0, 3, 1, 2
            )  # N x np x np x df -> N x df x np x np
            append_to_h5dataset(
                hdf5_file,
                f"images_hier_{hierarchy_i}",
                torch_to_numpy(features_i),
            )

        # Add labels to the hdf5 file
        labels = du.onehot_to_class(labels)
        labels = labels.squeeze(1)  # N x 1 x H x W -> N x H x W
        append_to_h5dataset(hdf5_file, "labels", torch_to_numpy(labels))


if __name__ == "__main__":
    print(f"Running {__file__}")

    # Loading general parameters
    # :=========================================================================:

    dataset_config, preproc_config = get_configuration_arguments()

    seed = preproc_config["seed"]
    device = preproc_config["device"]

    seed_everything(seed)
    device = define_device(device)

    # Load dataset
    # :=========================================================================:
    dataset_name = preproc_config["dataset"]
    splits = preproc_config["splits"]
    n_classes = dataset_config[dataset_name]["n_classes"]
    num_aug_epochs = preproc_config[PREPROCESSING_TYPE]["augmentation"].pop(
        "num_epochs"
    )
    aug_params = preproc_config[PREPROCESSING_TYPE]["augmentation"]

    print(f"loading splits {splits} of dataset {preproc_config['dataset']}")

    # Dataset definition
    datasets = get_datasets(
        dataset_type='Normal',
        dataset_name=dataset_name,
        paths_preprocessed=dataset_config[dataset_name]["paths_preprocessed"],
        paths_original=dataset_config[dataset_name]["paths_original"],
        splits=splits,
        resolution_proc=dataset_config[dataset_name]["resolution_proc"],
        dim_proc=dataset_config[dataset_name]["dim"],
        n_classes=n_classes,
        aug_params=aug_params,
        mode="2D",
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
    dino_type = preproc_config[PREPROCESSING_TYPE]["dino_model"]
    dino_fe = DinoV2FeatureExtractor(dino_type).to(device)

    # Write new hdf5 file with precomputed features
    # :=========================================================================:
    hierarchy_level = preproc_config[PREPROCESSING_TYPE]["hierarchy_levels"]
    chunk_size = preproc_config[PREPROCESSING_TYPE]["chunk_size"]
    compression = preproc_config[PREPROCESSING_TYPE]["compression"]
    auto_chunk = preproc_config[PREPROCESSING_TYPE]["auto_chunk"]
    out_dir_suffix = preproc_config[PREPROCESSING_TYPE]["out_dir_suffix"]

    if auto_chunk:
        chunk_size = True

    if compression not in ['gzip', 'lzf', None]:
        compression = int(compression)
    
    for split, dataloader in dataloaders.items():
        print(f"Computing features for {split} split")

        image_size = dataloader.dataset.dim_proc

        # Create hdf5 file object
        path_preprocessed = dataloader.dataset.path_preprocessed
        out_dir = os.path.join(
            os.path.dirname(path_preprocessed),
            f'_dino_features_{out_dir_suffix}',
        )
        os.makedirs(out_dir, exist_ok=True)
        
        h5filename = os.path.basename(path_preprocessed).replace('.hdf5', '')
        h5filename += f"_dino_{dino_type}_hier_{hierarchy_level}.hdf5"

        h5path = os.path.join(out_dir, h5filename)

        hdf5_file = create_hdf5_datasets(
            h5path,
            image_size,
            hierarchy_level,
            dino_fe,
            num_aug_epochs,
            len(dataloader.dataset),
            compression=compression,
            chunk_size=chunk_size,
        )

        # Iterate once over the dataset without data augmentation
        dataloader.dataset.augment = False

        add_precomputed_dino_features_to_h5(
            hdf5_file,
            dataloader,
            dino_fe,
            hierarchy_level,
            device,
            pbar_desc=f"Computing features without data augmentation for {split} split",
        )

        # Iterate over the dataset with data augmentation for n epochs
        dataloader.dataset.augment = True
        print(
            f"Computing features with data augmentation for {split} split"
            f" for {num_aug_epochs} epochs"
        )
        for epoch in range(num_aug_epochs):
            add_precomputed_dino_features_to_h5(
                hdf5_file,
                dataloader,
                dino_fe,
                hierarchy_level,
                device,
                pbar_desc=f"Computing features with data augmentation for {split} split, epoch {epoch}",
            )

        hdf5_file.close()

    print("Done")
