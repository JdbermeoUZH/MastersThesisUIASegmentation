import os
import sys
import argparse
from typing import Optional

import wandb
import numpy as np
import torch
from tqdm import tqdm
import joblib
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from tta_uia_segmentation.src.models.pca.IncrementalPCA import IncrementalPCA
from tta_uia_segmentation.src.models.pca.utils import flatten_pixels
from tta_uia_segmentation.src.dataset.io import get_datasets
from tta_uia_segmentation.src.utils.io import (
    load_config,
    dump_config,
    print_config,
    rewrite_config_arguments,
)
from tta_uia_segmentation.src.utils.utils import (
    seed_everything,
    define_device,
    parse_bool,
    torch_to_numpy,
)
from tta_uia_segmentation.src.utils.logging import setup_wandb


TRAIN_MODE = "incremental_pca"


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
        "train_config_file",
        type=str,
        help="Path to yaml config file with parameters for training.",
    )

    # I/O and logging
    # :=========================================================================:
    parser.add_argument(
        "--logdir",
        type=str,
        help="Path to directory where logs and checkpoints are saved. Default: logs",
    )
    parser.add_argument(
        "--wandb_log", type=parse_bool, help="Log training to wandb. Default: False."
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="Name of wandb run. Default: None",
    )
    # Model parameters
    # :=========================================================================:
    parser.add_argument(
        "--dino_model",
        type=str,
        help='Name of DINO model to use. Default: "large"',
        choices=["small", "base", "large", "giant"],
    )
    parser.add_argument(
        "--num_pca_components", type=int, help="Number of PCA components"
    )
    parser.add_argument(
        "--precalculated_fts",
        type=parse_bool,
        help="Whether to use precalculated features. Default: True",
    )

    # Training loop
    # :=========================================================================:
    parser.add_argument(
        "--resume",
        type=parse_bool,
        help="Resume training from last checkpoint. Default: True.",
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs to train. Default: 100"
    )
    parser.add_argument(
        "--check_loss_every",
        type=int,
        help="Check loss every n iterations. Default: 10",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        help="Save checkpoint every n iterations. Default: 100",
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training. Default: 10"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers for dataloader. Default: 1",
    )
    # Dataset and its transformations to use for training
    # :=========================================================================:
    parser.add_argument(
        "--dataset", type=str, help="Name of dataset to use for training. Default: USZ"
    )
    parser.add_argument(
        "--hierarchy_level",
        type=int,
        help="Hierarchy level for DINO dataset. Default: 2",
    )
    parser.add_argument(
        "--move_data_to_node",
        type=parse_bool,
        help="Move data to node before training. Default: True",
    )

    args = parser.parse_args()

    return args


def get_configuration_arguments() -> tuple[dict, dict]:
    args = preprocess_cmd_args()

    dataset_config = load_config(args.dataset_config_file)
    dataset_config = rewrite_config_arguments(dataset_config, args, "dataset")

    train_config = load_config(args.train_config_file)
    train_config = rewrite_config_arguments(train_config, args, "train")

    train_config["train_mode"] = TRAIN_MODE

    train_config[TRAIN_MODE] = rewrite_config_arguments(
        train_config[TRAIN_MODE], args, f"train, {TRAIN_MODE}"
    )

    return dataset_config, train_config


def measure_error_on_dataloader(
    dataloader: DataLoader,
    split: str,
    ipca: IncrementalPCA,
    num_components_recosntruct: Optional[int] = None,
) -> float:
    n_samples = 0
    error = 0
    pbar = tqdm(dataloader, desc=f"Measuring error on {split} set")

    for x, *_ in pbar:
        # Flatten the images from NCHW to (N**H*W)C
        if isinstance(x, torch.Tensor):
            x_batch, *_ = flatten_pixels(x)
        elif isinstance(x, list):
            x_batch = torch.cat([flatten_pixels(x_i)[0] for x_i in x], dim=0)
        else:
            raise ValueError("x must be a tensor or a list of tensors")
        n_samples += x_batch.shape[0]

        x_batch_recon = ipca.reconstruct(x_batch, num_components_recosntruct)
 
        batch_mse = mse_loss(
            x_batch.to(x_batch_recon.device), x_batch_recon
        )

        error += float(batch_mse)

    return error / n_samples


if __name__ == "__main__":

    print(f"Running {__file__}")

    # Loading general parameters
    # :=========================================================================:
    dataset_config, train_config = get_configuration_arguments()

    params = {"datset": dataset_config, "training": train_config}
    resume = train_config["resume"]
    seed = train_config["seed"]
    device = train_config["device"]
    wandb_log = train_config["wandb_log"]
    wandb_run_name = train_config["wandb_run_name"]
    logdir = train_config[TRAIN_MODE]["logdir"]
    wandb_project = train_config[TRAIN_MODE]["wandb_project"]

    # Write or load parameters to/from logdir, used if a run is resumed.
    # :=========================================================================:
    is_resumed = os.path.exists(os.path.join(logdir, "params.yaml")) and resume
    print(f"training resumed: {is_resumed}")

    if is_resumed:
        params = load_config(os.path.join(logdir, "params.yaml"))
        cpt_fp = os.path.join(logdir, train_config["checkpoint_last"])
        assert os.path.exists(cpt_fp), f"Checkpoint file {cpt_fp} does not exist"

    else:
        os.makedirs(logdir, exist_ok=True)
        dump_config(os.path.join(logdir, "params.yaml"), params)
        cpt_fp = None

    print_config(params, keys=["training"])

    # Setup wandb logging
    # :=========================================================================:
    if wandb_log:
        wandb_dir = setup_wandb(params, logdir, wandb_project, run_name=wandb_run_name)
    else:
        wandb_dir = None

    # Define the dataset that is to be used for training
    # :=========================================================================:
    print("Defining dataset")
    seed_everything(seed)
    device = define_device(device)

    splits = ["train", "val"]

    dataset_name = train_config["dataset"]
    n_classes = dataset_config[dataset_name]["n_classes"]
    node_data_path = train_config["node_data_path"] if train_config["move_data_to_node"] else None

    dataset_type = (
        "DinoFeatures" if train_config[TRAIN_MODE]["precalculated_fts"] else "Normal"
    )

    dataset_kwargs = dict(
        dataset_name=dataset_name,
        paths_preprocessed=dataset_config[dataset_name]["paths_preprocessed"],
        paths_original=dataset_config[dataset_name]["paths_original"],
        resolution_proc=dataset_config[dataset_name]["resolution_proc"],
        n_classes=n_classes,
        dim_proc=dataset_config[dataset_name]["dim"],
        aug_params=train_config[TRAIN_MODE]["augmentation"],
        load_original=False,
        node_data_path=node_data_path,
    )

    if dataset_type == "DinoFeatures":
        dataset_kwargs["paths_preprocessed_dino"] = dataset_config[dataset_name][
            "paths_preprocessed_dino"
        ]
        dataset_kwargs["hierarchy_level"] = train_config[TRAIN_MODE]["hierarchy_level"]
        dataset_kwargs["dino_model"] = train_config[TRAIN_MODE]["dino_model"]
        del dataset_kwargs["aug_params"]

    # Dataset definition
    train_dataset, val_dataset = get_datasets(
        dataset_type=dataset_type, splits=splits, **dataset_kwargs
    )

    # Define dataloaders
    batch_size = train_config[TRAIN_MODE]["batch_size"]
    num_workers = train_config[TRAIN_MODE]["num_workers"]

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # Initialize IncrementalPCA and scaler
    pca_components = train_config[TRAIN_MODE]["pca_components"]
    epochs = train_config[TRAIN_MODE]["epochs"]
    check_loss_every = train_config[TRAIN_MODE]["check_loss_every"]
    explained_var_pcs = train_config[TRAIN_MODE]["explained_var_pcs"]
    checkpoint_every = train_config[TRAIN_MODE]["checkpoint_every"]

    ipca = IncrementalPCA(n_components=pca_components)

    # Define the incremental PCA model
    # :=========================================================================:
    best_val_error = np.inf
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        n_samples = 0
        for step_i, (x, *_) in enumerate(tqdm(train_dataloader)):
            # Flatten the images from NCHW to (N**H*W)C
            if isinstance(x, torch.Tensor):
                x_batch, *_ = flatten_pixels(x)
            elif isinstance(x, list):
                x_batch = torch.cat([flatten_pixels(x_i)[0] for x_i in x], dim=0)
            else:
                raise ValueError("x must be a tensor or a list of tensors")

            n_samples += x_batch.shape[0]

            # Fit the IncrementalPCA model
            ipca.partial_fit(x_batch)

            # Move calculated PCs to device for faster reconstruction
            ipca.to_device(device)

            if check_loss_every is not None and step_i % check_loss_every == 0:
                if explained_var_pcs is not None:
                    num_components = ipca.num_components_to_keep(explained_var_pcs)
                else:
                    num_components = None

                train_error = measure_error_on_dataloader(
                    train_dataloader, "train", ipca, num_components
                )
                val_error = measure_error_on_dataloader(
                    val_dataloader, "val", ipca, num_components
                )

                if wandb_log:
                    wandb.log({"train_error": train_error, "val_error": val_error})

                print(
                    f"Step: {step_i}, Train error: {train_error}, Val error: {val_error}"
                )

                if val_error < best_val_error:
                    best_val_error = val_error
                    ipca.save(os.path.join(logdir, "ipca_best.pkl"))

            if checkpoint_every is not None and step_i % checkpoint_every == 0:
                ipca.save(os.path.join(logdir, f"ipca_step_{step_i}.pkl"))

    print("Training done.")

    # Save the final model
    ipca.save(os.path.join(logdir, "ipca_last.pkl"))
