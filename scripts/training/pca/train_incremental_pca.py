import os
import sys
import argparse

import wandb
import numpy as np
from torch import Tensor
from tqdm import tqdm
import joblib
from torch.utils.data import DataLoader
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

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
    torch_to_numpy
)
from tta_uia_segmentation.src.utils.logging import setup_wandb


TRAIN_TYPE = "incremental_pca"


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

    args = parser.parse_args()

    return args


def get_configuration_arguments() -> tuple[dict, dict]:
    args = preprocess_cmd_args()

    dataset_config = load_config(args.dataset_config_file)
    dataset_config = rewrite_config_arguments(dataset_config, args, "dataset")

    train_config = load_config(args.train_config_file)
    train_config = rewrite_config_arguments(train_config, args, "train")

    train_config[TRAIN_TYPE] = rewrite_config_arguments(
        train_config[TRAIN_TYPE], args, f"train, {TRAIN_TYPE}"
    )

    train_config[TRAIN_TYPE]["augmentation"] = rewrite_config_arguments(
        train_config[TRAIN_TYPE]["augmentation"],
        args,
        f"train, {TRAIN_TYPE}, augmentation",
    )

    return dataset_config, train_config


def measure_reconstruction_error(x: np.ndarray, ipca: IncrementalPCA, scaler: StandardScaler) -> float:
    x_normalized = scaler.transform(x)
    x_reconstructed = ipca.inverse_transform(ipca.transform(x_normalized))
    breakpoint()
    return float(mean_squared_error(x_normalized, x_reconstructed))


def measure_error_on_dataloader(dataloader: DataLoader, split: str, ipca: IncrementalPCA, scaler: StandardScaler) -> float:
    n_samples = 0
    error = 0
    pbar = tqdm(
        dataloader,
        desc=f"Measuring error on {split} set"
    )
    
    for x, *_ in pbar:
        x_batch = x.view(-1, x.shape[0])
        n_samples += x.shape[0]
        x_batch = torch_to_numpy(x_batch)

        error += measure_reconstruction_error(x_batch, ipca, scaler)

    return error/n_samples



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
    logdir = train_config[TRAIN_TYPE]["logdir"]
    wandb_project = train_config[TRAIN_TYPE]["wandb_project"]

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

    print_config(params, keys=["training", "model"])

    # Setup wandb logging
    # :=========================================================================:
    wandb_dir = setup_wandb(params, logdir, wandb_project) if wandb_log else None

    # Define the dataset that is to be used for training
    # :=========================================================================:
    print("Defining dataset")
    seed_everything(seed)
    device = define_device(device)

    splits = ["train", "val"]

    dataset_name = train_config["dataset"]
    n_classes = dataset_config[dataset_name]["n_classes"]
    batch_size = train_config[TRAIN_TYPE]["batch_size"]
    num_workers = train_config[TRAIN_TYPE]["num_workers"]
    dataset_type = (
        "DinoFeatures" if train_config[TRAIN_TYPE]["precalculated_fts"] 
        else "Normal"
    )

    dataset_kwargs = dict(
        dataset_name=dataset_name,
        paths_preprocessed=dataset_config[dataset_name]["paths_preprocessed"],
        paths_original=dataset_config[dataset_name]["paths_original"],
        resolution_proc=dataset_config[dataset_name]["resolution_proc"],
        n_classes=n_classes,
        dim_proc=dataset_config[dataset_name]["dim"],
        aug_params=train_config[TRAIN_TYPE]["augmentation"],
        load_original=False,
    )

    if dataset_type == "DinoFeatures":
        dataset_kwargs["paths_preprocessed_dino"] = dataset_config[dataset_name][
            "paths_preprocessed_dino"
        ]
        dataset_kwargs["hierarchy_level"] = train_config[TRAIN_TYPE]["hierarchy_level"]
        dataset_kwargs["dino_model"] = train_config[TRAIN_TYPE]["dino_model"]
        del dataset_kwargs["aug_params"]
        
    # Dataset definition
    train_dataset, val_dataset = get_datasets(
        dataset_type=dataset_type, splits=splits, **dataset_kwargs
    )

    # Define dataloaders
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
    pca_components = train_config[TRAIN_TYPE]["num_pca_components"]
    epochs = train_config["epochs"]
    check_loss_every = train_config[TRAIN_TYPE]["check_loss_every"]

    scaler = StandardScaler()
    ipca = IncrementalPCA(n_components=pca_components)

    # Define the incremental PCA model
    # :=========================================================================:
    best_val_error = np.inf
    for epoch in epochs:
        print(f"Epoch {epoch}")
        n_samples = 0
        for step_i, (x, *_) in enumerate(tqdm(train_dataloader)):
            # Flatten the images from NCHW to (N**H*W)C
            x_batch = x.view(-1, x.shape[0])
            n_samples += x_batch.shape[0]

            # Convert to numpy array
            x_batch = torch_to_numpy(x_batch)

            # Normalize the data
            scaler.partial_fit(x_batch)  # Update running mean and std
            x_batch_normalized = scaler.transform(x_batch) 

            # Fit the IncrementalPCA model
            ipca.partial_fit(x_batch_normalized)

            if step_i % check_loss_every == 0:
                train_error = measure_error_on_dataloader(train_dataloader, "train", ipca, scaler)
                val_error = measure_error_on_dataloader(val_dataloader, "val", ipca, scaler)

                if wandb_log:
                    wandb.log({"train_error": train_error, "val_error": val_error})

                print(f"Step: {step_i}, Train error: {train_error}, Val error: {val_error}")

                # Save the model
                joblib.dump(ipca, os.path.join(logdir, f"ipca_latest_step_{step_i}.pkl"))
                joblib.dump(scaler, os.path.join(logdir, f"scaler_latest_step_{step_i}.pkl"))

                if val_error < best_val_error:
                    best_val_error = val_error
                    joblib.dump(ipca, os.path.join(logdir, "ipca_best.pkl"))
                    joblib.dump(scaler, os.path.join(logdir, "scaler_best.pkl"))
    
    print("Training done.")

    # Save the final model
    joblib.dump(ipca, os.path.join(logdir, "ipca_final.pkl"))
    joblib.dump(scaler, os.path.join(logdir, "scaler_final.pkl"))




