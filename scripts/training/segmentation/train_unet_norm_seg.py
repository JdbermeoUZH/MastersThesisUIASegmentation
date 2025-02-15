import os
import sys
import argparse

import wandb
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from tta_uia_segmentation.src.dataset.io import get_datasets
from tta_uia_segmentation.src.models.io import define_and_possibly_load_norm_seg
from tta_uia_segmentation.src.train import SegTrainer
from tta_uia_segmentation.src.utils.loss import DiceLoss
from tta_uia_segmentation.src.utils.io import (
    load_config,
    dump_config,
    print_config,
    write_to_csv,
    rewrite_config_arguments,
)
from tta_uia_segmentation.src.utils.utils import (
    seed_everything,
    define_device,
    parse_bool,
)
from tta_uia_segmentation.src.utils.logging import setup_wandb


TRAIN_MODE = "segmentation"


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
        "model_config_file",
        type=str,
        help="Path to yaml config file with parameters that define the model.",
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
    parser.add_argument(
        "--checkpoint_last",
        type=str,
        help="Name of last checkpoint file. Default: checkpoint_last.pth",
    )
    parser.add_argument(
        "--checkpoint_best",
        type=str,
        help="Name of best checkpoint file. Default: checkpoint_best.pth",
    )

    # Model parameters
    # :=========================================================================:
    # Normalization module
    parser.add_argument(
        "--kernel_size",
        type=int,
        help="Kernel size for the convolutional layers of the normalization module. Default: 3",
    )

    # Segmentation module
    parser.add_argument(
        "--channel_size",
        type=int,
        nargs="+",
        help="Number of feature maps for each block. Default: [16, 32, 64]",
    )
    parser.add_argument(
        "--channels_bottleneck",
        type=int,
        help="Number of channels in bottleneck layer of model. Default: 128",
    )
    parser.add_argument(
        "--skips",
        type=parse_bool,
        help="Whether to use skip connections on each block, specified as a space-separated list of booleans (True or False)"
        "Default: True True True",
    )
    parser.add_argument(
        "--n_dimensions",
        type=int,
        help="Number of dimensions of the model, i.e: 1D, 2D, or 3D. Default: 2",
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
        "--optimizer_type", type=str, help='Type of optimizer to use. Default: "adam"'
    )
    parser.add_argument(
        "--weight_decay", type=float, help="Weight decay for optimizer. Default: 1e-5"
    )
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate for optimizer. Default: 1e-4"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        help="Maximum gradient norm for gradient clipping. Default: 1.0",
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training. Default: 16"
    )
    parser.add_argument(
        "--grad_acc_steps",
        type=int,
        help="Number of gradient accumulation steps. Default: 1",
    )
    parser.add_argument(
        "--warmup_steps",
        type=float,
        help="Number of warmup steps for scheduler. Default: 0.",
    )
    parser.add_argument(
        "--lr_decay",
        type=parse_bool,
        help="Whether to use linear learning rate decay. Default: False",
    )
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers for dataloader. Default: 0"
    )
    parser.add_argument(
        "--validate_every", type=int, help="Validate every n epochs. Default: 1"
    )
    parser.add_argument(
        "--seed", type=int, help="Seed for random number generators. Default: 0"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for training. Default cuda",
    )

    # Loss function
    parser.add_argument(
        "--loss_type",
        type=str,
        help="Type of loss function to use. Default: dice",
        choices=["dice", "ce"],
    )
    parser.add_argument(
        "--smooth", type=float, help="Smoothing factor for dice loss. Default: 1e-5"
    )
    parser.add_argument(
        "--epsilon", type=float, help="Epsilon factor for dice loss. Default: 1e-10"
    )
    parser.add_argument(
        "--fg_only",
        type=parse_bool,
        help="Whether to calculate dice loss only on foreground. Default: True",
    )
    parser.add_argument(
        "--debug_mode",
        type=parse_bool,
        help="Whether to run in debug mode. Default: False",
    )

    # Dataset and its transformations to use for training
    # :=========================================================================:
    parser.add_argument(
        "--dataset", type=str, help="Name of dataset to use for training. Default: USZ"
    )

    args = parser.parse_args()

    return args


def get_configuration_arguments() -> tuple[dict, dict, dict]:
    args = preprocess_cmd_args()

    dataset_config = load_config(args.dataset_config_file)
    dataset_config = rewrite_config_arguments(dataset_config, args, "dataset")

    model_config = load_config(args.model_config_file)
    model_config = rewrite_config_arguments(model_config, args, "model")

    train_config = load_config(args.train_config_file)
    train_config = rewrite_config_arguments(train_config, args, "train")

    train_config["train_mode"] = TRAIN_MODE

    train_config[TRAIN_MODE] = rewrite_config_arguments(
        train_config[TRAIN_MODE], args, f"train, {TRAIN_MODE}"
    )

    train_config[TRAIN_MODE]["augmentation"] = rewrite_config_arguments(
        train_config[TRAIN_MODE]["augmentation"],
        args,
        f"train, {TRAIN_MODE}, augmentation",
    )

    return dataset_config, model_config, train_config


if __name__ == "__main__":

    print(f"Running {__file__}")

    # Loading general parameters
    # :=========================================================================:
    dataset_config, model_config, train_config = get_configuration_arguments()

    params = {"datset": dataset_config, "model": model_config, "training": train_config}
    resume = train_config["resume"]
    seed = train_config["seed"]
    device = train_config["device"]
    wandb_log = train_config["wandb_log"]
    wandb_run_name = train_config["wandb_run_name"]
    logdir = train_config[TRAIN_MODE]["logdir"]
    wandb_project = train_config[TRAIN_MODE]["wandb_project"]

    seed_everything(seed)
    device = define_device(device)

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
    if wandb_log:
        wandb_dir = setup_wandb(params, logdir, wandb_project, run_name=wandb_run_name)
    else:
        wandb_dir = None
    # Define the dataset that is to be used for training
    # :=========================================================================:
    print("Defining dataset")
    splits = ["train", "val"]

    dataset_name = train_config["dataset"]
    n_classes = dataset_config[dataset_name]["n_classes"]

    dataset_kwargs = dict(
        dataset_name=dataset_name,
        paths_preprocessed=dataset_config[dataset_name]["paths_preprocessed"],
        paths_original=dataset_config[dataset_name]["paths_original"],
        resolution_proc=dataset_config[dataset_name]["resolution_proc"],
        n_classes=n_classes,
        dim_proc=dataset_config[dataset_name]["dim"],
        aug_params=train_config[TRAIN_MODE]["augmentation"],
        load_original=False,
    )

    # Dataset definition
    train_dataset, val_dataset = get_datasets(
        dataset_type="Normal", splits=splits, **dataset_kwargs
    )

    # Define the dataloaders that will be used for training
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

    print("Dataloaders defined")

    # Define the 2D segmentation model
    # :=========================================================================:
    norm_seg = define_and_possibly_load_norm_seg(
        n_classes=n_classes,
        model_params_norm=model_config["normalization_2D"],
        model_params_seg=model_config["segmentation_2D"],
        cpt_fp=cpt_fp,
        device=device,
    )

    # Define the Trainer that will be used to train the model
    # :=========================================================================:
    print("Defining trainer: training loop, optimizer and loss")

    loss_type = train_config[TRAIN_MODE]["loss_type"]
    if loss_type == "dice":
        loss = DiceLoss(
            smooth=float(train_config[TRAIN_MODE]["smooth"]),
            epsilon=float(train_config[TRAIN_MODE]["epsilon"]),
            debug_mode=train_config[TRAIN_MODE]["debug_mode"],
            fg_only=train_config[TRAIN_MODE]["fg_only"],
        )
    elif loss_type == "ce":
        loss = CrossEntropyLoss()
    else:
        raise ValueError(f"Loss type {loss_type} not supported")
    

    trainer = SegTrainer(
        seg=norm_seg,
        learning_rate=float(train_config[TRAIN_MODE]["learning_rate"]),
        loss_func=loss,
        is_resumed=is_resumed,
        optimizer_type=train_config[TRAIN_MODE]["optimizer_type"],
        weight_decay=float(train_config[TRAIN_MODE]["weight_decay"]),
        grad_acc_steps=train_config[TRAIN_MODE]["grad_acc_steps"],
        max_grad_norm=train_config[TRAIN_MODE]["max_grad_norm"],
        checkpoint_best=train_config["checkpoint_best"],
        checkpoint_last=train_config["checkpoint_last"],
        device=device,
        logdir=logdir,
        wandb_log=wandb_log,
        wandb_dir=wandb_dir,
    )

    if wandb_log:
        wandb.save(
            os.path.join(wandb_dir, trainer.last_checkpoint_name),  # type: ignore
            base_path=wandb_dir,
        )
        wandb.watch(list(norm_seg.trainable_modules.values()), trainer.loss_function, log="all")

    # Start training
    # :=========================================================================:
    validate_every = train_config[TRAIN_MODE]["validate_every"]

    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=train_config[TRAIN_MODE]["epochs"],
        validate_every=validate_every,
        warmup_steps=train_config[TRAIN_MODE]["warmup_steps"],
        lr_decay=train_config[TRAIN_MODE]["lr_decay"],
    )

    train_losses = trainer.training_losses
    validation_losses = np.repeat(trainer.validation_losses, validate_every)
    validation_losses = validation_losses[: len(train_losses)]
    validation_scores = np.repeat(trainer.validation_scores, validate_every)
    validation_scores = validation_scores[: len(train_losses)]
    write_to_csv(
        path=os.path.join(logdir, "training_statistics.csv"),
        data=np.stack(
            [
                train_losses,
                validation_losses,
                validation_scores,
            ],
            1,
        ),
        header=["training_losses", "validation_losses", "validation_scores"],
    )

    if wandb_log:
        wandb.finish()
