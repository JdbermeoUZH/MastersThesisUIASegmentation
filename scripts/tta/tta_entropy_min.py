import os
import sys
import copy
import argparse

import torch
import wandb

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tta_uia_segmentation.src.tta import TTAEntropyMin
from tta_uia_segmentation.src.tta.utils import write_summary
from tta_uia_segmentation.src.dataset.io import get_datasets
from tta_uia_segmentation.src.dataset.utils import ensure_nd
from tta_uia_segmentation.src.models.io import (
    define_and_possibly_load_norm_seg,
    define_and_possibly_load_dino_seg,
)
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
    assert_in,
    default,
)
from tta_uia_segmentation.src.utils.logging import setup_wandb


TTA_MODE = "entropy_min"


def preprocess_cmd_args() -> argparse.Namespace:
    """_
    Parse command line arguments and return them as a Namespace object.

    All options specified will overwrite whaterver is specified in the config files.

    """
    parser = argparse.ArgumentParser(description="Test Time Adaption with DAE")

    parser.add_argument(
        "dataset_config_file",
        type=str,
        help="Path to yaml config file with parameters that define the dataset.",
    )
    parser.add_argument(
        "tta_config_file",
        type=str,
        help="Path to yaml config file with parameters for test time adaptation",
    )
    parser.add_argument(
        "--print_config",
        type=parse_bool,
        help="Print the configuration parameters. Default: True",
    )

    # Segmentation model parameters
    # :================================================================================================:
    parser.add_argument(
        "--seg_dir",
        type=str,
        help="Path to directory where segmentation checkpoints are saved",
    )

    parser.add_argument(
        "--viz_interm_outs",
        type=str,
        nargs="*",
        help="Intermediate outputs to visualize. Default: ['Normalized Image'] for norm_seg and [] for dino",
    )

    # TTA parameters. If provided, overrides default parameters from config file.
    # :================================================================================================:
    parser.add_argument(
        "--start", type=int, help="starting volume index to be used for testing"
    )
    parser.add_argument(
        "--stop",
        type=int,
        help="stopping volume index to be used for testing (index not included)",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        help="Path to directory where logs and checkpoints are saved. Default: logs",
    )

    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="Name of wandb run. Default: None",
    )

    parser.add_argument("--classes_of_interest", type=int, nargs="+")

    parser.add_argument(
        "--wandb_project",
        type=str,
        help='Name of wandb project to log to. Default: "tta"',
    )
    parser.add_argument(
        "--wandb_log", type=parse_bool, help="Log tta to wandb. Default: False."
    )
    parser.add_argument(
        "--device", type=str, help='Device to use for training. Default: "cuda"'
    )

    # TTA loop
    # :================================================================================================:
    # Optimization parameters
    parser.add_argument(
        "--num_steps",
        type=int,
        help="Number of steps to take in TTA loop. Default: 100",
    )

    parser.add_argument(
        "--fit_at_test_time",
        type=str,
        choices=["bn_layers", "all", "normalizer"],
        help="Fit at test time. Default: None",
    )

    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate for optimizer. Default: 1e-5"
    )
    parser.add_argument(
        "--weight_decay", type=float, help="Weight decay for optimizer. Default: 1e-4"
    )

    parser.add_argument(
        "--batch_size", type=int, help="Batch size for tta. Default: 64"
    )
    parser.add_argument(
        "--gradient_acc_steps",
        type=int,
        help="Number of gradient accumulation steps. Default: 1",
    )

    parser.add_argument(
        "--lr_decay", type=parse_bool, help="Use lr decay. Default: True"
    )

    parser.add_argument(
        "--lr_scheduler_step_size",
        type=int,
        help="Step size for lr scheduler. Default: 20",
    )

    parser.add_argument(
        "--lr_scheduler_gamma",
        type=float,
        help="Gamma for lr scheduler. Default: 0.7",
    )

    parser.add_argument(
        "--num_workers", type=int, help="Number of workers for dataloader. Default: 2"
    )
    parser.add_argument(
        "--evaluate_every",
        type=int,
        help="Evaluate model every n steps. Default: 25",
    )
    parser.add_argument(
        "--save_predicted_vol_as_nifti",
        type=parse_bool,
        help="Save predicted volume as nifti. Default True",
    )

    # Entropy Min Loss function params
    parser.add_argument(
        "--class_prior_type",
        type=str,
        choices=["uniform", "data"],
        help="Type of class prior to use for entropy min loss. Default: data",
    )
    parser.add_argument(
        "--use_kl_loss",
        type=parse_bool,
        help="Use KL loss in entropy min loss. Default: True",
    )
    parser.add_argument(
        "--weighted_loss",
        type=parse_bool,
        help="Use weighted loss in entropy min loss. Default: True",
    )
    parser.add_argument(
        "--clases_to_exclude_ent_term",
        type=int,
        nargs="+",
        help="Classes to exclude from entropy term. Default: []",
    )
    parser.add_argument(
        "--classes_to_exclude_kl_term",
        type=int,
        nargs="+",
        help="Classes to exclude from KL term. Default: [0]",
    )
    parser.add_argument(
        "--filter_low_support_classes",
        type=parse_bool,
        help="Set probability of low support classes to 0 in KL term. Default: True",
    )

    # Dataset and its transformations to use for TTA
    # :================================================================================================:
    parser.add_argument("--dataset", type=str, help="Name of dataset to use for tta")
    parser.add_argument(
        "--split", type=str, help="Name of split to use for tta. Default: test"
    )
    parser.add_argument("--n_classes", type=int, help="Number of classes in dataset")
    parser.add_argument(
        "--rescale_factor",
        type=float,
        help="Rescale factor for images in dataset. Default: None",
    )
    parser.add_argument(
        "--load_dataset_in_memory",
        type=parse_bool,
        help="Whether to load the entire dataset in memory. Default: False",
    )

    args = parser.parse_args()

    return args


def get_configuration_arguments() -> tuple[dict, dict]:
    args = preprocess_cmd_args()

    dataset_config = load_config(args.dataset_config_file)
    dataset_config = rewrite_config_arguments(dataset_config, args, "dataset")

    tta_config = load_config(args.tta_config_file)
    tta_config = rewrite_config_arguments(tta_config, args, "tta")

    tta_config[TTA_MODE] = rewrite_config_arguments(
        tta_config[TTA_MODE], args, f"tta, {TTA_MODE}"
    )

    return dataset_config, tta_config


if __name__ == "__main__":

    print(f"Running {__file__}")

    # Loading general parameters
    # :=========================================================================:
    dataset_config, tta_config = get_configuration_arguments()

    seg_dir = tta_config["seg_dir"]
    classes_of_interest = default(tta_config["classes_of_interest"], tuple())
    classes_of_interest: tuple[int | str, ...] = tuple(classes_of_interest)

    params_seg = load_config(os.path.join(seg_dir, "params.yaml"))
    train_params_seg = params_seg["training"]
    train_mode = train_params_seg["train_mode"]

    model_params_seg = params_seg["model"]

    params = {
        "datset": dataset_config,
        "model": model_params_seg,
        "training": train_params_seg,
        "tta": tta_config,
    }

    seed = tta_config["seed"]
    device = tta_config["device"]
    wandb_log = tta_config["wandb_log"]
    wandb_run_name = tta_config["wandb_run_name"]
    start_new_exp = tta_config["start_new_exp"]
    logdir = tta_config[TTA_MODE]["logdir"]
    wandb_project = tta_config[TTA_MODE]["wandb_project"]

    seed_everything(seed)
    device = define_device(device)

    os.makedirs(logdir, exist_ok=True)
    dump_config(os.path.join(logdir, "params.yaml"), params)

    if tta_config["print_config"]:
        print_config(params, keys=["datset", "model", "tta"])

    # Setup wandb logging
    # :=========================================================================:
    if wandb_log:
        wandb_dir = setup_wandb(params, logdir, wandb_project, run_name=wandb_run_name)
    else:
        wandb_dir = None

    # Define the dataset that is to be used for training
    # :=========================================================================:
    print("Defining dataset and its datloader that will be used for TTA")

    dataset_name = tta_config["dataset"]
    split = tta_config["split"]
    dataset_type = tta_config["dataset_type"]

    n_classes = dataset_config[dataset_name]["n_classes"]
    aug_params = tta_config[TTA_MODE]["augmentation"]

    dataset_kwargs = dict(
        dataset_name=dataset_name,
        paths_preprocessed=dataset_config[dataset_name]["paths_preprocessed"],
        paths_original=dataset_config[dataset_name]["paths_original"],
        resolution_proc=dataset_config[dataset_name]["resolution_proc"],
        n_classes=n_classes,
        dim_proc=dataset_config[dataset_name]["dim"],
        aug_params=aug_params,
        load_original=True,
    )

    if dataset_type == "Normal":
        dataset_kwargs['load_in_memory'] = tta_config["load_dataset_in_memory"]
        dataset_kwargs["node_data_path"] = tta_config["node_data_path"]

        dataset_kwargs["mode"] = tta_config["dataset_mode"]
        dataset_kwargs["load_in_memory"] = tta_config["load_in_memory"]
        dataset_kwargs["orientation"] = tta_config["eval_orientation"]
        assert_in(
            dataset_kwargs["orientation"], "orientation", ["depth", "height", "width"]
        )

    if dataset_type == "DinoFeatures":
        dataset_kwargs["paths_preprocessed_dino"] = dataset_config[dataset_name][
            "paths_preprocessed_dino"
        ]
        dataset_kwargs["hierarchy_level"] = train_params_seg[train_mode][
            "hierarchy_level"
        ]
        dataset_kwargs["dino_model"] = train_params_seg[train_mode]["dino_model"]
        del dataset_kwargs["aug_params"]

    # So far only depth orientation is used
    (test_dataset,) = get_datasets(
        dataset_type=dataset_type, splits=[split], **dataset_kwargs
    )

    print("Datasets loaded")

    # Load models
    # :=========================================================================:
    print("Loading segmentation model")

    cpt_type = "checkpoint_best" if tta_config["load_best_cpt"] else "checkpoint_last"
    cpt_seg_fp = os.path.join(seg_dir, train_params_seg[cpt_type])

    if train_mode == "segmentation":
        seg = define_and_possibly_load_norm_seg(
            n_classes=n_classes,
            model_params_norm=model_params_seg["normalization_2D"],  # type: ignore
            model_params_seg=model_params_seg["segmentation_2D"],  # type: ignore
            cpt_fp=cpt_seg_fp,
            device=device,
        )
    elif train_mode == "segmentation_dino":
        seg = define_and_possibly_load_dino_seg(
            train_dino_cfg=train_params_seg["segmentation_dino"],
            decoder_cfg=model_params_seg["resnet_decoder_dino"],
            n_classes=n_classes,
            cpt_fp=cpt_seg_fp,
            device=device,
            load_dino_fe=True,
        )
        seg.precalculated_fts = False  # We will calculate them on the fly
    else:
        raise ValueError(f"Invalid segmentation model train mode: {train_mode}")

    # Define the TTADAE object that does the test time adapatation
    # :=========================================================================:
    viz_interm_outs = default(tta_config["viz_interm_outs"], tuple())
    entropy_min_loss_kwargs = dict(
        use_kl_loss=tta_config[TTA_MODE]["use_kl_loss"],
        weighted_loss=tta_config[TTA_MODE]["weighted_loss"],
        clases_to_exclude_ent_term=tta_config[TTA_MODE]["clases_to_exclude_ent_term"],
        classes_to_exclude_kl_term=tta_config[TTA_MODE]["classes_to_exclude_kl_term"],
        filter_low_support_classes=tta_config[TTA_MODE]["filter_low_support_classes"],
    )

    entropy_min_tta = TTAEntropyMin(
        seg=seg,
        n_classes=n_classes,
        classes_of_interest=classes_of_interest,
        class_prior_type=tta_config[TTA_MODE]["class_prior_type"],
        fit_at_test_time=tta_config[TTA_MODE]["fit_at_test_time"],
        learning_rate=tta_config[TTA_MODE]["learning_rate"],
        weight_decay=tta_config[TTA_MODE]["weight_decay"],
        lr_decay=tta_config[TTA_MODE]["lr_decay"],
        lr_scheduler_step_size=tta_config[TTA_MODE]["lr_scheduler_step_size"],
        lr_scheduler_gamma=tta_config[TTA_MODE]["lr_scheduler_gamma"],
        entropy_min_loss_kwargs=entropy_min_loss_kwargs,
        viz_interm_outs=viz_interm_outs,
        aug_params=aug_params,
        wandb_log=wandb_log,
        device=device,
        seed=seed,
    )

    # Arguments related to visualization of the results
    # :=========================================================================:
    slice_vols_for_viz = (
        (((24, 72), (0, -1), (0, -1))) if dataset_name.startswith("vu") else None
    )

    # Start the TTA loop
    # :=========================================================================:
    start_idx = 0 if tta_config["start"] is None else tta_config["start"]
    stop_idx = (
        test_dataset.get_num_volumes()
        if tta_config["stop"] is None
        else tta_config["stop"]
    )

    save_predicted_vol_as_nifti = tta_config["save_predicted_vol_as_nifti"]

    # Calculate the class ratios on the source domain if class_prior_type is data
    if tta_config[TTA_MODE]["class_prior_type"] == "data":
        sd_dataset = train_params_seg["dataset"]
        sd_split = "train"
        dataset_kwargs['dataset_name'] = sd_dataset
        dataset_kwargs['paths_preprocessed'] = dataset_config[sd_dataset]["paths_preprocessed"]
        dataset_kwargs['paths_original'] = dataset_config[sd_dataset]["paths_original"]
        dataset_kwargs['resolution_proc'] = dataset_config[sd_dataset]["resolution_proc"]
        dataset_kwargs['dim_proc'] = dataset_config[sd_dataset]["dim"]
        
        (source_dataset, ) = get_datasets(
            dataset_type=dataset_type, splits=[split], **dataset_kwargs
        )
        entropy_min_tta.fit_class_prior(source_dataset)

    print("---------------------TTA---------------------")
    print("start vol_idx:", start_idx)
    print("end vol_idx:", stop_idx)

    dice_scores_fg = {
        "dice_score_fg_classes": [],
        "dice_score_fg_classes_sklearn": [],
    }

    dice_scores_classes_of_interest = {
        cls: copy.deepcopy(dice_scores_fg) for cls in classes_of_interest
    }

    for vol_idx in range(start_idx, stop_idx):

        seed_everything(seed)
        print(f"processing volume {vol_idx}")
        # Get the volume on which to run the adapation
        x, *_ = test_dataset[vol_idx]
        x = ensure_nd(5, x) # type: ignore
        assert isinstance(x, torch.Tensor)

        # Get the preprocessed vol for that has the same position as
        # the original vol (preprocessed vol may have a translation in xy
        x_preprocessed, *_ = test_dataset.get_preprocessed_original_volume(vol_idx)
        preprocessed_pix_size = test_dataset.get_processed_pixel_size_w_orientation()

        x_original, y_original_gt = test_dataset.get_original_volume(vol_idx)
        gt_pix_size = test_dataset.get_original_pixel_size_w_orientation(vol_idx)

        base_file_name = f"{test_dataset.dataset_name}_{split}_vol_{vol_idx:03d}"

        # Run Adaptation
        # :=========================================================================:
        num_steps = tta_config[TTA_MODE]["num_steps"]
        batch_size = tta_config[TTA_MODE]["batch_size"]
        num_workers = tta_config["num_workers"]
    
        entropy_min_tta.tta(
            x=x,
            num_steps=num_steps,
            gradient_acc_steps=tta_config[TTA_MODE]["gradient_acc_steps"],
            evaluate_every=tta_config[TTA_MODE]["evaluate_every"],
            registered_x_preprocessed=x_preprocessed,
            x_original=x_original,
            y_gt=y_original_gt.float(),
            preprocessed_pix_size=preprocessed_pix_size,
            gt_pix_size=gt_pix_size,
            batch_size=batch_size,
            num_workers=num_workers,
            classes_of_interest=classes_of_interest,  # type: ignore
            output_dir=logdir,
            file_name=base_file_name,
            store_visualization=True,
            save_predicted_vol_as_nifti=False,
            slice_vols_for_viz=slice_vols_for_viz,
        )

        # Persist results of adaptation run
        os.makedirs(logdir, exist_ok=True)

        # Store csv with dice scores for all classes
        entropy_min_tta.write_current_dice_scores(
            vol_idx, logdir, dataset_name, iteration_type=""
        )

        for dice_score_name in dice_scores_fg:
            dice_scores_fg[dice_score_name].append(
                entropy_min_tta.get_current_average_test_score(dice_score_name)
            )

        for cls in classes_of_interest:
            for dice_score_name in dice_scores_fg:
                dice_scores_classes_of_interest[cls][dice_score_name].append(
                    entropy_min_tta.get_current_test_score(
                        dice_score_name, cls - 1 if "fg" in dice_score_name else cls  # type: ignore
                    )
                )

        # Store results of the last iteration
        # :=========================================================================:
        print("\nEvaluating last iteration")
        last_iter_dir = os.path.join(logdir, "last_iteration")
        os.makedirs(last_iter_dir, exist_ok=True)

        entropy_min_tta.evaluate(
            x_preprocessed=x_preprocessed,
            x_original=x_original,
            y_gt=y_original_gt.float(),
            preprocessed_pix_size=preprocessed_pix_size,
            gt_pix_size=gt_pix_size,
            batch_size=batch_size,
            num_workers=num_workers,
            classes_of_interest=classes_of_interest,
            output_dir=logdir,
            file_name=base_file_name,
            store_visualization=True,
            save_predicted_vol_as_nifti=save_predicted_vol_as_nifti,
            slice_vols_for_viz=slice_vols_for_viz,
        )

        # Store csv with dice scores for all classes
        entropy_min_tta.write_current_dice_scores(
            num_steps, last_iter_dir, dataset_name, iteration_type="last_iteration"
        )

        print("--------------------------------------------")

    write_summary(logdir, dice_scores_fg, dice_scores_classes_of_interest)

    if wandb_log:
        wandb.finish()
