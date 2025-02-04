import os
import sys
import copy
import argparse

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

from tta_uia_segmentation.src.tta import TTADAE
from tta_uia_segmentation.src.dataset.io import get_datasets
from tta_uia_segmentation.src.dataset.utils import ensure_nd
from tta_uia_segmentation.src.models.io import (
    define_and_possibly_load_norm_seg,
    define_and_possibly_load_dino_seg,
    load_dae_and_atlas_from_configs_and_cpt
)
from tta_uia_segmentation.src.utils.io import (
    load_config,
    dump_config,
    print_config, 
    rewrite_config_arguments)
from tta_uia_segmentation.src.utils.utils import (
    seed_everything,
    parse_bool,
    assert_in,
    default
)
from tta_uia_segmentation.src.utils.logging import setup_wandb
from tta_uia_segmentation.src.utils.loss import DiceLoss


TTA_MODE = "dae"


def preprocess_cmd_args() -> argparse.Namespace:
    """_
    Parse command line arguments and return them as a Namespace object.
    
    All options specified will overwrite whaterver is specified in the config files.
    
    """
    parser = argparse.ArgumentParser(description="Test Time Adaption with DAE")
    
    parser.add_argument('dataset_config_file', type=str, help='Path to yaml config file with parameters that define the dataset.')
    parser.add_argument('tta_config_file', type=str, help='Path to yaml config file with parameters for test time adaptation')
    
    # TTA parameters. If provided, overrides default parameters from config file.
    # :================================================================================================:
    parser.add_argument('--start', type=int, help='starting volume index to be used for testing')
    parser.add_argument('--stop', type=int, help='stopping volume index to be used for testing (index not included)')
    parser.add_argument('--logdir', type=str, help='Path to directory where logs and checkpoints are saved. Default: logs') 
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="Name of wandb run. Default: None",
    ) 

    parser.add_argument("--continuous_tta", type=parse_bool, help="Whether to run continuous TTA. Default: False")
    parser.add_argument('--dae_dir', type=str, help='Path to directory where DAE checkpoints are saved')
    parser.add_argument('--seg_dir', type=str, help='Path to directory where segmentation checkpoints are saved')
    parser.add_argument('--wandb_project', type=str, help='Name of wandb project to log to. Default: "tta"')
    parser.add_argument('--wandb_log', type=parse_bool, help='Log tta to wandb. Default: False.')
    parser.add_argument('--start_new_exp', type=parse_bool, help='Start a new wandb experiment. Default: False')
    parser.add_argument('--device', type=str, help='Device to use for training. Default: "cuda"')
    parser.add_argument('--debug_mode', type=parse_bool, help='Whether to run in debug mode. Default: False')
    
    parser.add_argument('--save_checkpoints', type=parse_bool, help='Whether to save checkpoints. Default: True')

    # TTA loop
    # -------------:
    # Optimization parameters
    parser.add_argument('--num_steps', type=int, help='Number of steps to take in TTA loop. Default: 100')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer. Default: 1e-4')
    parser.add_argument('--accumulate_over_volume', type=parse_bool, help='Whether to accumulate over volume. Default: True')
    parser.add_argument('--batch_size', type=int, help='Batch size for tta. Default: 4')
    parser.add_argument('--num_workers', type=int, help='Number of workers for dataloader. Default: 0')
    parser.add_argument('--max_grad_norm', type=float,  nargs="?", const=None, help='Maximum gradient norm. Default: 1.0')

    # Loss function parameters
    parser.add_argument('--smooth', type=float, help='Smooth parameter for dice loss. Added to both numerator and denominator. Default: 0.')
    parser.add_argument('--epsilon', type=float, help='Epsilon parameter for dice loss (avoid division by zero). Default: 1e-5')

    # DAE and Atlas parameters
    parser.add_argument('--alpha', type=float, help='Proportion of how much better the dice of the DAE pseudolabel and predicted segmentation'
                                                    'should be than the dice of the Atlas pseudolabel. Default: 1')
    parser.add_argument('--beta', type=float, help='Minimum dice of the Atlas pseudolabel and the predicted segmentation. Default: 0.25')
    parser.add_argument('--calculate_dice_every', type=int, help='Calculate dice every n steps. Default: 25')
    parser.add_argument('--update_dae_output_every', type=int, help='Update DAE output every n steps. Default: 25')
    
    # Dataset and its transformations to use for TTA
    # ---------------------------------------------------:
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for tta. Default: USZ')
    parser.add_argument('--split', type=str, help='Name of split to use for tta. Default: test')
    parser.add_argument('--n_classes', type=int, help='Number of classes in dataset. Default: 21')
    parser.add_argument('--image_size', type=int, nargs='+', help='Size of images in dataset. Default: [560, 640, 160]')
    parser.add_argument('--resolution_proc', type=float, nargs='+', help='Resolution of images in dataset. Default: [0.3, 0.3, 0.6]')
    parser.add_argument('--rescale_factor', type=float, help='Rescale factor for images in dataset. Default: None')
    parser.add_argument('--classes_of_interest' , type=int, nargs='+')

    # Augmentations
    parser.add_argument('--aug_da_ratio', type=float, help='Ratio of images to apply DA to. Default: 0.25')
    parser.add_argument('--aug_sigma', type=float, help='augmentation. Default: 20') #TODO: specify what this is
    parser.add_argument('--aug_alpha', type=float, help='augmentation. Default: 1000') #TODO: specify what this is
    parser.add_argument('--aug_trans_min', type=float, help='Minimum value for translation augmentation. Default: -10')
    parser.add_argument('--aug_trans_max', type=float, help='Maximum value for translation augmentation. Default: 10')
    parser.add_argument('--aug_rot_min', type=float, help='Minimum value for rotation augmentation. Default: -10')
    parser.add_argument('--aug_rot_max', type=float, help='Maximum value for rotation augmentation. Default: 10') 
    parser.add_argument('--aug_scale_min', type=float, help='Minimum value for zooming augmentation. Default: 0.9') 
    parser.add_argument('--aug_scale_max', type=float, help='Maximum value for zooming augmentation. Default: 1.1')
    parser.add_argument('--aug_brightness_min', type=float, help='Minimum value for brightness augmentation. Default: 0.0') 
    parser.add_argument('--aug_brightness_max', type=float, help='Maximum value for brightness augmentation. Default: 0.0')   
    parser.add_argument('--aug_noise_mean', type=float, help='Mean value for noise augmentation. Default: 0.0')
    parser.add_argument('--aug_noise_std', type=float, help='Standard deviation value for noise augmentation. Default: 0.0') 
    
    # Backround suppression for normalized images
    parser.add_argument('--bg_supp_x_norm_eval', type=parse_bool, help='Whether to suppress background for normalized images during evaluation. Default: True')
    parser.add_argument('--bg_supp_x_norm_tta_dae', type=parse_bool, help='Whether to suppress background for normalized images during TTA with DAE. Default: True')
    parser.add_argument('--bg_supression_type', choices=['fixed_value', 'random_value', 'none', None], help='Type of background suppression to use. Default: fixed_value')
    parser.add_argument('--bg_supression_value', type=float, help='Value to use for background suppression. Default: -0.5')
    parser.add_argument('--bg_supression_min', type=float, help='Minimum value to use for random background suppression. Default: -0.5')
    parser.add_argument('--bg_supression_max', type=float, help='Maximum value to use for random background suppression. Default: 1.0')
    parser.add_argument('--bg_supression_max_source', type=str, choices=['thresholding', 'ground_truth'], help='Maximum value to use for random background suppression. Default: "thresholding"')
    parser.add_argument('--bg_supression_thresholding', type=str, choices=['otsu', 'yen', 'li', 'minimum', 'mean', 'triangle', 'isodata'], help='Maximum value to use for random background suppression. Default: "otsu"') 
    parser.add_argument('--bg_supression_hole_filling', type=parse_bool, help='Whether to use hole filling for background suppression. Default: True')
    args = parser.parse_args()
    
    return args


def get_configuration_arguments() -> tuple[dict, dict]:
    args = preprocess_cmd_args()
    
    dataset_config = load_config(args.dataset_config_file)
    dataset_config = rewrite_config_arguments(dataset_config, args, 'dataset')
    
    tta_config = load_config(args.tta_config_file)
    tta_config = rewrite_config_arguments(tta_config, args, 'tta')
    
    tta_config['dae'] = rewrite_config_arguments(
        tta_config['dae'], args, 'tta, dae')
    
    tta_config['dae']['augmentation'] = rewrite_config_arguments(
        tta_config['dae']['augmentation'], args, 'tta, dae, augmentation',
        prefix_to_remove='aug_')
    
    tta_config['dae']['bg_suppression_opts'] = rewrite_config_arguments(
        tta_config['dae']['bg_suppression_opts'], args, 'tta, dae, bg_suppression_opts',
        prefix_to_remove='bg_supression_')
    
    return dataset_config, tta_config


if __name__ == '__main__':
    
    # Load Hyperparameters
    print(f'Running {__file__}')
    
    # Loading general parameters
    # :=========================================================================:
    dataset_config, tta_config = get_configuration_arguments()

    continuous_tta          = tta_config['continuous_tta']
    
    seg_dir                 = tta_config['seg_dir']
    dae_dir                 = tta_config[TTA_MODE]['dae_dir']

    params_dae              = load_config(os.path.join(dae_dir, 'params.yaml'))
    model_params_dae        = params_dae['model']['dae']
    train_params_dae        = params_dae['training']
    
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


    seed                    = tta_config['seed']
    device                  = tta_config['device']
    wandb_log               = tta_config['wandb_log']
    wandb_run_name = tta_config["wandb_run_name"]
    start_new_exp           = tta_config['start_new_exp']
    logdir                  = tta_config[TTA_MODE]['logdir']
    wandb_project           = tta_config[TTA_MODE]['wandb_project']  
  
    classes_of_interest = default(tta_config["classes_of_interest"], tuple())
    classes_of_interest: tuple[int | str, ...] = tuple(classes_of_interest)
           
    os.makedirs(logdir, exist_ok=True)
    dump_config(os.path.join(logdir, 'params.yaml'), params)
    print_config(params, keys=['datset', 'model', 'tta'])

    # Setup wandb logging
    # :=========================================================================:
    if wandb_log:
        wandb_dir = setup_wandb(
            params=params,
            logdir=logdir,
            wandb_project=wandb_project,
            start_new_exp=start_new_exp,
            run_name=wandb_run_name)    
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
        with_norm = (
            train_params_seg["segmentation_dino"]["with_norm_module"]
            if "with_norm_module" in train_params_seg["segmentation_dino"]
            else False
        )
        if with_norm:
            norm_cfg = model_params_seg["normalization_2D"]
        else:
            norm_cfg = None
        
        train_params_seg["segmentation_dino"]["precalculated_fts"] = False
        
        seg = define_and_possibly_load_dino_seg(
            train_dino_cfg=train_params_seg["segmentation_dino"],
            decoder_cfg=model_params_seg["resnet_decoder_dino"],
            norm_cfg=norm_cfg,
            n_classes=n_classes,
            cpt_fp=cpt_seg_fp,
            device=device,
            load_dino_fe=True,
        )
    else:
        raise ValueError(f"Invalid segmentation model train mode: {train_mode}")

    if wandb_log:
        wandb.watch([seg], 
            log='all', 
            log_freq=1,
            criterion={
                "gradients": {
                    "norm_too_high": lambda x: torch.norm(x) > 10.0,
                    "contains_nan": lambda x: torch.isnan(x).any(),
                    "contains_inf": lambda x: torch.isinf(x).any(),
                    "too_small": lambda x: torch.abs(x).mean() < 1e-7
                },
                "parameters": {
                    "diverging": lambda x: torch.abs(x).max() > 100,
                    "dead": lambda x: torch.abs(x).mean() < 1e-10
                }
            }
        )

    print('Loading DAE model and Atlas')

    # DAE
    dae, atlas = load_dae_and_atlas_from_configs_and_cpt(
        n_classes = n_classes,
        model_params_dae = model_params_dae,
        cpt_fp = os.path.join(dae_dir, train_params_dae[cpt_type]),
        device = device,
    )
    
    # Define the TTADAE object that does the test time adapatation
    # :=========================================================================:
    rescale_factor              = train_params_dae['dae']['rescale_factor']

    debug_mode                  = tta_config['debug_mode']

    learning_rate               = tta_config[TTA_MODE]['learning_rate']
    max_grad_norm               = tta_config[TTA_MODE]['max_grad_norm']
    alpha                       = tta_config[TTA_MODE]['alpha']
    beta                        = tta_config[TTA_MODE]['beta']
    smooth                      = tta_config[TTA_MODE]['smooth']
    epsilon                     = tta_config[TTA_MODE]['epsilon']
    fit_at_test_time            = tta_config[TTA_MODE]['fit_at_test_time']
    bg_supp_x_norm_tta_dae      = tta_config[TTA_MODE]['bg_supp_x_norm_tta_dae']

    use_only_dae_pl             = tta_config[TTA_MODE]['use_only_dae_pl']
    use_only_atlas           = tta_config[TTA_MODE]['use_only_atlas']

    dice_loss = DiceLoss(smooth=smooth, epsilon=epsilon, debug_mode=debug_mode)
    
    dae_tta = TTADAE(
        seg=seg,
        dae=dae,
        n_classes=n_classes,
        atlas=atlas,
        rescale_factor=rescale_factor,
        fit_at_test_time=fit_at_test_time,
        aug_params=aug_params,
        loss_func=dice_loss,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        alpha=alpha,
        beta=beta,
        use_only_dae_pl=use_only_dae_pl,
        use_only_atlas=use_only_atlas,
        classes_of_interest=classes_of_interest,
        seed=seed,
        wandb_log=wandb_log,
        device=device,
        bg_supp_x_norm_tta_dae=bg_supp_x_norm_tta_dae,
        debug_mode=debug_mode,
    )
    
    # Do TTA with a DAE
    # :=========================================================================:
    num_steps                   = tta_config[TTA_MODE]['num_steps']
    batch_size                  = tta_config[TTA_MODE]['batch_size']
    num_workers                 = tta_config['num_workers']
    save_checkpoints            = tta_config[TTA_MODE]['save_checkpoints']
    update_dae_output_every     = tta_config[TTA_MODE]['update_dae_output_every']
    const_aug_per_volume        = tta_config[TTA_MODE]['const_aug_per_volume']
    accumulate_over_volume      = tta_config[TTA_MODE]['accumulate_over_volume']
    calculate_dice_every        = tta_config[TTA_MODE]['calculate_dice_every']
    
    # Dictionaries to store dice scores
    # : ========================================================================:
    dice_scores_fg = {
        "dice_score_fg_classes": [],
        "dice_score_fg_classes_sklearn": [],
    }

    dice_scores_classes_of_interest = {
        cls: copy.deepcopy(dice_scores_fg) for cls in classes_of_interest
    } 

    # Arguments related to visualization of the results
    # :=========================================================================:
    slice_vols_for_viz = (((10, 58), (0, -1), (0, -1))) if dataset_name.startswith('vu') \
        else None

    # Start the TTA loop per volume
    # :=========================================================================:
    start_idx = 0 if tta_config["start"] is None else tta_config["start"]
    stop_idx = (
        test_dataset.get_num_volumes()
        if tta_config["stop"] is None
        else tta_config["stop"]
    )

    save_predicted_vol_as_nifti = tta_config["save_predicted_vol_as_nifti"]
    
    print('---------------------TTA---------------------')
    print('start vol_idx:', start_idx)
    print('end vol_idx:', stop_idx)
    print("Running TTA in mode: " + "continuous" if continuous_tta else "episodical")
    print('--------------------------------------------')

    for vol_idx in range(start_idx, stop_idx):

        if debug_mode:
            print('DEBUG: Check this is always the same for different volumes ')
            print(dae_tta.tta_fitted_params[-1])

        seed_everything(seed)
        print(f'processing volume {vol_idx}')

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
        dae_tta.tta(
            x=x,
            registered_x_preprocessed=x_preprocessed,
            x_original=x_original,
            y_gt=y_original_gt.float(),
            preprocessed_pix_size=preprocessed_pix_size,
            gt_pix_size=gt_pix_size,
            num_steps = num_steps,
            batch_size = batch_size,
            num_workers=num_workers,
            calculate_dice_every = calculate_dice_every,
            update_dae_output_every = update_dae_output_every,
            accumulate_over_volume = accumulate_over_volume,
            const_aug_per_volume = const_aug_per_volume,
            save_checkpoints = save_checkpoints,
            output_dir = logdir,
            file_name=base_file_name,
            slice_vols_for_viz=slice_vols_for_viz,
            store_visualization=True,
            save_predicted_vol_as_nifti=False,
        )
        
        # Persist results of adaptation run
        os.makedirs(logdir, exist_ok=True)

        # Store csv with dice scores for all classes
        dae_tta.write_current_dice_scores(
            vol_idx, logdir, dataset_name, iteration_type=""
        )

        for dice_score_name in dice_scores_fg:
            dice_scores_fg[dice_score_name].append(
                dae_tta.get_current_average_test_score(dice_score_name)
            )

        for cls in classes_of_interest:
            for dice_score_name in dice_scores_fg:
                dice_scores_classes_of_interest[cls][dice_score_name].append(
                    dae_tta.get_current_test_score(
                        dice_score_name, cls - 1 if "fg" in dice_score_name else cls  # type: ignore
                    )
                )

        # Store results of the last iteration
        # :=========================================================================:
        print("\nEvaluating last iteration")
        last_iter_dir = os.path.join(logdir, "last_iteration")
        os.makedirs(last_iter_dir, exist_ok=True)

        dae_tta.evaluate(
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
        dae_tta.write_current_dice_scores(
            num_steps, last_iter_dir, dataset_name, iteration_type="last_iteration"
        )
        
        # Store results of best scoring iteration (lowest dice btw. prediction and pseudo-label)
        # :=========================================================================:
        
        # Get evaluation for best scoring iteration with prediction in as Nifti volumes
        print('\nEvaluating best scoring iteration')
        best_score_iter_dir = os.path.join(logdir, 'best_scoring_iteration')
        os.makedirs(best_score_iter_dir, exist_ok=True)

        dae_tta.load_best_state()

        dae_tta.evaluate(
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
        dae_tta.write_current_dice_scores(
            num_steps, last_iter_dir, dataset_name, iteration_type="last_iteration"
        )

        # Reset the state to fit to new volume
        breakpoint()
        if not continuous_tta:
            dae_tta.reset_state()
        print('--------------------------------------------')
            
    if wandb_log:
        wandb.finish()
