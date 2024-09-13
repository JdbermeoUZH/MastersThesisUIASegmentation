import os
import sys
import argparse

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

from tta_uia_segmentation.src.tta import TTADAE
from tta_uia_segmentation.src.dataset.dataset_in_memory import get_datasets
from tta_uia_segmentation.src.models.io import (
    load_norm_and_seg_from_configs_and_cpt,
    load_dae_and_atlas_from_configs_and_cpt,
    load_domain_statistiscs
)
from tta_uia_segmentation.src.utils.io import (
    load_config, dump_config, print_config, write_to_csv,
    rewrite_config_arguments)
from tta_uia_segmentation.src.utils.utils import seed_everything, define_device, parse_bool
from tta_uia_segmentation.src.utils.logging import setup_wandb
from tta_uia_segmentation.src.utils.loss import DiceLoss, dice_score


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
    parser.add_argument('--dae_dir', type=str, help='Path to directory where DAE checkpoints are saved')
    parser.add_argument('--seg_dir', type=str, help='Path to directory where segmentation checkpoints are saved')
    parser.add_argument('--wandb_project', type=str, help='Name of wandb project to log to. Default: "tta"')
    parser.add_argument('--wandb_log', type=parse_bool, help='Log tta to wandb. Default: False.')
    parser.add_argument('--start_new_exp', type=parse_bool, help='Start a new wandb experiment. Default: False')
    parser.add_argument('--device', type=str, help='Device to use for training. Default: "cuda"')
    parser.add_argument('--debug_mode', type=parse_bool, help='Whether to run in debug mode. Default: False')
    # TTA loop
    # -------------:
    # Optimization parameters
    parser.add_argument('--num_steps', type=int, help='Number of steps to take in TTA loop. Default: 100')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer. Default: 1e-4')
    parser.add_argument('--accumulate_over_volume', type=parse_bool, help='Whether to accumulate over volume. Default: True')
    parser.add_argument('--batch_size', type=int, help='Batch size for tta. Default: 4')
    parser.add_argument('--num_workers', type=int, help='Number of workers for dataloader. Default: 0')

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


def write_dice_scores(dae_tta: TTADAE, logdir: str, dataset_name: str, iteration_type: str):
    """
    Write dice scores to CSV files.

    Parameters
    ----------
    dae_tta : TTADAE
        The Test Time Adaptation DAE object.
    logdir : str
        Directory to save the CSV files.
    dataset_name : str
        Name of the dataset.
    iteration_type : str
        Type of iteration (e.g., 'last_iteration', 'best_scoring_iteration').

    """
    test_scores_fg = dae_tta.get_all_test_scores_as_df(name_contains='dice_score_fg_classes')
    
    for score_name, score_df in test_scores_fg.items():
        score_name = score_name.replace('/', '__')
        file_name = f'{score_name}_{dataset_name}_{iteration_type}.csv'
        last_scores = score_df.iloc[-1].values
        write_to_csv(
            os.path.join(logdir, file_name),
            np.hstack([[[f'volume_{i:02d}']], [last_scores]]),
            mode='a',
        )


if __name__ == '__main__':
    
    # Load Hyperparameters
    print(f'Running {__file__}')
    
    # Loading general parameters
    # :=========================================================================:
    dataset_config, tta_config = get_configuration_arguments()
    
    tta_mode                = 'dae'
    dae_dir                 = tta_config[tta_mode]['dae_dir']
    seg_dir                 = tta_config['seg_dir']
    
    params_dae              = load_config(os.path.join(dae_dir, 'params.yaml'))
    model_params_dae        = params_dae['model']['dae']
    train_params_dae        = params_dae['training']
    
    params_seg              = load_config(os.path.join(seg_dir, 'params.yaml'))
    model_params_norm       = params_seg['model']['normalization_2D']
    model_params_seg        = params_seg['model']['segmentation_2D']
    train_params_seg        = params_seg['training']
    
    params                  = { 
                               'datset': dataset_config,
                               'model': {'norm': model_params_norm, 'seg': model_params_seg, 'dae': model_params_dae},
                               'training': {'seg': train_params_seg, 'dae': train_params_dae},
                               'tta': tta_config
                               }
        
    seed                    = tta_config['seed']
    device                  = tta_config['device']
    wandb_log               = tta_config['wandb_log']
    start_new_exp           = tta_config['start_new_exp']
    logdir                  = tta_config[tta_mode]['logdir']
    wandb_project           = tta_config[tta_mode]['wandb_project']  
    
    os.makedirs(logdir, exist_ok=True)
    dump_config(os.path.join(logdir, 'params.yaml'), params)
    print_config(params, keys=['datset', 'model', 'tta'])

    # Setup wandb logging
    # :=========================================================================:
    if wandb_log:
        wandb_dir = setup_wandb(params, logdir, wandb_project, start_new_exp)
    
    # Define the dataset that is to be used for training
    # :=========================================================================:
    print('Defining dataset and its datloader that will be used for TTA')
    seed_everything(seed)
    device                 = define_device(device)
    dataset_name           = tta_config['dataset']
    split                  = tta_config['split']
    n_classes              = dataset_config[dataset_name]['n_classes']
    bg_suppression_opts    = tta_config['bg_suppression_opts']
    aug_params             = tta_config[tta_mode]['augmentation']
  
    test_dataset, = get_datasets(
        dataset_name    = dataset_name,
        paths           = dataset_config[dataset_name]['paths_processed'],
        paths_original  = dataset_config[dataset_name]['paths_original'],
        splits          = [split],
        image_size      = tta_config['image_size'],
        resolution_proc = dataset_config[dataset_name]['resolution_proc'],
        dim_proc        = dataset_config[dataset_name]['dim'],
        n_classes       = n_classes,
        aug_params      = aug_params,
        deformation     = None,
        load_original   = True,
        bg_suppression_opts=bg_suppression_opts,
    )
        
    print('Datasets loaded')

    # Load models
    # :=========================================================================:
    print('Loading segmentation model')
    cpt_type = 'checkpoint_best' if tta_config['load_best_cpt'] \
        else 'checkpoint_last'
    cpt_seg_fp = os.path.join(seg_dir, train_params_seg[cpt_type])
    
    norm, seg = load_norm_and_seg_from_configs_and_cpt(
        n_classes = n_classes,
        model_params_norm = model_params_norm,
        model_params_seg = model_params_seg,
        cpt_fp = cpt_seg_fp,
        device = device,
    )
    
    min_max_clip_q = tta_config[tta_mode]['min_max_quantile']   
    norm_sd_statistics = load_domain_statistiscs(
        cpt_fp = cpt_seg_fp,
        frozen = True,
        momentum = 0.96,
        min_max_clip_q=min_max_clip_q,
    )

    if wandb_log:
        wandb.watch([norm], log='all', log_freq=1)

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

    learning_rate               = tta_config[tta_mode]['learning_rate']
    alpha                       = tta_config[tta_mode]['alpha']
    beta                        = tta_config[tta_mode]['beta']
    smooth                      = tta_config[tta_mode]['smooth']
    epsilon                     = tta_config[tta_mode]['epsilon']
    update_norm_td_statistics   = tta_config[tta_mode]['update_norm_td_statistics']
    bg_supp_x_norm_eval         = tta_config[tta_mode]['bg_supp_x_norm_eval']
    bg_supp_x_norm_tta_dae      = tta_config[tta_mode]['bg_supp_x_norm_tta_dae']
    bg_suppression_opts         = tta_config[tta_mode]['bg_suppression_opts']
    normalization_strategy      = tta_config[tta_mode]['normalization_strategy']
    manually_norm_img_before_seg_tta = tta_config[tta_mode]['manually_norm_img_before_seg_tta']
    manually_norm_img_before_seg_eval = tta_config[tta_mode]['manually_norm_img_before_seg_eval']

    test_eval_metrics           = { 
        'dice_score_all_classes': lambda y_pred, y_gt: dice_score(
            y_pred, y_gt, soft=False, reduction='none', smooth=1e-5), 
        'dice_score_fg_classes': lambda y_pred, y_gt: dice_score(
            y_pred, y_gt, soft=False, reduction='none', foreground_only=True, smooth=1e-5)
        }
    
    dice_loss = DiceLoss(smooth=smooth, epsilon=epsilon, debug_mode=debug_mode)
    
    dae_tta = TTADAE(
        norm=norm,
        seg=seg,
        dae=dae,
        atlas=atlas,
        norm_sd_statistics=norm_sd_statistics,
        n_classes=n_classes,
        rescale_factor=rescale_factor,
        loss_func=dice_loss,
        learning_rate=learning_rate,
        alpha=alpha,
        beta=beta,
        wandb_log=wandb_log,
        debug_mode=debug_mode,
        device=device,
        update_norm_td_statistics=update_norm_td_statistics,
        bg_supp_x_norm_eval=bg_supp_x_norm_eval,
        bg_suppression_opts_eval=bg_suppression_opts,
        bg_supp_x_norm_tta_dae=bg_supp_x_norm_tta_dae,
        bg_suppression_opts_tta_dae=bg_suppression_opts,
        normalization_strategy=normalization_strategy,
        manually_norm_img_before_seg_tta=manually_norm_img_before_seg_tta,
        manually_norm_img_before_seg_eval=manually_norm_img_before_seg_eval,
    )
    
    # Do TTA with a DAE
    # :=========================================================================:
    num_steps                   = tta_config[tta_mode]['num_steps']
    batch_size                  = tta_config[tta_mode]['batch_size']
    num_workers                 = tta_config['num_workers']
    save_checkpoints            = tta_config[tta_mode]['save_checkpoints']
    update_dae_output_every     = tta_config[tta_mode]['update_dae_output_every']
    const_aug_per_volume        = tta_config[tta_mode]['const_aug_per_volume']
    accumulate_over_volume      = tta_config[tta_mode]['accumulate_over_volume']
    calculate_dice_every        = tta_config[tta_mode]['calculate_dice_every']
    
    slice_vols_for_viz = (((10, 58), (0, -1), (0, -1))) if dataset_name.startswith('vu') \
        else None

    start_idx = 0
    stop_idx = len(test_dataset.get_volume_indices())  # == number of volumes
    if tta_config['start'] is not None:
        start_idx = tta_config['start']
    if tta_config['stop'] is not None:
        stop_idx = tta_config['stop']
    
    print('---------------------TTA---------------------')
    print('start vol_idx:', start_idx)
    print('end vol_idx:', stop_idx)
        
    for i in range(start_idx, stop_idx):

        seed_everything(seed)
        print(f'processing volume {i}')

        dae_tta.tta(
            dataset = test_dataset,
            vol_idx = i,
            num_steps = num_steps,
            batch_size = batch_size,
            num_workers=num_workers,
            calculate_dice_every = calculate_dice_every,
            update_dae_output_every = update_dae_output_every,
            accumulate_over_volume = accumulate_over_volume,
            const_aug_per_volume = const_aug_per_volume,
            save_checkpoints = save_checkpoints,
            logdir = logdir,
            slice_vols_for_viz=slice_vols_for_viz,
        )
        
        # Get evaluation for last iteration with prediction in as Nifti volumes
        print('\nEvaluating last iteration')
        last_iter_dir = os.path.join(logdir, 'last_iteration')
        os.makedirs(last_iter_dir, exist_ok=True)

        dae_tta.evaluate(
            dataset=test_dataset,
            vol_idx=i,
            iteration=num_steps,
            output_dir=last_iter_dir,
            store_visualization=True,
            save_predicted_vol_as_nifti=True,
            batch_size=batch_size,
            num_workers=num_workers,
            slice_vols_for_viz=slice_vols_for_viz,
        )
        
        # Store csv with dice scores for all classes 
        write_dice_scores(dae_tta, last_iter_dir, dataset_name, iteration_type='last_iteration')

        # Get evaluation for best scoring iteration with prediction in as Nifti volumes
        print('\nEvaluating best scoring iteration')
        best_score_iter_dir = os.path.join(logdir, 'best_scoring_iteration')
        os.makedirs(best_score_iter_dir, exist_ok=True)

        dae_tta.load_best_state_norm()

        dae_tta.evaluate(
            dataset=test_dataset,
            vol_idx=i,
            iteration=dae_tta.get_current_iteration(),
            output_dir=best_score_iter_dir,
            store_visualization=True,
            save_predicted_vol_as_nifti=True,
            batch_size=batch_size,
            num_workers=num_workers,
            slice_vols_for_viz=slice_vols_for_viz,
        )

        # Store csv with dice scores for all classes at the best scoring iteration
        write_dice_scores(dae_tta, best_score_iter_dir, dataset_name, iteration_type='best_scoring_iteration')

        # Write the score of to a file
        write_to_csv(
            os.path.join(best_score_iter_dir, f'dice_scores_all_classes_wrt_pl_{dataset_name}.csv'),
            np.hstack([[[f'volume_{i:02d}']], 
                       [[dae_tta.get_current_iteration()]], 
                       [[dae_tta.get_model_selection_score()]], 
                       ]),
            mode='a',
        )   

        dae_tta.reset_state()
        print('--------------------------------------------')
            
    if wandb_log:
        wandb.finish()
