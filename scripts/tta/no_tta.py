import os
import sys
import copy
import argparse

import wandb
import numpy as np

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

from tta_uia_segmentation.src.tta import NoTTA
from tta_uia_segmentation.src.dataset.dataset_in_memory import get_datasets
from tta_uia_segmentation.src.models.io import (
    load_norm_and_seg_from_configs_and_cpt
)
from tta_uia_segmentation.src.utils.io import (
    load_config, dump_config, print_config, rewrite_config_arguments)
from tta_uia_segmentation.src.utils.utils import seed_everything, define_device, parse_bool
from tta_uia_segmentation.src.utils.logging import setup_wandb


def write_summary(logdir, dice_scores_fg, dice_scores_classes_of_interest):
    """
    Write summary of dice scores to a file and print to console.

    Parameters
    ----------
    logdir : str
        Directory to save the summary file.
    dice_scores_fg : dict
        Dictionary of foreground dice scores.
    dice_scores_classes_of_interest : dict
        Dictionary of dice scores for classes of interest.

    Returns
    -------
    None
    """
    summary_file = os.path.join(logdir, 'summary.txt')
    with open(summary_file, 'w') as f:
        def write_and_print(text):
            print(text)
            f.write(text + '\n')

        def write_scores(scores, prefix=''):
            for name, values in scores.items():
                mean, std = np.mean(values), np.std(values)
                write_and_print(f'{prefix}{name} : {mean:.3f} +- {std:.5f}')

        write_and_print('Dataset level metrics\n')
        write_scores(dice_scores_fg)

        write_and_print('\nClass level metrics\n')
        for cls, scores in dice_scores_classes_of_interest.items():
            write_and_print(f'Class {cls}')
            write_scores(scores, prefix='  ')

        write_and_print('\nClass average')
        avg_scores = {name: [np.mean(cls_scores[name]) for cls_scores in dice_scores_classes_of_interest.values()]
                        for name in dice_scores_fg}
        write_scores(avg_scores, prefix='  ')


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
    parser.add_argument('--seg_dir', type=str, help='Path to directory where segmentation checkpoints are saved')
    
    parser.add_argument('--classes_of_interest' , type=int, nargs='+')

    parser.add_argument('--wandb_project', type=str, help='Name of wandb project to log to. Default: "tta"')
    parser.add_argument('--wandb_log', type=parse_bool, help='Log tta to wandb. Default: False.')
    parser.add_argument('--device', type=str, help='Device to use for training. Default: "cuda"')
    
    # TTA loop
    # -------------:
    # Optimization parameters
    parser.add_argument('--batch_size', type=int, help='Batch size for tta. Default: 4')
    parser.add_argument('--num_workers', type=int, help='Number of workers for dataloader. Default: 0')
    
    # Dataset and its transformations to use for TTA
    # ---------------------------------------------------:
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for tta. Default: USZ')
    parser.add_argument('--split', type=str, help='Name of split to use for tta. Default: test')
    parser.add_argument('--n_classes', type=int, help='Number of classes in dataset. Default: 21')
    parser.add_argument('--image_size', type=int, nargs='+', help='Size of images in dataset. Default: [560, 640, 160]')
    parser.add_argument('--resolution_proc', type=float, nargs='+', help='Resolution of images in dataset. Default: [0.3, 0.3, 0.6]')
    parser.add_argument('--rescale_factor', type=float, help='Rescale factor for images in dataset. Default: None')
    
    # Backround suppression for normalized images
    parser.add_argument('--bg_supp_x_norm_eval', type=parse_bool, help='Whether to suppress background for normalized images during evaluation. Default: True')
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
    
    tta_config['no_tta'] = rewrite_config_arguments(
        tta_config['no_tta'], args, 'tta, no_tta')
    
    tta_config['no_tta']['bg_suppression_opts'] = rewrite_config_arguments(
        tta_config['no_tta']['bg_suppression_opts'], args, 'tta, no_tta, bg_suppression_opts',
        prefix_to_remove='bg_supression_')
    
    return dataset_config, tta_config




if __name__ == '__main__':
    
    # Load Hyperparameters
    print(f'Running {__file__}')
    
    # Loading general parameters
    # :=========================================================================:
    dataset_config, tta_config = get_configuration_arguments()
    
    tta_mode                = 'no_tta'
    seg_dir                 = tta_config['seg_dir']
    classes_of_interest     = tta_config['classes_of_interest']
        
    params_seg              = load_config(os.path.join(seg_dir, 'params.yaml'))
    model_params_norm       = params_seg['model']['normalization_2D']
    model_params_seg        = params_seg['model']['segmentation_2D']
    train_params_seg        = params_seg['training']
    
    params                  = { 
                               'datset': dataset_config,
                               'model': {'norm': model_params_norm, 'seg': model_params_seg},
                               'training': {'seg': train_params_seg},
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
  
    test_dataset, = get_datasets(
        dataset_name    = dataset_name,
        paths           = dataset_config[dataset_name]['paths_processed'],
        paths_original  = dataset_config[dataset_name]['paths_original'],
        splits          = [split],
        image_size      = tta_config['image_size'],
        resolution_proc = dataset_config[dataset_name]['resolution_proc'],
        dim_proc        = dataset_config[dataset_name]['dim'],
        n_classes       = n_classes,
        aug_params      = None,
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
    
    if wandb_log:
        wandb.watch([norm], log='all', log_freq=1)

    # Define the TTADAE object that does the test time adapatation
    # :=========================================================================:
    bg_supp_x_norm_eval         = tta_config[tta_mode]['bg_supp_x_norm_eval']
    bg_suppression_opts         = tta_config[tta_mode]['bg_suppression_opts']

    no_tta = NoTTA(
        norm=norm,
        seg=seg,
        n_classes=n_classes,
        classes_of_interest=classes_of_interest,
        wandb_log=wandb_log,
        device=device,
        bg_supp_x_norm_eval=bg_supp_x_norm_eval,
        bg_suppression_opts_eval=bg_suppression_opts,
    )
    
    # Evaluate the dice score per volume 
    # :=========================================================================:
    batch_size                  = tta_config[tta_mode]['batch_size']
    num_workers                 = tta_config['num_workers']

    # Arguments related to visualization of the results
    # :=========================================================================:    
    slice_vols_for_viz = (((24, 72), (0, -1), (0, -1))) if dataset_name.startswith('vu') \
        else None
    
    # Start the TTA loop
    # :=========================================================================:
    start_idx = 0 if tta_config['start'] is None else tta_config['start']
    stop_idx = len(test_dataset.get_volume_indices()) if tta_config['stop'] is None \
        else tta_config['stop']
    
    print('---------------------TTA---------------------')
    print('start vol_idx:', start_idx)
    print('end vol_idx:', stop_idx)
    
    dice_scores_fg = {
        'dice_score_fg_classes': [],
        'dice_score_fg_classes_sklearn': [],
    }

    dice_scores_classes_of_interest = {
        cls: copy.deepcopy(dice_scores_fg) for cls in classes_of_interest
    }

    for i in range(start_idx, stop_idx):

        seed_everything(seed)
        print(f'processing volume {i}')

        no_tta.tta(
            dataset = test_dataset,
            vol_idx = i,
            batch_size = batch_size,
            num_workers=num_workers,
            logdir = logdir,
            slice_vols_for_viz=slice_vols_for_viz,
        )
        
        # Get evaluation for last iteration with prediction in as Nifti volumes
        os.makedirs(logdir, exist_ok=True)

        # Store csv with dice scores for all classes 
        no_tta.write_current_dice_scores(i, logdir, dataset_name, iteration_type='')

        for dice_score_name in dice_scores_fg:
            dice_scores_fg[dice_score_name].append(
                no_tta.get_current_average_test_score(dice_score_name)
            )

        for cls in classes_of_interest:
            for dice_score_name in dice_scores_fg:
                dice_scores_classes_of_interest[cls][dice_score_name].append(
                    no_tta.get_current_test_score(
                        dice_score_name,
                        cls - 1 if 'fg' in dice_score_name else cls
                    )
                )

        print('--------------------------------------------')

    write_summary(logdir, dice_scores_fg, dice_scores_classes_of_interest)

    if wandb_log:
        wandb.finish()
