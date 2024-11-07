import os
import sys
import argparse

import wandb
import torch
from torch.utils.data import Subset
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

from tta_uia_segmentation.src.tta import DiffusionTTA
from tta_uia_segmentation.src.dataset.dataset_in_memory import get_datasets
from tta_uia_segmentation.src.models.io import (
    load_cddpm_from_configs_and_cpt,
    define_and_possibly_load_norm_seg
)
from tta_uia_segmentation.src.utils.io import (
    load_config, dump_config, print_config, write_to_csv, rewrite_config_arguments)
from tta_uia_segmentation.src.utils.utils import seed_everything, define_device
from tta_uia_segmentation.src.utils.logging import setup_wandb


torch.autograd.set_detect_anomaly(True)

tta_mode = 'diffusionTTA'

def preprocess_cmd_args() -> argparse.Namespace:
    """_
    Parse command line arguments and return them as a Namespace object.
    
    All options specified will overwrite whaterver is specified in the config files.
    
    """
    parse_bool = lambda s: s.strip().lower() == 'true'
    parser = argparse.ArgumentParser(description="Test Time Adaption with DAE")
    
    parser.add_argument('dataset_config_file', type=str, help='Path to yaml config file with parameters that define the dataset.')
    parser.add_argument('tta_config_file', type=str, help='Path to yaml config file with parameters for test time adaptation')
    
    # TTA parameters. If provided, overrides default parameters from config file.
    # :================================================================================================:
    parser.add_argument('--start', type=int, help='starting volume index to be used for testing')
    parser.add_argument('--stop', type=int, help='stopping volume index to be used for testing (index not included)')
    parser.add_argument('--logdir', type=str, help='Path to directory where logs and checkpoints are saved. Default: logs')  
    parser.add_argument('--seg_dir', type=str, help='Path to directory where segmentation checkpoints are saved')
    parser.add_argument('--wandb_log', type=parse_bool, help='Log tta to wandb. Default: False.')
    parser.add_argument('--wandb_project', type=str, help='Name of wandb project to log to. Default: "tta"')
    parser.add_argument('--start_new_exp', type=parse_bool, help='Start a new wandb experiment. Default: False')
    
    # TTA loop
    # -------------:
    # optimization params
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer. Default: 8e-5')
    parser.add_argument('--learning_rate_norm', type=float, help='Learning rate for optimizer. Default: None')
    parser.add_argument('--learning_rate_seg', type=float, help='Learning rate for optimizer. Default: None')
    parser.add_argument('--learning_rate_ddpm', type=float, help='Learning rate for optimizer. Default: None')  
    parser.add_argument('--num_steps', type=int, help='Number of steps to take in TTA loop. Default: 100')
    parser.add_argument('--num_t_noise_pairs_per_img', type=int, help='Number of t, noise pairs per image. Default: 180')
    parser.add_argument('--pair_sampling_type', type=str, help='Type of sampling for t, noise pairs. Default: "uniform"', choices=['one_per_volume', 'one_per_image'])
    parser.add_argument('--batch_size', type=int, help='Batch size for tta. Default: 4')
    parser.add_argument('--minibatch_size_ddpm', type=int, help='Minibatch size for DDPM. Default: 2')
    parser.add_argument('--num_workers', type=int, help='Number of workers for dataloader. Default: 0')

    # DDPM params
    parser.add_argument('--ddpm_dir', type=str, help='Path to directory where DDPM checkpoints are saved')
    parser.add_argument('--cpt_fn', type=str, help='Name of checkpoint file to load for DDPM')
    parser.add_argument('--ddpm_loss', type=str, help='Type of DDPM loss. Default: None', choices=['jacobian', 'sds', 'dds', 'pds', None])
    parser.add_argument('--t_ddpm_range', type=float, nargs=2, help='Quantile range of t values for DDPM. Default: [0.2, 0.98]')       
    parser.add_argument('--t_sampling_strategy', type=str, help='Sampling strategy for t values. Default: uniform')
    parser.add_argument('--min_max_intenities_norm_imgs', type=float, nargs=2, help='Min and max intensities for normalization before evaluating DDPM')
   
    # Probably not used arguments
    parser.add_argument('--seed', type=int, help='Seed for random number generators. Default: 0')   
    parser.add_argument('--device', type=str, help='Device to use for tta. Default cuda', )
    parser.add_argument('--save_checkpoints', type=parse_bool,
                        help='Whether to save checkpoints. Default: True')
    
    # Dataset and its transformations to use for TTA
    # ---------------------------------------------------:
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for tta.')
    parser.add_argument('--split', type=str, help='Name of split to use for tta')
    
    parser.add_argument('--n_classes', type=int, help='Number of classes in dataset.')
    parser.add_argument('--classes_of_interest' , type=int, nargs='+')
    parser.add_argument('--image_size', type=int, nargs='+', help='Size of images in dataset.')
    parser.add_argument('--resolution_proc', type=float, nargs='+', help='Resolution of images in dataset.')
    parser.add_argument('--rescale_factor', type=float, help='Rescale factor for images in dataset.')
    
    # Evaluation parameters
    # ----------------------:
    parser.add_argument('--calculate_dice_every', type=int, help='Calculate dice score every n steps. Default: 10')
    args = parser.parse_args()
        
    return args


def get_configuration_arguments() -> tuple[dict, dict]:
    args = preprocess_cmd_args()
    
    dataset_config = load_config(args.dataset_config_file)
    dataset_config = rewrite_config_arguments(dataset_config, args, 'dataset')
    
    tta_config = load_config(args.tta_config_file)
    tta_config = rewrite_config_arguments(tta_config, args, 'tta')
    
    tta_config[tta_mode] = rewrite_config_arguments(
        tta_config[tta_mode], args, 'tta, tta_mode')
    
    tta_config[tta_mode]['augmentation'] = rewrite_config_arguments(
        tta_config[tta_mode]['augmentation'], args, 'tta, tta_mode, augmentation',
        prefix_to_remove='aug_')
    
    
    return dataset_config, tta_config


if __name__ == '__main__':
    
    # Load Hyperparameters
    print(f'Running {__file__}')
    
    # Loading general parameters
    # :=========================================================================:
    dataset_config, tta_config = get_configuration_arguments()

    seg_dir                 = tta_config['seg_dir']
    ddpm_dir                = tta_config[tta_mode]['ddpm_dir']
    
    params_seg              = load_config(os.path.join(seg_dir, 'params.yaml'))
    model_params_norm       = params_seg['model']['normalization_2D']
    model_params_seg        = params_seg['model']['segmentation_2D']
    train_params_seg        = params_seg['training']
    
    params_ddpm             = load_config(os.path.join(ddpm_dir, 'params.yaml'))
    model_params_ddpm       = params_ddpm['model']['ddpm_unet']
    train_params_ddpm       = params_ddpm['training']['ddpm']
    
    params                  = { 
                               'datset': dataset_config,
                               'model': {'norm': model_params_norm, 'seg': model_params_seg, 
                                         'ddpm': model_params_ddpm},
                               'training': {'seg': train_params_seg, 
                                            'ddpm': train_params_ddpm},
                               'tta': tta_config
                               }
        
    seed                    = tta_config['seed']
    device                  = tta_config['device']
    wandb_log               = tta_config['wandb_log']
    start_new_exp           = tta_config['start_new_exp']
    logdir                  = tta_config[tta_mode]['logdir']
    wandb_project           = tta_config[tta_mode]['wandb_project']
    
    seed_everything(seed)
    
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
    device                 = define_device(device)
    dataset                = tta_config['dataset']
    split                  = tta_config['split']
    n_classes              = dataset_config[dataset]['n_classes']
    aug_params             = tta_config[tta_mode]['augmentation']
    
  
    test_dataset, = get_datasets(
        paths           = dataset_config[dataset]['paths_processed'],
        paths_original  = dataset_config[dataset]['paths_original'],
        splits          = [split],
        image_size      = tta_config['image_size'],
        resolution_proc = dataset_config[dataset]['resolution_proc'],
        dim_proc        = dataset_config[dataset]['dim'],
        n_classes       = n_classes,
        aug_params      = aug_params,
        deformation     = None,
        load_original   = True,
        bg_suppression_opts=None,
    )
        
    print('Datasets loaded')

    # Define the segmentation model
    # :=========================================================================:
    print('Loading segmentation model')
    cpt_type = 'checkpoint_best' if tta_config['load_best_cpt'] \
        else 'checkpoint_last'
    cpt_seg_fp = os.path.join(seg_dir, train_params_seg[cpt_type])
    
    norm, seg = define_and_possibly_load_norm_seg(
        n_classes = n_classes,
        model_params_norm = model_params_norm,
        model_params_seg = model_params_seg,
        cpt_fp = cpt_seg_fp,
        device = device,
        return_norm_seg_state_dict=False,
    )
    
    print('Segmentation model loaded')

    # DDPM
    unconditional_rate          = tta_config[tta_mode]['unconditional_rate']
    cpt_ddpm_fp                 = os.path.join(ddpm_dir, tta_config[tta_mode]['cpt_fn'])
    img_size                    = dataset_config[dataset]['dim'][-1]
    
    ddpm = load_cddpm_from_configs_and_cpt(
        train_ddpm_cfg           = train_params_ddpm,
        model_ddpm_cfg           = model_params_ddpm,
        n_classes                = n_classes,
        cpt_fp                   = cpt_ddpm_fp,
        device                   = device,
        unconditional_rate       = unconditional_rate
    )
   
    # Define the TTADAE object that does the test time adapatation
    # :=========================================================================:    
    learning_rate               = tta_config[tta_mode]['learning_rate']
    learning_rate_norm          = tta_config[tta_mode]['learning_rate_norm']
    learning_rate_seg           = tta_config[tta_mode]['learning_rate_seg']
    learning_rate_ddpm          = tta_config[tta_mode]['learning_rate_ddpm']
    
    # How the segmentation is evaluated
    classes_of_interest         = tta_config[tta_mode]['classes_of_interest']
        
    # DDPM-TTA params    
    ddpm_loss                   = tta_config[tta_mode]['ddpm_loss']
    w_cfg                       = tta_config[tta_mode]['w_cfg']
    pair_sampling_type          = tta_config[tta_mode]['pair_sampling_type']
    t_ddpm_range                = tta_config[tta_mode]['t_ddpm_range']
    t_sampling_strategy         = tta_config[tta_mode]['t_sampling_strategy']
    minibatch_size_ddpm         = tta_config[tta_mode]['minibatch_size_ddpm']

    tta = DiffusionTTA(
        norm                    = norm,
        seg                     = seg,
        ddpm                    = ddpm,          
        n_classes               = n_classes,
        learning_rate           = learning_rate,
        learning_rate_norm      = learning_rate_norm,
        learning_rate_seg       = learning_rate_seg,
        learning_rate_ddpm      = learning_rate_ddpm,
        classes_of_interest     = classes_of_interest,
        ddpm_loss               = ddpm_loss,
        pair_sampling_type      = pair_sampling_type,
        t_ddpm_range            = t_ddpm_range,
        t_sampling_strategy     = t_sampling_strategy,
        minibatch_size_ddpm     = minibatch_size_ddpm,
        w_cfg                   = w_cfg,
        wandb_log               = wandb_log,  
    )
    
    # Do TTA with a DAE
    # :=========================================================================:
    num_steps                   = tta_config[tta_mode]['num_steps']
    batch_size                  = tta_config[tta_mode]['batch_size']
    num_workers                 = tta_config['num_workers']
    save_checkpoints            = tta_config[tta_mode]['save_checkpoints']
    dataset_repetition          = tta_config[tta_mode]['dataset_repetition']
    const_aug_per_volume        = tta_config[tta_mode]['const_aug_per_volume']
    calculate_dice_every        = tta_config[tta_mode]['calculate_dice_every']
    num_t_noise_pairs_per_img   = tta_config[tta_mode]['num_t_noise_pairs_per_img']
    
    if wandb_log:
        wandb.watch([norm], log='all', log_freq=1)
        
    indices_per_volume = test_dataset.get_volume_indices()
    
    start_idx = 0 if tta_config['start'] is None else tta_config['start']
    stop_idx = len(indices_per_volume) if tta_config['stop'] is None \
        else tta_config['stop']
    
    print('---------------------TTA---------------------')
    print('start vol_idx:', start_idx)
    print('end vol_idx:', stop_idx)
    print(f'dataset: {dataset}')
    print(f'Using {num_steps} steps')
    print(f'Using {num_t_noise_pairs_per_img} t and noise samples per step per volume')
    print(f'pair_sampling_type: {pair_sampling_type}')
    print(f'ddpm_loss: {ddpm_loss}')
    print(f'w_cfg: {w_cfg}')
    print(f'const_aug_per_volume: {const_aug_per_volume}')
    print(f'pair_sampling_type: {pair_sampling_type}')
    print(f't_ddpm_range: {t_ddpm_range}')
    print(f't_sampling_strategy: {t_sampling_strategy}')
    print(f'bath_size: {batch_size}')
    print(f'minibatch_size_ddpm: {minibatch_size_ddpm}')
    
    dice_scores = torch.zeros((len(indices_per_volume), n_classes))
    
    dice_per_vol = {}
    
    for i in range(start_idx, stop_idx):
        seed_everything(seed)
        
        indices = indices_per_volume[i]
        print(f'Processing volume {i}')

        volume_dataset = Subset(test_dataset, indices)

        tta.reset_initial_state()

        dice_scores_wrt_gt = tta.tta(
            volume_dataset = volume_dataset,
            dataset_name = dataset,
            index = i,
            num_steps = num_steps,
            num_t_noise_pairs_per_img=num_t_noise_pairs_per_img,
            batch_size = batch_size,
            num_workers=num_workers,
            calculate_dice_every = calculate_dice_every,
            dataset_repetition = dataset_repetition,
            const_aug_per_volume = const_aug_per_volume,
            save_checkpoints = save_checkpoints,
            device=device,
            logdir = logdir
        )
        
        tta.norm.eval()
        #tta.seg.eval()
        dice_scores[i, :], _ = tta.test_volume(
            volume_dataset = volume_dataset,
            dataset_name = dataset,
            index = i,
            batch_size = batch_size,
            num_workers = num_workers,
            device = device,
            logdir = logdir,
            classes_of_interest = classes_of_interest,
        )
        
        dice_scores_wrt_gt[num_steps] = dice_scores[i, 1:].mean().item() 
        
        dice_per_vol[i] = dice_scores_wrt_gt
        
        # Store csv of dice_scores
        dice_per_vol_df = pd.DataFrame(dice_per_vol).add_prefix('vol_')
        dice_per_vol_df.to_csv(os.path.join(
            logdir, 
            f'dice_scores_fg_{dataset}_per_step_'
            f'start_vol_{start_idx}_stop_vol_{stop_idx}.csv')
        )
                
        write_to_csv(
            os.path.join(logdir, f'scores_{dataset}_last_iteration.csv'),
            np.hstack([[[f'volume_{i:02d}']], dice_scores[None, i, :].numpy()]),
            mode='a',
        )

        os.makedirs(os.path.join(logdir, 'optimal_metrics'), exist_ok=True)
            
        print(f'Volume {i} done!! \n\n')

    print(f'Overall mean dice (only foreground): {dice_scores[:, 1:].mean()}')
    
    # Log the dice scores history over the volumes
    # :=========================================================================:
    # plot the mean and std confidence interval of the dice scores over the volumes
        
    plt.errorbar(
        x=dice_per_vol_df.index.values,
        y=dice_per_vol_df.mean(axis=1).values,
        yerr=dice_per_vol_df.mean(axis=1).values, 
        fmt='-x',
    )
    
    plt.xlabel('Step')
    plt.ylabel('Dice score aggregated over volumes')
    plt.title('Dice score (foreground only) aggregated over volumes vs TTA step')
    plt.savefig(os.path.join(logdir, 'Dice score aggregated over volumes vs TTA step.png'))    
    plt.close()
    
    if wandb_log:
        for step, mean_dice_over_vols in dice_per_vol_df.mean(axis=1).items():
            wandb.log({f'mean_dice_over_vols': mean_dice_over_vols}, step=int(step))
    
        wandb.finish()

