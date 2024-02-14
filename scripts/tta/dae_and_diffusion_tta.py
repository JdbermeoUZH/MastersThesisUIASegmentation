import os
import sys
import argparse

import wandb
import torch
from torch.utils.data import Subset
import numpy as np

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

from tta_uia_segmentation.src.tta import TTADAEandDDPM
from tta_uia_segmentation.src.dataset.dataset_in_memory import get_datasets
from tta_uia_segmentation.src.models import Normalization, UNet
from tta_uia_segmentation.src.models.io import load_ddpm_from_configs_and_cpt, load_segmentation_model_and_cpt
from tta_uia_segmentation.src.utils.io import load_config, dump_config, print_config, write_to_csv, rewrite_config_arguments
from tta_uia_segmentation.src.utils.utils import seed_everything, define_device
from tta_uia_segmentation.src.utils.logging import setup_wandb
from tta_uia_segmentation.src.utils.loss import DiceLoss


torch.autograd.set_detect_anomaly(True)

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
    parser.add_argument('--start', required=False, type=int, help='starting volume index to be used for testing')
    parser.add_argument('--stop', required=False, type=int, help='stopping volume index to be used for testing (index not included)')
    parser.add_argument('--logdir', type=str, help='Path to directory where logs and checkpoints are saved. Default: logs')  
    parser.add_argument('--dae_dir', type=str, help='Path to directory where DAE checkpoints are saved')
    parser.add_argument('--seg_dir', type=str, help='Path to directory where segmentation checkpoints are saved')
    parser.add_argument('--wandb_log', type=lambda s: s.strip().lower() == 'true', help='Log tta to wandb. Default: False.')

    # TTA loop
    # -------------:
    parser.add_argument('--alpha', type=float, help='Proportion of how much better the dice of the DAE pseudolabel and predicted segmentation'
                                                    'should be than the dice of the Atlas pseudolabel. Default: 1')
    parser.add_argument('--beta', type=float, help='Minimum dice of the Atlas pseudolabel and the predicted segmentation. Default: 0.25')
    parser.add_argument('--num_steps', type=int, help='Number of steps to take in TTA loop. Default: 100')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer. Default: 1e-4')
    parser.add_argument('--batch_size', type=int, help='Batch size for tta. Default: 4')
    parser.add_argument('--num_workers', type=int, help='Number of workers for dataloader. Default: 0')
    parser.add_argument('--seed', type=int, help='Seed for random number generators. Default: 0')   
    parser.add_argument('--device', type=str, help='Device to use for tta. Default cuda', )
    parser.add_argument('--save_checkpoints', type=lambda s: s.strip().lower() == 'true',
                        help='Whether to save checkpoints. Default: True')
    
    # Dataset and its transformations to use for TTA
    # ---------------------------------------------------:
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for tta. Default: USZ')
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
    parser.add_argument('--aug_gamma_min', type=float, help=' augmentation. Default: 1.0') #TODO: specify what this is
    parser.add_argument('--aug_gamma_max', type=float, help=' augmentation. Default: 1.0') #TODO: specify what this is
    parser.add_argument('--aug_brightness_min', type=float, help='Minimum value for brightness augmentation. Default: 0.0') 
    parser.add_argument('--aug_brightness_max', type=float, help='Maximum value for brightness augmentation. Default: 0.0')   
    parser.add_argument('--aug_noise_mean', type=float, help='Mean value for noise augmentation. Default: 0.0')
    parser.add_argument('--aug_noise_std', type=float, help='Standard deviation value for noise augmentation. Default: 0.0') 
    
    # Backround suppression for normalized images
    parser.add_argument('--bg_supression_type', choices=['fixed_value', 'random_value', 'none', None], help='Type of background suppression to use. Default: fixed_value')
    parser.add_argument('--bg_supression_value', type=float, help='Value to use for background suppression. Default: -0.5')
    parser.add_argument('--bg_supression_min', type=float, help='Minimum value to use for random background suppression. Default: -0.5')
    parser.add_argument('--bg_supression_max', type=float, help='Maximum value to use for random background suppression. Default: 1.0')
    parser.add_argument('--bg_supression_max_source', type=str, choices=['thresholding', 'ground_truth'], help='Maximum value to use for random background suppression. Default: "thresholding"')
    parser.add_argument('--bg_supression_thresholding', type=str, choices=['otsu', 'yen', 'li', 'minimum', 'mean', 'triangle', 'isodata'], help='Maximum value to use for random background suppression. Default: "otsu"') 
    parser.add_argument('--bg_supression_hole_filling', type=lambda s: s.strip().lower() == 'true', help='Whether to use hole filling for background suppression. Default: True')
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
    
    tta_mode                = 'dae_and_diffusion'
    dae_dir                 = tta_config[tta_mode]['dae_dir']
    seg_dir                 = tta_config[tta_mode]['seg_dir']
    ddpm_dir                = tta_config[tta_mode]['ddpm_dir']
    
    params_dae              = load_config(os.path.join(dae_dir, 'params.yaml'))
    model_params_dae        = params_dae['model']['dae']
    train_params_dae        = params_dae['training']
    
    params_seg              = load_config(os.path.join(seg_dir, 'params.yaml'))
    model_params_norm       = params_dae['model']['normalization_2D']
    model_params_seg        = params_dae['model']['segmentation_2D']
    train_params_seg        = params_dae['training']
    
    params_ddpm             = load_config(os.path.join(ddpm_dir, 'params.yaml'))
    model_params_ddpm       = params_ddpm['model']['ddpm_unet']
    train_params_ddpm       = params_ddpm['training']['ddpm']
    
    params                  = { 
                               'datset': dataset_config,
                               'model': {'norm': model_params_norm, 'seg': model_params_seg, 
                                         'dae': model_params_dae, 'ddpm': model_params_ddpm},
                               'training': {'seg': train_params_seg, 'dae': train_params_dae,
                                            'ddpm': train_params_ddpm},
                               'tta': tta_config
                               }
        
    seed                    = tta_config['seed']
    device                  = tta_config['device']
    wandb_log               = tta_config['wandb_log']
    logdir                  = tta_config[tta_mode]['logdir']
    wandb_project           = tta_config[tta_mode]['wandb_project']
    
    os.makedirs(logdir, exist_ok=True)
    dump_config(os.path.join(logdir, 'params.yaml'), params)
    print_config(params, keys=['datset', 'model', 'tta'])

    # Setup wandb logging
    # :=========================================================================:
    if wandb_log:
        wandb_dir = setup_wandb(params, logdir, wandb_project)
    
    # Define the dataset that is to be used for training
    # :=========================================================================:
    print('Defining dataset and its datloader that will be used for TTA')
    seed_everything(seed)
    device                 = define_device(device)
    dataset                = tta_config['dataset']
    n_classes              = dataset_config[dataset]['n_classes']
    bg_suppression_opts    = tta_config[tta_mode]['bg_suppression_opts']
    aug_params             = tta_config[tta_mode]['augmentation']
  
    test_dataset, = get_datasets(
        paths           = dataset_config[dataset]['paths_processed'],
        paths_original  = dataset_config[dataset]['paths_original'],
        splits          = ['test'],
        image_size      = tta_config['image_size'],
        resolution_proc = dataset_config[dataset]['resolution_proc'],
        dim_proc        = dataset_config[dataset]['dim'],
        n_classes       = n_classes,
        aug_params      = aug_params,
        deformation     = None,
        load_original   = True,
        bg_suppression_opts=bg_suppression_opts,
    )
        
    print('Datasets loaded')

    # Define the  segmentation model
    # :=========================================================================:
    norm = Normalization(
        n_layers                = model_params_norm['n_layers'],
        image_channels          = model_params_norm['image_channels'],
        channel_size            = model_params_norm['channel_size'],
        kernel_size             = model_params_norm['kernel_size'],
        activation              = model_params_norm['activation'], 
        batch_norm              = model_params_norm['batch_norm'],
        residual                = model_params_norm['residual'],
        n_dimensions            = model_params_norm['n_dimensions'] 
    ).to(device)

    seg = UNet(
        in_channels             = model_params_seg['image_channels'],
        n_classes               = n_classes,
        channels                = model_params_seg['channel_size'],
        channels_bottleneck     = model_params_seg['channels_bottleneck'],
        skips                   = model_params_seg['skips'],
        n_dimensions            = model_params_seg['n_dimensions'] 
    ).to(device)
    
    dae = UNet(
        in_channels             = n_classes,
        n_classes               = n_classes,
        channels                = model_params_dae['channel_size'],
        channels_bottleneck     = model_params_dae['channels_bottleneck'],
        skips                   = model_params_dae['skips'],
        n_dimensions            = model_params_dae['n_dimensions']
    ).to(device)
    
    ## Load their checkpoints
    # Segmentation network
    checkpoint = torch.load(os.path.join(seg_dir, train_params_seg['checkpoint_best']), 
                            map_location=device)
    norm.load_state_dict(checkpoint['norm_state_dict'])
    seg.load_state_dict(checkpoint['seg_state_dict'])
    norm_state_dict = checkpoint['norm_state_dict']

    # DAE
    checkpoint = torch.load(os.path.join(dae_dir, train_params_dae['checkpoint_best']), 
                            map_location=device)
    dae.load_state_dict(checkpoint['dae_state_dict'])

    checkpoint = torch.load(os.path.join(dae_dir, 'atlas.h5py'), map_location=device)
    atlas = checkpoint['atlas']

    # DDPM

    ddpm = load_ddpm_from_configs_and_cpt(
        train_ddpm_cfg           = train_params_ddpm,
        model_ddpm_cfg           = model_params_ddpm,
        cpt_fp                   = os.path.join(ddpm_dir, tta_config[tta_mode]['cpt_fn']),
        img_size                 = dataset_config[dataset]['dim'][-1],
        device                   = device,
        sampling_timesteps       = tta_config[tta_mode]['sampling_timesteps'],
    )

    del checkpoint
   
    # Define the TTADAE object that does the test time adapatation
    # :=========================================================================:
    learning_rate               = tta_config[tta_mode]['learning_rate']

    tta = TTADAEandDDPM(
        norm                    = norm,
        seg                     = seg,
        dae                     = dae,
        atlas                   = atlas,
        ddpm                    = ddpm,          
        loss_func               = DiceLoss(),
        learning_rate           = learning_rate,
        dddpm_loss_alpha        = tta_config[tta_mode]['dddpm_loss_alpha'],
        min_t_diffusion_tta     = tta_config[tta_mode]['min_t_diffusion_tta'],
        max_t_diffusion_tta     = tta_config[tta_mode]['max_t_diffusion_tta'],
        sampling_timesteps      = tta_config[tta_mode]['sampling_timesteps'],
    )
    
    # Do TTA with a DAE
    # :=========================================================================:
    bg_suppression_opts_tta     = tta_config[tta_mode]['bg_suppression_opts']
    alpha                       = tta_config[tta_mode]['alpha']
    beta                        = tta_config[tta_mode]['beta']
    rescale_factor              = train_params_dae['dae']['rescale_factor']
    num_steps                   = tta_config[tta_mode]['num_steps']
    batch_size                  = tta_config[tta_mode]['batch_size']
    num_workers                 = tta_config['num_workers']
    save_checkpoints            = tta_config[tta_mode]['save_checkpoints']
    update_dae_output_every     = tta_config[tta_mode]['update_dae_output_every']
    dataset_repetition          = tta_config[tta_mode]['dataset_repetition']
    const_aug_per_volume        = tta_config[tta_mode]['const_aug_per_volume']
    accumulate_over_volume      = tta_config[tta_mode]['accumulate_over_volume']
    calculate_dice_every        = tta_config[tta_mode]['calculate_dice_every']
    

    if wandb_log:
        wandb.watch([norm], log='all', log_freq=1)
        
    indices_per_volume = test_dataset.get_volume_indices()
    
    start_idx = 0
    stop_idx = len(indices_per_volume)  # == number of volumes
    if tta_config['start'] is not None:
        start_idx = tta_config['start']
    if tta_config['stop'] is not None:
        stop_idx = tta_config['stop']
    
    print('---------------------TTA---------------------')
    print('start vol_idx:', start_idx)
    print('end vol_idx:', stop_idx)
        
    dice_scores = torch.zeros((len(indices_per_volume), n_classes))
    
    for i in range(start_idx, stop_idx):

        indices = indices_per_volume[i]
        print(f'processing volume {i}')

        volume_dataset = Subset(test_dataset, indices)

        norm.load_state_dict(norm_state_dict)

        norm, norm_dict, metrics_best = tta.tta(
            volume_dataset = volume_dataset,
            dataset_name = dataset,
            n_classes =n_classes,
            index = i,
            rescale_factor_dae = rescale_factor,
            alpha = alpha,
            beta = beta,
            bg_suppression_opts = bg_suppression_opts,
            bg_suppression_opts_tta = bg_suppression_opts_tta,
            num_steps = num_steps,
            batch_size = batch_size,
            num_workers=num_workers,
            calculate_dice_every = calculate_dice_every,
            update_dae_output_every = update_dae_output_every,
            accumulate_over_volume = accumulate_over_volume,
            dataset_repetition = dataset_repetition,
            const_aug_per_volume = const_aug_per_volume,
            save_checkpoints = save_checkpoints,
            device=device,
            logdir = logdir
        )
        
        dice_scores[i, :], _ = tta.test_volume(
            volume_dataset = volume_dataset,
            dataset_name = dataset,
            index = i,
            n_classes = n_classes,
            batch_size = batch_size,
            num_workers = num_workers,
            bg_suppression_opts = bg_suppression_opts,
            device = device,
            logdir = logdir,
        )

        write_to_csv(
            os.path.join(logdir, f'scores_{dataset}_last_iteration.csv'),
            np.hstack([[[f'volume_{i:02d}']], dice_scores[None, i, :].numpy()]),
            mode='a',
        )

        os.makedirs(os.path.join(logdir, 'optimal_metrics'), exist_ok=True)
        dump_config(
            os.path.join(logdir, 'optimal_metrics', f'{dataset}_{i:02d}.yaml'),
            metrics_best,
        )

        for key in norm_dict.keys():
            print(f'Model at minimum {key} = {metrics_best[key]}')

            tta.norm.load_state_dict(norm_dict[key])
            scores, _ = tta.test_volume(
                volume_dataset=volume_dataset,
                dataset_name=dataset,
                index=i,
                n_classes=n_classes,
                batch_size=batch_size,
                num_workers=num_workers,
                appendix=f'_min_{key}',
                bg_suppression_opts=bg_suppression_opts,
                logdir=logdir,
                device=device,
            )

            write_to_csv(
                os.path.join(logdir, f'scores_{dataset}_{key}.csv'),
                np.hstack([[[f'volume_{i:02d}']], scores.numpy()]),
                mode='a',
            )

    print(f'Overall mean dice (only foreground): {dice_scores[:, 1:].mean()}')

    if wandb_log:
        wandb.finish()

