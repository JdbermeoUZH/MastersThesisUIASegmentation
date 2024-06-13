import os
import sys
import argparse
from dataclasses import asdict

import wandb
import torch
from torch.utils.data import Subset
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

from tta_uia_segmentation.src.tta import TTADAEandDDPM
from tta_uia_segmentation.src.dataset.dataset_in_memory import get_datasets
from tta_uia_segmentation.src.models.io import (
    load_cddpm_from_configs_and_cpt,
    load_norm_and_seg_from_configs_and_cpt,
    load_dae_and_atlas_from_configs_and_cpt,
    load_domain_statistiscs
)
from tta_uia_segmentation.src.utils.io import (
    load_config, dump_config, print_config, write_to_csv, rewrite_config_arguments)
from tta_uia_segmentation.src.utils.utils import seed_everything, define_device
from tta_uia_segmentation.src.utils.logging import setup_wandb


torch.autograd.set_detect_anomaly(True)

tta_mode = 'dae_and_ddpm'

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
    parser.add_argument('--dae_dir', type=str, help='Path to directory where DAE checkpoints are saved')
    parser.add_argument('--seg_dir', type=str, help='Path to directory where segmentation checkpoints are saved')
    parser.add_argument('--wandb_log', type=parse_bool, help='Log tta to wandb. Default: False.')
    parser.add_argument('--wandb_project', type=str, help='Name of wandb project to log to. Default: "tta"')
    parser.add_argument('--start_new_exp', type=parse_bool, help='Start a new wandb experiment. Default: False')
    
    # TTA loop
    # -------------:
    # optimization params
    parser.add_argument('--num_steps', type=int, help='Number of steps to take in TTA loop. Default: 100')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer. Default: 1e-4')
    parser.add_argument('--batch_size', type=int, help='Batch size for tta. Default: 4')
    parser.add_argument('--num_workers', type=int, help='Number of workers for dataloader. Default: 0')
    parser.add_argument('--accumulate_over_volume', type=parse_bool, help='Whether to accumulate gradients over volume. Default: False')
    
    parser.add_argument('--ddpm_loss_beta', type=float, help='Weight for DDPM loss. Default: 1.0')
    parser.add_argument('--dae_loss_alpha', type=float, help='Weight for DAE loss. Default: 1.0')
    parser.add_argument('--ddpm_uncond_loss_gamma', type=float, help='Weight for DDPM unconditional loss. Default: 1.0')
    parser.add_argument('--classifier_free_guidance_weight', type=float, help='Weight for classifier free guidance. Default: None')
    parser.add_argument('--x_norm_regularization_loss', type=str, help='Type of x_norm regularization loss. Default: None', choices=['sift', 'zncc', 'mi', 'sq_grad', None])
    parser.add_argument('--x_norm_regularization_eta', type=float, help='Gamma for x_norm regularization loss. Default: 1.0')
    
    parser.add_argument('--dae_loss_except_on_classes', type=int, nargs='+', help='Classes to exclude from DAE loss. Default: None')
    parser.add_argument('--ddpm_loss_only_on_classes', type=int, nargs='+', help='Classes to include in DDPM loss. Default: None')
    
    parser.add_argument('--finetune_bn', type=parse_bool, help='Finetune batchnorm layers. Default: False')
    parser.add_argument('--track_running_stats_bn', type=parse_bool, help='Track running stats of batchnorm layers. Default: False')
    parser.add_argument('--subset_bn_layers', type=int, nargs='+', help='Subset of batchnorm layers to finetune. Default: None')
    
    parser.add_argument('--use_adaptive_beta', type=parse_bool, help='Use adaptive beta for DDPM loss. Default: True')
    parser.add_argument('--adaptive_beta_momentum', type=float, help='Momentum for adaptive beta. ')
    
    parser.add_argument('--frac_vol_diffusion_tta', type=float, help='Fraction of volume to diffuse. Default: 0.5')
    parser.add_argument('--use_ddpm_after_step', type=int, help='Use DDPM after x steps. Default: None')
    parser.add_argument('--use_ddpm_after_dice', type=float, help='Use DDPM after dice is below x. Default: None')
    parser.add_argument('--warmup_steps_for_ddpm_loss', type=int, help='Warmup steps for DDPM loss. Default: 0')
    
    # Manual normalization of statistics before segmentation
    parser.add_argument('--manually_norm_img_before_seg_val', type=parse_bool, help='Whether to manually normalize images before segmentation. Default: False')
    parser.add_argument('--manually_norm_img_before_seg_tta', type=parse_bool, help='Whether to manually normalize images before segmentation in TTA. Default: False')
    parser.add_argument('--normalization_strategy', type=str, help='Normalization strategy to use. Default: None', choices=['standardize', 'min_max', None])
    
    # Seg model params
    parser.add_argument('--bg_supp_x_norm_dae', type=parse_bool, help='Whether to use background suppression for DAE loss. Default: True')
    parser.add_argument('--bg_supp_x_norm_ddpm', type=parse_bool, help='Whether to use background suppression for DDPM. Default: False')
    parser.add_argument('--bg_supp_x_norm_eval', type=parse_bool, help='Whether to use background suppression for evaluation. Default: False')
    
    # DDPM params
    parser.add_argument('--ddpm_dir', type=str, help='Path to directory where DDPM checkpoints are saved')
    parser.add_argument('--cpt_fn', type=str, help='Name of checkpoint file to load for DDPM')
    parser.add_argument('--ddpm_loss', type=str, help='Type of DDPM loss. Default: None', choices=['jacobian', 'sds', 'dds', 'pds', None])
    parser.add_argument('--t_ddpm_range', type=float, nargs=2, help='Quantile range of t values for DDPM. Default: [0.2, 0.98]')       
    parser.add_argument('--t_sampling_strategy', type=str, help='Sampling strategy for t values. Default: uniform')
    parser.add_argument('--unconditional_rate', type=float, help='Rate of unconditional forward passes. Default: 0.0')
    parser.add_argument('--use_y_pred_for_ddpm_loss', type=parse_bool, help='Whether to use predicted segmentation as conditional for DDPM. Default: True')
    parser.add_argument('--use_y_gt_for_ddpm_loss', type=parse_bool, help='Whether to use ground truth segmetnation as conditional for DDPM. ONLY FOR DEBUGGING. Default: False')
    parser.add_argument('--detach_x_norm_from_ddpm_loss', type=parse_bool, help='Whether to detach x_norm from DDPM loss. Default: False')
    parser.add_argument('--minibatch_size_ddpm', type=int, help='Minibatch size for DDPM. Default: 2')
    parser.add_argument('--min_max_intenities_norm_imgs', type=float, nargs=2, help='Min and max intensities for normalization before evaluating DDPM')
    parser.add_argument('--update_norm_td_statistics', type=parse_bool, help='Whether to update the normalization statistics. Default: False')
    
    # DAE and Atlas params
    parser.add_argument('--alpha', type=float, help='Proportion of how much better the dice of the DAE pseudolabel and predicted segmentation'
                                                    'should be than the dice of the Atlas pseudolabel. Default: 1')
    parser.add_argument('--beta', type=float, help='Minimum dice of the Atlas pseudolabel and the predicted segmentation. Default: 0.25')
    parser.add_argument('--use_atlas_only_for_intit', type=parse_bool, help='Whether to use the atlas only for initialization. Default: False')
    parser.add_argument('--calculate_dice_every', type=int, help='Calculate dice every x steps. Default: 25')

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
    
    # Backround suppression for normalized images
    parser.add_argument('--bg_supression_type', choices=['fixed_value', 'random_value', 'none', None], help='Type of background suppression to use. Default: none')
    
    parser.add_argument('--min_max_quantile', type=float, nargs=2, help='Quantile range for min max clipping with manual normalization. Default: [0.1, 0.975]')

    args = parser.parse_args()
    
    _check_args(args)
    
    return args


def _check_args(args: argparse.Namespace) -> None:
    if args.use_y_gt_for_ddpm_loss and args.use_y_pred_for_ddpm_loss:
        raise ValueError('Cannot use both y_gt and y_pred for DDPM loss')


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
    
    tta_config[tta_mode]['bg_suppression_opts'] = rewrite_config_arguments(
        tta_config[tta_mode]['bg_suppression_opts'], args, 'tta, tta_mode, bg_suppression_opts',
        prefix_to_remove='bg_supression_')
    
    return dataset_config, tta_config


if __name__ == '__main__':
    
    # Load Hyperparameters
    print(f'Running {__file__}')
    
    # Loading general parameters
    # :=========================================================================:
    dataset_config, tta_config = get_configuration_arguments()

    seg_dir                 = tta_config['seg_dir']

    dae_dir                 = tta_config[tta_mode]['dae_dir']
    ddpm_dir                = tta_config[tta_mode]['ddpm_dir']
    
    params_dae              = load_config(os.path.join(dae_dir, 'params.yaml'))
    model_params_dae        = params_dae['model']['dae']
    train_params_dae        = params_dae['training']
    
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
                                         'dae': model_params_dae, 'ddpm': model_params_ddpm},
                               'training': {'seg': train_params_seg, 'dae': train_params_dae,
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
    bg_suppression_opts    = tta_config['bg_suppression_opts']
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
        bg_suppression_opts=bg_suppression_opts,
    )
        
    print('Datasets loaded')

    # Define the segmentation model
    # :=========================================================================:
    print('Loading segmentation model')
    cpt_type = 'checkpoint_best' if tta_config['load_best_cpt'] \
        else 'checkpoint_last'
    cpt_seg_fp = os.path.join(seg_dir, train_params_seg[cpt_type])
    
    norm, seg, norm_seg_state_sd = load_norm_and_seg_from_configs_and_cpt(
        n_classes = n_classes,
        model_params_norm = model_params_norm,
        model_params_seg = model_params_seg,
        cpt_fp = cpt_seg_fp,
        device = device,
        return_norm_seg_state_dict=True,
    )
    
    # Load statistics of the source domain
    min_max_clip_q = tta_config[tta_mode]['min_max_quantile']   
    
    norm_sd_statistics = load_domain_statistiscs(
        cpt_fp = cpt_seg_fp,
        frozen = True,
        momentum = 0.96,
        min_max_clip_q=min_max_clip_q,
    )
    
    norm_seg_state_sd['norm_td_statistics'] = asdict(norm_sd_statistics)

    print('Segmentation model loaded')
    
    # DAE
    dae, atlas = load_dae_and_atlas_from_configs_and_cpt(
        n_classes = n_classes,
        model_params_dae = model_params_dae,
        cpt_fp = os.path.join(dae_dir, train_params_dae[cpt_type]),
        device = device,
    )

    # DDPM
    sampling_timesteps          = tta_config[tta_mode]['sampling_timesteps'] 
    unconditional_rate          = tta_config[tta_mode]['unconditional_rate']
    cpt_ddpm_fp                 = os.path.join(ddpm_dir, tta_config[tta_mode]['cpt_fn'])
    img_size                    = dataset_config[dataset]['dim'][-1]
    
    ddpm_loss_only_on_classes   = tta_config[tta_mode]['ddpm_loss_only_on_classes']
    ddpm = load_cddpm_from_configs_and_cpt(
        train_ddpm_cfg           = train_params_ddpm,
        model_ddpm_cfg           = model_params_ddpm,
        n_classes                = n_classes if ddpm_loss_only_on_classes is None else len(ddpm_loss_only_on_classes) + 1,
        cpt_fp                   = cpt_ddpm_fp,
        device                   = device,
        sampling_timesteps       = sampling_timesteps,
        unconditional_rate       = unconditional_rate
    )
   
    # Define the TTADAE object that does the test time adapatation
    # :=========================================================================:    
    learning_rate               = tta_config[tta_mode]['learning_rate']
    dae_loss_alpha              = tta_config[tta_mode]['dae_loss_alpha']
    ddpm_loss_beta              = tta_config[tta_mode]['ddpm_loss_beta']
    ddpm_uncond_loss_gamma      = tta_config[tta_mode]['ddpm_uncond_loss_gamma']
    classifier_free_guidance_weight = tta_config[tta_mode]['classifier_free_guidance_weight']

    # Finetunning of batchnorm layers
    finetune_bn                 = tta_config[tta_mode]['finetune_bn']
    subset_bn_layers            = tta_config[tta_mode]['subset_bn_layers']
    track_running_stats_bn      = tta_config[tta_mode]['track_running_stats_bn']
    
    # Supress background of x_norm
    bg_supp_x_norm_dae          = tta_config[tta_mode]['bg_supp_x_norm_dae']
    bg_supp_x_norm_ddpm         = tta_config[tta_mode]['bg_supp_x_norm_ddpm']
    bg_supp_x_norm_eval         = tta_config[tta_mode]['bg_supp_x_norm_eval']
    
    # Adaptive beta for DDPM loss
    use_adaptive_beta           = tta_config[tta_mode]['use_adaptive_beta']
    adaptive_beta_momentum      = tta_config[tta_mode]['adaptive_beta_momentum']

    # Regularization btw x and x_norm
    x_norm_regularization_eta   = tta_config[tta_mode]['x_norm_regularization_eta']
    x_norm_regularization_loss  = tta_config[tta_mode]['x_norm_regularization_loss']
    
    # How the segmentation is evaluated
    bg_suppression_opts_tta     = tta_config[tta_mode]['bg_suppression_opts']
    update_norm_td_statistics   = tta_config[tta_mode]['update_norm_td_statistics'] 
    manually_norm_img_before_seg_val= tta_config[tta_mode]['manually_norm_img_before_seg_val']
    manually_norm_img_before_seg_tta= tta_config[tta_mode]['manually_norm_img_before_seg_tta']
    normalization_strategy      = tta_config[tta_mode]['normalization_strategy']
    classes_of_interest         = tta_config[tta_mode]['classes_of_interest']
    
    # DAE-TTA params
    alpha                       = tta_config[tta_mode]['alpha']
    beta                        = tta_config[tta_mode]['beta']
    rescale_factor              = train_params_dae['dae']['rescale_factor']
    use_atlas_only_for_init     = tta_config[tta_mode]['use_atlas_only_for_init']
    dae_loss_except_on_classes  = tta_config[tta_mode]['dae_loss_except_on_classes']
    
    # DDPM-TTA params    
    ddpm_loss                   = tta_config[tta_mode]['ddpm_loss']
    min_max_intenities_norm_imgs= tta_config[tta_mode]['min_max_intenities_norm_imgs']
    minibatch_size_ddpm         = tta_config[tta_mode]['minibatch_size_ddpm']
    frac_vol_diffusion_tta      = tta_config[tta_mode]['frac_vol_diffusion_tta']
    t_ddpm_range                = tta_config[tta_mode]['t_ddpm_range']
    t_sampling_strategy         = tta_config[tta_mode]['t_sampling_strategy']
    sampling_timesteps          = tta_config[tta_mode]['sampling_timesteps']
    use_y_pred_for_ddpm_loss    = tta_config[tta_mode]['use_y_pred_for_ddpm_loss']
    use_y_gt_for_ddpm_loss      = tta_config[tta_mode]['use_y_gt_for_ddpm_loss']
    detach_x_norm_from_ddpm_loss= tta_config[tta_mode]['detach_x_norm_from_ddpm_loss']
    use_ddpm_after_step         = tta_config[tta_mode]['use_ddpm_after_step']
    use_ddpm_after_dice         = tta_config[tta_mode]['use_ddpm_after_dice']
    warmup_steps_for_ddpm_loss  = tta_config[tta_mode]['warmup_steps_for_ddpm_loss']
    ddpm_loss_only_on_classes   = tta_config[tta_mode]['ddpm_loss_only_on_classes']
    
    # Only use adaptive beta if either of the DDPM losses are used along with the DAE loss
    use_adaptive_beta = use_adaptive_beta and \
        ((ddpm_loss_beta > 0 or ddpm_uncond_loss_gamma > 0) and dae_loss_alpha > 0)
    
    tta = TTADAEandDDPM(
        norm                    = norm,
        seg                     = seg,
        dae                     = dae,
        atlas                   = atlas,
        norm_sd_statistics      = norm_sd_statistics,
        ddpm                    = ddpm,          
        n_classes               = n_classes,
        learning_rate           = learning_rate,
        finetune_bn             = finetune_bn,
        track_running_stats_bn  = track_running_stats_bn,
        subset_bn_layers        = subset_bn_layers,
        bg_supp_x_norm_dae      = bg_supp_x_norm_dae,
        bg_supp_x_norm_ddpm     = bg_supp_x_norm_ddpm,
        bg_supp_x_norm_eval     = bg_supp_x_norm_eval,        
        dae_loss_alpha          = dae_loss_alpha,
        alpha                   = alpha,
        beta                    = beta,
        use_atlas_only_for_init = use_atlas_only_for_init,
        dae_loss_except_on_classes=dae_loss_except_on_classes,
        update_norm_td_statistics=update_norm_td_statistics,
        manually_norm_img_before_seg_val=manually_norm_img_before_seg_val,
        manually_norm_img_before_seg_tta=manually_norm_img_before_seg_tta,
        classes_of_interest     = classes_of_interest,
        normalization_strategy  = normalization_strategy,
        rescale_factor          = rescale_factor,
        ddpm_loss               = ddpm_loss,
        ddpm_loss_only_on_classes=ddpm_loss_only_on_classes,
        ddpm_loss_beta          = ddpm_loss_beta,
        use_adaptive_beta       = use_adaptive_beta,
        adaptive_beta_momentum  = adaptive_beta_momentum,
        ddpm_uncond_loss_gamma  = ddpm_uncond_loss_gamma,
        minibatch_size_ddpm     = minibatch_size_ddpm,
        frac_vol_diffusion_tta  = frac_vol_diffusion_tta,
        t_ddpm_range            = t_ddpm_range,
        t_sampling_strategy     = t_sampling_strategy,
        min_max_intenities_norm_imgs=min_max_intenities_norm_imgs,
        use_y_pred_for_ddpm_loss=use_y_pred_for_ddpm_loss,
        use_y_gt_for_ddpm_loss  = use_y_gt_for_ddpm_loss,
        detach_x_norm_from_ddpm_loss=detach_x_norm_from_ddpm_loss,
        use_ddpm_after_step     = use_ddpm_after_step,
        use_ddpm_after_dice     = use_ddpm_after_dice,
        warmup_steps_for_ddpm_loss=warmup_steps_for_ddpm_loss,
        x_norm_regularization_loss_eta = x_norm_regularization_eta,
        x_norm_regularization_loss = x_norm_regularization_loss,
        wandb_log               = wandb_log,
        device                  = device,
        bg_suppression_opts     = bg_suppression_opts,
        bg_suppression_opts_tta = bg_suppression_opts_tta,    
    )
    
    # Do TTA with a DAE
    # :=========================================================================:
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
    
    start_idx = 0 if tta_config['start'] is None else tta_config['start']
    stop_idx = len(indices_per_volume) if tta_config['stop'] is None \
        else tta_config['stop']
    
    print('---------------------TTA---------------------')
    print('start vol_idx:', start_idx)
    print('end vol_idx:', stop_idx)
    
    if dae_loss_alpha > 0:
        if beta <= 1.0:
            print(f'Using DAE and Atlas loss with weight: {dae_loss_alpha: .5f}')
            if use_atlas_only_for_init:
                print(f'Using only the Atlas for initialization'
                      'Once it switches to the DAE, it will not switch back to the Atlas')

        else:
            print(f'Using only the Atlas as a pseudo-label with weight: {ddpm_loss_beta: .5f}')    
                
    if ddpm_loss_beta > 0:
        print(f'Using DDPM loss with weight: {ddpm_loss_beta}')
        print(f'DDPM loss type: {ddpm_loss}')
    
        if use_adaptive_beta:
            print(f'Using adaptive beta for DDPM loss with momentum: {adaptive_beta_momentum}')
            
    if ddpm_uncond_loss_gamma > 0 and ddpm.also_unconditional:
        if classifier_free_guidance_weight is None:
            print(f'Using DDPM unconditional loss with weight: {ddpm_uncond_loss_gamma}')
        else:
            print(f'Using DDPM unconditional loss with weight: {ddpm_uncond_loss_gamma} '
                  f'in classifier free guidance mode with weight: {classifier_free_guidance_weight}')
            
        if use_adaptive_beta:
            print(f'Using adaptive beta for DDPM loss with momentum: {adaptive_beta_momentum}')
            
    if x_norm_regularization_eta > 0:
        print(f'Using x_norm regularization with eta: {x_norm_regularization_eta}'
              f' and loss: {x_norm_regularization_loss}')
        
    dice_scores = torch.zeros((len(indices_per_volume), n_classes))
    
    dice_per_vol = {}
    
    for i in range(start_idx, stop_idx):
        seed_everything(seed)
        
        indices = indices_per_volume[i]
        print(f'Processing volume {i}')

        volume_dataset = Subset(test_dataset, indices)

        tta.reset_initial_state(norm_seg_state_sd)

        norm_seg_dict, metrics_best, dice_scores_wrt_gt = tta.tta(
            volume_dataset = volume_dataset,
            dataset_name = dataset,
            index = i,
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
        
        # Calculate dice scores for the last step
        tta.norm.eval()
        dice_scores[i, :], _ = tta.test_volume(
            volume_dataset = volume_dataset,
            dataset_name = dataset,
            index = i,
            batch_size = batch_size,
            num_workers = num_workers,
            device = device,
            logdir = logdir,
            bg_suppression_opts=bg_suppression_opts,
            classes_of_interest=classes_of_interest,
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
        dump_config(
            os.path.join(logdir, 'optimal_metrics', f'{dataset}_{i:02d}.yaml'),
            metrics_best,
        )

        for key in norm_seg_dict.keys():
            # Save the normalization statistics of the best model (model with higest agreement to PL)
            print(f'Model at minimum {key} (best agreement with PL) = {metrics_best[key]}')

            tta.load_state_norm_seg_dict(norm_seg_dict[key])
            scores, _ = tta.test_volume(
                volume_dataset=volume_dataset,
                dataset_name=dataset,
                index=i,
                batch_size=batch_size,
                num_workers=num_workers,
                appendix=f'_min_{key}',
                bg_suppression_opts=bg_suppression_opts,
                classes_of_interest=classes_of_interest,
                logdir=logdir,
                device=device,
            )

            write_to_csv(
                os.path.join(logdir, f'scores_{dataset}_{key}.csv'),
                np.hstack([[[f'volume_{i:02d}']], scores.numpy()]),
                mode='a',
            )
            
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

