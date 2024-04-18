""" 
TODO:
 - Conditioning by multiplication has the difficulty that masks with lower support are not denoised
    - How to weigh each mask by the inverse of their support? 
"""

import os
import re
import sys
import glob
import argparse

import wandb
from accelerate import Accelerator

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..')))

from tta_uia_segmentation.src.dataset.dataset_in_memory_for_ddpm import get_datasets
from tta_uia_segmentation.src.models import ConditionalGaussianDiffusion, ConditionalUnet
from tta_uia_segmentation.src.models.io import load_norm_from_configs_and_cpt
from tta_uia_segmentation.src.train import CDDPMTrainer
from tta_uia_segmentation.src.utils.io import (
    load_config, dump_config, print_config, rewrite_config_arguments)
from tta_uia_segmentation.src.utils.utils import seed_everything, define_device
from tta_uia_segmentation.src.utils.logging import setup_wandb



def preprocess_cmd_args() -> argparse.Namespace:
    """_
    Parse command line arguments and return them as a Namespace object.
    
    All options specified will overwrite whaterver is specified in the config files.
    
    """
    parser = argparse.ArgumentParser(description="Train Segmentation Model (with shallow normalization module)")
    parse_bool = lambda s: s.strip().lower() == 'true'
    
    parser.add_argument('dataset_config_file', type=str, help='Path to yaml config file with parameters that define the dataset.')
    parser.add_argument('model_config_file', type=str, help='Path to yaml config file with parameters that define the model.')
    parser.add_argument('train_config_file', type=str, help='Path to yaml config file with parameters for training.')
    
    # Training parameters. If provided, overrides default parameters from config file.
    # :================================================================================================:
    parser.add_argument('--resume', type=parse_bool, help='Resume training from last checkpoint. Default: True.') 
    parser.add_argument('--logdir', type=str, help='Path to directory where logs and checkpoints are saved. Default: logs')  
    parser.add_argument('--wandb_log', type=parse_bool, help='Log training to wandb. Default: False.')
    parser.add_argument('--start_new_exp', type=parse_bool, help='Start a new wandb experiment. Default: False')

    # Model parameters
    # ----------------:
    parser.add_argument('--dim', type=int, help='Number of feature maps in the first block. Default: 64')
    parser.add_argument('--dim_mults', type=int, nargs='+', help='Multiplicative factors for the number of feature maps in each block. Default: [1, 2, 4, 8]')
    parser.add_argument('--condition_by_mult', type=parse_bool, help='Whether to condition by multiplication or concatenation. Default: False')
    
    # Noising parameters
    # ------------------:
    parser.add_argument('--timesteps', type=int, help='Number of timesteps in diffusion process. Default: 1000')
    parser.add_argument('--save_and_sample_every', type=int, help='Save and sample every n steps. Default: 1000')
    parser.add_argument('--sampling_timesteps', type=int, help='Number of timesteps to sample from. Default: 1000')
    parser.add_argument('--num_validation_samples', type=int, help='Number of samples to generate for metric evaluation. Default: 1000')
    parser.add_argument('--num_viz_samples', type=int, help='Number of samples to generate for visualization. Default: 16')
    
    # Training loop
    # -------------:
    parser.add_argument('--objective', type=str, help='Objective to use for training. Default: gaussian')
    parser.add_argument('--also_uncondtional', type=parse_bool, help='Whether to also train an unconditional model. Default: False')
    parser.add_argument('--batch_size', type=int, help='Batch size for training. Default: 4')
    parser.add_argument('--gradient_accumulate_every', type=int, help='Number of steps to accumulate gradients over. Default: 1')
    parser.add_argument('--train_num_steps', type=int, help='Total number of training steps. Default: 50000') 
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer. Default: 1e-4')
    parser.add_argument('--uncoditional_rate', type=float, help='Rate at which to forward pass unconditionally. Default: 0.2')
    parser.add_argument('--num_workers', type=int, help='Number of workers for dataloader. Default: 0')
    parser.add_argument('--seed', type=int, help='Seed for random number generators. Default: 0')   
    parser.add_argument('--device', type=str, help='Device to use for training. Default cuda', )
    
    # Dataset and its transformations to use for training
    # ---------------------------------------------------:
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for training')
    parser.add_argument('--n_classes', type=int, help='Number of classes in dataset')
    parser.add_argument('--image_size', type=int, nargs='+', help='Size of images in dataset')
    parser.add_argument('--resolution_proc', type=float, nargs='+', help='Resolution of images in dataset')
    parser.add_argument('--rescale_factor', type=float, help='Rescale factor for images in dataset')
    
    parser.add_argument('--norm_dir', type=str, help='Path to directory where normalization model is saved')
    
    args = parser.parse_args()
    
    return args


def get_configuration_arguments() -> tuple[dict, dict, dict]:
    args = preprocess_cmd_args()
    
    dataset_config = load_config(args.dataset_config_file)
    dataset_config = rewrite_config_arguments(dataset_config, args, 'dataset')
    
    model_config = load_config(args.model_config_file)
    model_config = rewrite_config_arguments(model_config, args, 'model')

    train_config = load_config(args.train_config_file)
    train_config = rewrite_config_arguments(train_config, args, 'train')
    
    model_config['ddpm_unet'] = rewrite_config_arguments(
        model_config['ddpm_unet'], args, 'model, ddpm_unet')
    
    train_config['ddpm'] = rewrite_config_arguments(
        train_config['ddpm'], args, 'train, ddpm')
    
    return dataset_config, model_config, train_config


def get_last_milestone(logdir: str) -> int:
    pattern = r'model-(\d+)\.pt'
    checkpoints_fps = glob.glob(os.path.join(logdir, 'model-*.pt'))
    
    assert len(checkpoints_fps) > 0, "No milestone checkpoints found" 
    
    checkpoints_fns = [os.path.basename(fps) for fps in checkpoints_fps]
    
    milestones = [int(re.search(pattern, fn).group(1)) if re.search(pattern, fn) else -1
                  for fn in checkpoints_fns]
    
    last_milestone = max(milestones)
    assert last_milestone != -1, "Could not find the last milestone"
    
    return last_milestone


if __name__ == '__main__':

    print(f'Running {__file__}')
    train_type = 'ddpm'
    accelerator = Accelerator(
            split_batches = True,
            mixed_precision = 'no'
        )
    
    # Loading general parameters
    # :=========================================================================:
    dataset_config, model_config, train_config = get_configuration_arguments()
    
    resume          = train_config['resume']
    seed            = train_config['seed']
    device          = train_config['device']
    wandb_log       = train_config['wandb_log']
    start_new_exp   = train_config['start_new_exp']
    
    logdir          = train_config[train_type]['logdir']
    wandb_project   = train_config[train_type]['wandb_project']
    
    # Write or load parameters to/from logdir, used if a run is resumed.
    # :=========================================================================:
    is_resumed = os.path.exists(os.path.join(logdir, 'params.yaml')) and resume
    if accelerator.is_main_process: print(f'training resumed: {is_resumed}')

    if is_resumed:
        params = load_config(os.path.join(logdir, 'params.yaml'))
        
        # We need the original model and dataset definitions
        dataset_config = params['dataset']
        model_config = params['model']
        
        model_params_norm = params['model']['norm']
        train_params_norm = params['training']['norm']
        
    else:
        os.makedirs(logdir, exist_ok=True)
        
        params_norm = load_config(os.path.join(
        train_config[train_type]['norm_dir'], 'params.yaml'))

        model_params_norm = params_norm['model']['normalization_2D']
        train_params_norm = params_norm['training']

        params = {
            'dataset': dataset_config, 
            'model': {**model_config,  'norm': model_params_norm},
            'training': {**train_config, 'norm': train_params_norm}, 
        }

        if accelerator.is_main_process: dump_config(os.path.join(logdir, 'params.yaml'), params)

    if accelerator.is_main_process: print_config(params, keys=['training', 'model'])

    # Setup wandb logging
    # :=========================================================================:
    if wandb_log and accelerator.is_main_process:
        wandb_dir = setup_wandb(params, logdir, wandb_project, start_new_exp)
    
    # Define the dataset that is to be used for training
    # :=========================================================================:
    if accelerator.is_main_process: print('Defining dataset')
    seed_everything(seed)
    device              = define_device(device)
    dataset             = train_config[train_type]['dataset']
    n_classes           = dataset_config[dataset]['n_classes']
    batch_size          = train_config[train_type]['batch_size']
    norm_dir            = train_config[train_type]['norm_dir']   
                            
    # Load normalization model used in the segmentation network
    if train_config[train_type]['norm_with_nn_on_fly']:
        norm = load_norm_from_configs_and_cpt(
            model_params_norm=model_params_norm,
            cpt_fp=os.path.join(norm_dir, train_params_norm['checkpoint_best']),
            device='cpu'
        )
    else: 
        norm = None
    
    # Dataset definition
    train_dataset, val_dataset = get_datasets(
        splits          = ['train', 'val'],
        norm            = norm,
        paths           = dataset_config[dataset]['paths_processed'],
        paths_normalized_h5 = None, # dataset_config[dataset]['paths_normalized_with_nn'],
        use_original_imgs = train_config[train_type]['use_original_imgs'],
        one_hot_encode  = train_config[train_type]['one_hot_encode'],
        normalize       = train_config[train_type]['normalize'],
        paths_original  = dataset_config[dataset]['paths_original'],
        image_size      = train_config[train_type]['image_size'],
        resolution_proc = dataset_config[dataset]['resolution_proc'],
        dim_proc        = dataset_config[dataset]['dim'],
        n_classes       = n_classes,
        aug_params      = train_config[train_type]['augmentation'],
        bg_suppression_opts = train_config[train_type]['bg_suppression_opts'],
        deformation     = None,
        load_original   = False,
    )
    if accelerator.is_main_process: print('Dataloaders defined')
    
    # Define the denoiser model diffusion pipeline
    # :=========================================================================:
    
    dim                 = model_config['ddpm_unet']['dim']
    dim_mults           = model_config['ddpm_unet']['dim_mults']
    channels            = model_config['ddpm_unet']['channels']
    
    objective           = train_config[train_type]['objective']
    timesteps           = train_config[train_type]['timesteps']
    sampling_timesteps  = train_config[train_type]['sampling_timesteps']
    condition_by_mult   = train_config[train_type]['condition_by_mult']
    also_unconditional  = train_config[train_type]['also_unconditional']
    unconditional_rate  = train_config[train_type]['unconditional_rate']
    
    if accelerator.is_main_process: print(f'Using Device {device}')
    # Model definition
    model = ConditionalUnet(
        dim=dim,
        dim_mults=dim_mults,
        n_classes=n_classes,   
        flash_attn=True,
        image_channels=channels, 
        condition_by_concat= not condition_by_mult
    )
    
    diffusion = ConditionalGaussianDiffusion(
        model=model,
        image_size=train_config[train_type]['image_size'][-1],
        objective=objective,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        also_unconditional=also_unconditional,
        unconditional_rate=unconditional_rate,
    )

    # Execute the training loop
    # :=========================================================================:
    if accelerator.is_main_process: print('Defining trainer: training loop, optimizer and loss')
    
    batch_size                  = train_config[train_type]['batch_size']
    gradient_accumulate_every   = train_config[train_type]['gradient_accumulate_every']
    save_and_sample_every       = train_config[train_type]['save_and_sample_every']
    train_lr                    = float(train_config[train_type]['learning_rate'])
    num_workers                 = train_config[train_type]['num_workers']
    train_num_steps             = train_config[train_type]['train_num_steps']
    save_and_sample_every       = train_config[train_type]['save_and_sample_every']
    num_validation_samples      = train_config[train_type]['num_validation_samples']
    num_viz_samples             = train_config[train_type]['num_viz_samples']
    
    if accelerator.is_main_process:
        print(f'Objective: {objective}')
        print(f'Minibatch size: {batch_size}')
        print(f'Effective batch size: {batch_size * gradient_accumulate_every}')
        print(f'num_validation_samples: {num_validation_samples}')
        print(f'num_viz_samples: {num_viz_samples}')
        print(f'Number of parameters of the model: {sum(p.numel() for p in diffusion.parameters()):,}')
        
    trainer = CDDPMTrainer(
        diffusion,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size = batch_size,
        train_lr = train_lr,
        num_workers=num_workers,
        train_num_steps = train_num_steps,# total training steps
        num_validation_samples=num_validation_samples,          # number of samples to generate for metric evaluation
        num_viz_samples=num_viz_samples,                        # number of samples to generate for visualization
        gradient_accumulate_every = gradient_accumulate_every,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False,            # whether to calculate fid during training 
        results_folder=logdir,
        save_and_sample_every = save_and_sample_every,
        wandb_log=wandb_log,
        accelerator=accelerator
        )
    
    if wandb_log and accelerator.is_main_process:
        #wandb.save(os.path.join(wandb_dir, trainer.get_last_checkpoint_name()), base_path=wandb_dir)
        #wandb.watch([diffusion, model], trainer.get_loss_function(), log='all')
        wandb.watch([diffusion, model], log='all')
        
    # Resume previous point if necessary
    if resume:
        last_milestone = get_last_milestone(logdir)
        if accelerator.is_main_process: print(f'Resuming training from milestone {last_milestone}')
        trainer.load(last_milestone)
        
    # Start training
    # :=========================================================================:
    trainer.train()
    
    if wandb_log:
        wandb.finish()
