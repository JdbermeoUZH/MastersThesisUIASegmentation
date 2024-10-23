import os
import re
import sys
import glob
import argparse

import wandb
from diffusers import AutoencoderKL, UNet2DConditionModel, UNet2DModel

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..')))

from tta_uia_segmentation.src.dataset.dataset_in_memory_for_ddpm import get_datasets
from tta_uia_segmentation.src.models import ConditionalLatentGaussianDiffusion
from tta_uia_segmentation.src.models.io import load_norm_from_configs_and_cpt
from tta_uia_segmentation.src.train import CDDPMTrainer
from tta_uia_segmentation.src.utils.io import (
    load_config, dump_config, print_config, rewrite_config_arguments)
from tta_uia_segmentation.src.utils.utils import (
    seed_everything, is_main_process, print_if_main_process, count_parameters, parse_bool)
from tta_uia_segmentation.src.utils.logging import setup_wandb


def preprocess_cmd_args() -> argparse.Namespace:
    """_
    Parse command line arguments and return them as a Namespace object.
    
    All options specified will overwrite whaterver is specified in the config files.
    
    """
    parser = argparse.ArgumentParser(description="Train Segmentation Model (with shallow normalization module)")
    
    parser.add_argument('dataset_config_file', type=str, help='Path to yaml config file with parameters that define the dataset.')
    parser.add_argument('model_config_file', type=str, help='Path to yaml config file with parameters that define the model.')
    parser.add_argument('train_config_file', type=str, help='Path to yaml config file with parameters for training.')
    
    parser.add_argument('--seed', type=int, help='Seed for random number generators. Default: 0')   
    
    # Training parameters. If provided, overrides default parameters from config file.
    # :================================================================================================:
    parser.add_argument('--resume', type=parse_bool, help='Resume training from last checkpoint. Default: True.') 
    parser.add_argument('--logdir', type=str, help='Path to directory where logs and checkpoints are saved. Default: logs')  
    parser.add_argument('--wandb_log', type=parse_bool, help='Log training to wandb. Default: False.')
    parser.add_argument('--start_new_exp', type=parse_bool, help='Start a new wandb experiment. Default: False')

    # Model parameters
    # ----------------:
    parser.add_argument('--dim', type=int, help='Number of feature maps in the first block. Default: 64')
    parser.add_argument('--dim_mults', type=int, nargs='+', help='Multiplicative factors for the number of feature maps in each block. Default: [1, 2, 2, 2]')
    parser.add_argument('--channels', type=int, help='Number of channels in the input image. Default: 3')
    parser.add_argument('--use_x_attention', type=parse_bool, help='Whether to use cross-attention. Default: False')
    
    parser.add_argument('--fit_emb_for_cond_img', type=parse_bool, help='Whether to fit the embedding for the conditional image. Default: True')

    # Noising parameters
    # ------------------:
    parser.add_argument('--timesteps', type=int, help='Number of timesteps in diffusion process. Default: 1000')
    parser.add_argument('--reset_betas_zero_snr', type=parse_bool, help='Whether to reset betas to zero snr. Default: False')

    # Training Loss
    # ------------:
    parser.add_argument('--objective', type=str, help='Objective to use for training. Default: pred_v', choices=['pred_noise', 'pred_xt_m_1', 'pred_v', 'pred_x0'])
    parser.add_argument('--snr_weighting_gamma', type=float, help='Gamma parameter for snr weighting. Default: None or 5.0')
    parser.add_argument('--optimizer_type', type=str, help='Type of optimizer to use. Default: "adam"', choices=['adam', 'adamw', 'adamW8bit'])
    
    # Training loop
    # -------------:
    parser.add_argument('--train_num_steps', type=int, help='Total number of training steps. Default: 50000') 
    
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer. Default: 1e-4')
    parser.add_argument('--batch_size', type=int, help='Batch size for training. Default: 4')
    parser.add_argument('--gradient_accumulate_every', type=int, help='Number of steps to accumulate gradients over. Default: 1')
    parser.add_argument('--num_workers', type=int, help='Number of workers for dataloader. Default: 0')
    
    parser.add_argument('--unconditional_rate', type=float, help='Rate at which to forward pass unconditionally. Default: 0.2')
    
    parser.add_argument('--amp', type=parse_bool, help='Use automatic mixed precision. Default: True')
    parser.add_argument('--mixed_precision_type', type=str, help='Type of mixed precision to use. Default "fp16"', choices=["fp16", "bf16", "fp8"])
    parser.add_argument('--allow_tf32', type=parse_bool, help='Allow tf32. Use for Ampere GPUs. Default: False')
    parser.add_argument('--enable_xformers', type=parse_bool, help='Enable xformers. Default: False')
    parser.add_argument('--gradient_checkpointing', type=parse_bool, help='Use gradient checkpointing. Default: False')

    # Evaluation Parameters
    # ---------------------:
    parser.add_argument('--log_val_loss_every', type=int, help='Log validation loss every n steps. Default: 250')
    parser.add_argument('--save_and_sample_every', type=int, help='Save and sample every n steps. Default: 1000')
    parser.add_argument('--sampling_timesteps', type=int, help='Number of timesteps to sample from. Default: 1000')
    parser.add_argument('--num_validation_samples', type=int, help='Number of samples to generate for metric evaluation. Default: 1000')
    parser.add_argument('--num_viz_samples', type=int, help='Number of samples to generate for visualization. Default: 16')

    parser.add_argument('--calculate_fid', type=parse_bool, help='Whether to calculate FID during training. Default: False')
    
    # Dataset and its transformations to use for training
    # ---------------------------------------------------:
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for training')
    
    parser.add_argument('--use_original_imgs', type=parse_bool, help='Whether to use original images for training. Default: False')
    
    parser.add_argument('--image_size', type=int, nargs='+', help='Size of images in dataset')
    parser.add_argument('--rescale_factor', type=float, nargs='+', help='Rescale factor for images in dataset')
    parser.add_argument('--norm_q_range', type=float, nargs=2, help='Quantile range for normalization model')
    
    parser.add_argument('--norm_with_nn_on_fly', type=parse_bool, help='Whether to normalize with nn on the fly. Default: False')
    parser.add_argument('--norm_dir', type=str, nargs='*', help='Path to directory where normalization model is saved')
    
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
    
    model_config['lddpm_unet'] = rewrite_config_arguments(
        model_config['lddpm_unet'], args, 'model, lddpm_unet')
    
    train_config['lddpm'] = rewrite_config_arguments(
        train_config['lddpm'], args, 'train, lddpm')
    
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
    train_type = 'lddpm'
    unet_type = 'lddpm_unet'
    
    # Loading general parameters
    # :=========================================================================:
    dataset_config, model_config, train_config = get_configuration_arguments()
    
    resume          = train_config['resume']
    seed            = train_config['seed']
    wandb_log       = train_config['wandb_log']
    start_new_exp   = train_config['start_new_exp']
    
    logdir          = train_config[train_type]['logdir']
    wandb_project   = train_config[train_type]['wandb_project']
    
    # Write or load parameters to/from logdir, if a run is resumed.
    # :=========================================================================:
    is_resumed = os.path.exists(os.path.join(logdir, 'params.yaml')) and resume
    print_if_main_process(f'training resumed: {is_resumed}')

    if is_resumed:
        params = load_config(os.path.join(logdir, 'params.yaml'))
        
        # We need the original model and dataset definitions
        dataset_config = params['dataset']
        model_config = params['model']
        
        model_params_norm = params['model']['norm']
        train_params_norm = params['training']['norm']
        
    else:
        os.makedirs(logdir, exist_ok=True)
        
        params = {
            'dataset': dataset_config, 
            'model': {**model_config},
            'training': {**train_config}, 
        }
        
        # If using images normalized with a network, load the normalization parameters
        if not train_config[train_type]['use_original_imgs']:
            params_norm = load_config(os.path.join(
                train_config[train_type]['norm_dir'], 'params.yaml'))
            model_params_norm = params_norm['model']['normalization_2D']
            train_params_norm = params_norm['training']
            
            params['model']['norm'] = model_params_norm
            params['training']['norm'] = train_params_norm

        if is_main_process(): dump_config(os.path.join(logdir, 'params.yaml'), params)

    if is_main_process(): print_config(params, keys=['training', 'model'])

    # Setup wandb logging
    # :=========================================================================:
    if wandb_log and is_main_process():
        wandb_dir = setup_wandb(params, logdir, wandb_project, start_new_exp)
    
    # Define the dataset that is to be used for training
    # :=========================================================================:
    print_if_main_process('Defining dataset')

    seed_everything(seed)
    dataset             = train_config['dataset']
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
    image_size          = train_config[train_type]['image_size']
    vae_pretrained_on_nat_images = train_config[train_type]['vae_pretrained_on_nat_images']

    train_dataset, val_dataset = get_datasets(
        dataset_name    = dataset,
        splits          = ['train', 'val'],
        norm            = norm,
        paths           = dataset_config[dataset]['paths_processed'],
        paths_normalized_h5 = None, # dataset_config[dataset]['paths_normalized_with_nn'],
        use_original_imgs = train_config[train_type]['use_original_imgs'],
        one_hot_encode  = True,
        normalize       = train_config[train_type]['normalize'],
        norm_q_range    = train_config[train_type]['norm_q_range'],
        paths_original  = dataset_config[dataset]['paths_original'],
        image_size      = image_size,
        resolution_proc = dataset_config[dataset]['resolution_proc'],
        dim_proc        = dataset_config[dataset]['dim'],
        n_classes       = n_classes,
        aug_params      = train_config[train_type]['augmentation'],
        rescale_factor  = train_config[train_type]['rescale_factor'],
        load_original   = False,
        return_imgs_in_rgb = vae_pretrained_on_nat_images
    )
    print_if_main_process('Dataloaders defined')

    # Store in logdir the max and min, and class weights of the model (if any)
    ddpm_additional_params = {
        'max_intensity': train_dataset.images_max.item(),
        'min_intensity': train_dataset.images_min.item(),
    }
    
    if train_dataset.class_weights is not None:
        ddpm_additional_params['class_weights'] = train_dataset.class_weights.tolist()
    
    if is_main_process():
        dump_config(os.path.join(logdir, 'ddpm_additional_params.yaml'), ddpm_additional_params)
    
    # Define the diffusion pipeline
    # :=========================================================================:
    
    # Load Image encoder
    vae_path            = train_config[train_type]['vae_path']
    if vae_pretrained_on_nat_images:
        vae = AutoencoderKL.from_pretrained(vae_path, subfolder='vae')
    else:
        vae = AutoencoderKL.from_pretrained(vae_path)

    # Define Denoising Unet    
    dim                 = model_config[unet_type]['dim']
    dim_mults           = model_config[unet_type]['dim_mults']
    channels            = model_config[unet_type]['channels']
    use_x_attention     = model_config[unet_type]['use_x_attention']
    cond_type           = model_config[unet_type]['cond_type']
    time_embedding_dim  = model_config[unet_type]['time_embedding_dim']

    num_downsamples     = len(dim_mults) - 1
    train_image_size    = image_size[-1] * train_config[train_type]['rescale_factor'][-1]
    img_latent_dim      = train_image_size / (2 ** num_downsamples)
    block_out_channels  = [dim * m for m in dim_mults]
    in_channels         = vae.config.latent_channels 
    in_channels         *= 2 if cond_type == 'concat' else 1
    out_channels        = vae.config.latent_channels
    down_attn_type_blocks = 'CrossAttnDownBlock2D' if use_x_attention else 'AttnDownBlock2D'
    up_attn_type_blocks = 'CrossAttnUpBlock2D' if use_x_attention else 'AttnUpBlock2D'
    unet_cls            = UNet2DConditionModel if use_x_attention else UNet2DModel

    down_block_types = [down_attn_type_blocks] * num_downsamples + ['DownBlock2D']
    up_block_types = ['UpBlock2D'] + [up_attn_type_blocks] * num_downsamples 
     
    unet_kwargs = dict(
        sample_size = img_latent_dim,
        in_channels = in_channels,
        out_channels = out_channels,
        block_out_channels = block_out_channels,
        down_block_types = down_block_types,
        up_block_types = up_block_types,
    )

    if use_x_attention:
        unet_kwargs['cross_attention_dim'] = time_embedding_dim
        unet_kwargs['time_embedding_dim'] = time_embedding_dim

    unet = unet_cls(**unet_kwargs)
    
    # Define the diffusion model    
    objective           = train_config[train_type]['objective']
    timesteps           = train_config[train_type]['timesteps']
    sampling_timesteps  = train_config[train_type]['sampling_timesteps']
    
    unconditional_rate  = train_config[train_type]['unconditional_rate']

    fit_emb_for_cond_img= train_config[train_type]['fit_emb_for_cond_img']
    cond_type           = model_config[unet_type]['cond_type']

    snr_weighting_gamma = train_config[train_type]['snr_weighting_gamma']
    reset_betas_zero_snr = train_config[train_type]['reset_betas_zero_snr']

    ddpm = ConditionalLatentGaussianDiffusion(
        vae=vae,
        unet=unet,
        train_image_size=train_image_size,
        cond_img_channels=n_classes,
        objective=objective,
        forward_type='train_loss',
        num_train_timesteps=timesteps,
        num_sample_timesteps=sampling_timesteps,
        unconditional_rate=unconditional_rate,
        fit_emb_for_cond_img=fit_emb_for_cond_img,
        cond_type=cond_type,
        snr_weighting_gamma=snr_weighting_gamma,
        reset_betas_zero_snr=reset_betas_zero_snr,
    )
    
    # Execute the training loop
    # :=========================================================================:
    print_if_main_process('Defining trainer: training loop, optimizer and loss')
    train_num_steps             = train_config[train_type]['train_num_steps']

    optimizer_type              = train_config[train_type]['optimizer_type']
    train_lr                    = float(train_config[train_type]['learning_rate'])
    scale_lr                    = train_config[train_type]['scale_lr']
    batch_size                  = train_config[train_type]['batch_size']
    gradient_accumulate_every   = train_config[train_type]['gradient_accumulate_every']
    num_workers                 = train_config[train_type]['num_workers']

    amp                         = train_config[train_type]['amp']
    mixed_precision_type        = train_config[train_type]['mixed_precision_type']
    enable_xformers             = train_config[train_type]['enable_xformers']
    gradient_checkpointing      = train_config[train_type]['gradient_checkpointing']
    allow_tf32                  = train_config[train_type]['allow_tf32']

    save_and_sample_every       = train_config[train_type]['save_and_sample_every']
    log_val_loss_every          = train_config[train_type]['log_val_loss_every']
    num_validation_samples      = train_config[train_type]['num_validation_samples']
    num_viz_samples             = train_config[train_type]['num_viz_samples']
    calculate_fid               = train_config[train_type]['calculate_fid']

    trainer = CDDPMTrainer(
        ddpm,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer_type=optimizer_type,
        train_batch_size = batch_size,
        train_lr = train_lr,
        scale_lr = scale_lr,
        num_workers=num_workers,
        train_num_steps = train_num_steps,
        gradient_accumulate_every=gradient_accumulate_every,    # gradient accumulation steps
        calculate_fid=calculate_fid,            # whether to calculate fid during training 
        log_val_loss_every=log_val_loss_every,
        save_and_sample_every=save_and_sample_every,
        num_validation_samples=num_validation_samples,          # number of samples to generate for metric evaluation
        num_viz_samples=num_viz_samples,   
        results_folder=logdir,
        wandb_log=wandb_log,
        amp=amp,                          # turn on mixed precision
        mixed_precision_type=mixed_precision_type,              # mixed precision type
        enable_xformers=enable_xformers,
        gradient_checkpointing=gradient_checkpointing,
        allow_tf32=allow_tf32,
    )
    
    if wandb_log and is_main_process():
        #wandb.save(os.path.join(wandb_dir, trainer.get_last_checkpoint_name()), base_path=wandb_dir)
        #wandb.watch([diffusion, model], trainer.get_loss_function(), log='all')
        wandb.watch([ddpm], log='all')
        
    # Resume previous point if necessary
    if resume:
        last_milestone = get_last_milestone(logdir)
        print_if_main_process(f'Resuming training from milestone {last_milestone}')
        trainer.load(last_milestone)
        
    # Start training
    # :=========================================================================:
    
    conditions_run = f"""
    Image size for DDPM: {train_image_size} x {train_image_size}
    Using Cross Attention layers: {use_x_attention}
    Objective: {objective}
    Minibatch size: {batch_size}
    Effective batch size: {batch_size * gradient_accumulate_every}
    num_validation_samples: {num_validation_samples}
    num_viz_samples: {num_viz_samples}
    Number of parameters of the model: {count_parameters(ddpm):,}
    Using AMP: {amp}
    device: {trainer.device}
    """

    print_if_main_process(conditions_run)

    trainer.train()
    
    if wandb_log:
        wandb.finish()
