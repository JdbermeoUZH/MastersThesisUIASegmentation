"""
Train a diffusion model on images.
"""
import os
import sys
import argparse

import wandb
from mpi4py import MPI
import torch.distributed as dist
from improved_diffusion import dist_util, logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    create_gaussian_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..')))

from tta_uia_segmentation.src.dataset.dataset_in_memory_for_ddpm import get_datasets
from tta_uia_segmentation.src.train import OAICDDPMTrainer
from tta_uia_segmentation.src.utils.io import (
    load_config, rewrite_config_arguments, dump_config, print_config)
from tta_uia_segmentation.src.models.ConditionalGaussianDiffusionOAI import diffusion_defaults
from tta_uia_segmentation.src.models.UNetModelOAI import create_model_conditioned_on_seg_mask, model_defaults
#from tta_uia_segmentation.src.models.ConditionalGaussianDiffusionOAI import create_gaussian_diffusion
from tta_uia_segmentation.src.models.io import load_norm_from_configs_and_cpt
from tta_uia_segmentation.src.utils.logging import setup_wandb


def preprocess_cmd_args() -> argparse.Namespace:
    """_
    Parse command line arguments and return them as a Namespace object.
    
    All options specified will overwrite whaterver is specified in the config files.
    
    """
    parse_bool = lambda s: s.strip().lower() == 'true'
    parser = argparse.ArgumentParser(description="Train Segmentation Model (with shallow normalization module)")
    
    parser.add_argument('dataset_config_file', type=str, help='Path to yaml config file with parameters that define the dataset.')
    parser.add_argument('model_config_file', type=str, help='Path to yaml config file with parameters that define the model.')
    parser.add_argument('train_config_file', type=str, help='Path to yaml config file with parameters for training.')
    
    # Training parameters. If provided, overrides default parameters from config file.
    # :================================================================================================:
    parser.add_argument('--resume', type=parse_bool, 
                        help='Resume training from last checkpoint. Default: True.') 
    parser.add_argument('--logdir', type=str,  help='Path to directory where logs and checkpoints are saved. Default: logs')  
    parser.add_argument('--wandb_log', type=parse_bool, help='Log training to wandb. Default: False.')
    parser.add_argument('--wandb_project', type=str, help='Name of wandb project. Default: None')
    parser.add_argument('--start_new_exp', type=parse_bool, help='Start a new wandb experiment. Default: False')

    # Model parameters
    # ----------------:
    parser.add_argument('--num_channels', type=int, help='Number of channels in the first layer of the model. Default: 64')
    parser.add_argument('--channel_mult', type=int, nargs='+', help='multiplier for number of channels in each layer of the model. Default: 1 2 3 4')
    parser.add_argument('--num_heads', type=int, help='Number of heads in the attention layer. Default: 1')

    # Training loop
    # -------------:    
    parser.add_argument('--train_num_steps', type=int, help='Number of steps to train for. Default: 100000')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer. Default: 1e-4')
    parser.add_argument('--use_fp16', type=parse_bool, help='Use mixed precision training. Default: True')
    parser.add_argument('--batch_size', type=int, help='Batch size for training. Default: 4')
    parser.add_argument('--microbatch', type=int, help='Microbatch size for training. Default: -1')
    parser.add_argument('--num_workers', type=int, help='Number of workers for dataloader. Default: 0')
    parser.add_argument('--log_interval', type=int, help='Number of steps between logging. Default: 2')
    parser.add_argument('--save_interval', type=int, help='Number of steps between saving checkpoints. Default: 1000')
    
    # Diffusion model
    parser.add_argument('--learn_sigma', type=parse_bool, help='Whether to learn sigma. Default: False')
    parser.add_argument('--use_kl', type=parse_bool, help='Whether to use KL divergence. Default: False')
    parser.add_argument('--schedule_sampler', type=str, help='Schedule sampler for the loss. Default: uniform',
                        choices=['uniform', 'loss-second-moment'])
    parser.add_argument('--noise_schedule', type=str, help='Noise schedule for diffusion. Default: cosine',
                        choices=['linear', 'cosine'])
    
    # Dataset and its transformations to use for training
    # ---------------------------------------------------:
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for training')
    parser.add_argument('--n_classes', type=int, help='Number of classes in dataset.')
    parser.add_argument('--norm_dir', type=str, help='Path to directory where normalization model is saved.')
    parser.add_argument('--image_size', type=int, nargs='+', help='Size of images in dataset.')
    parser.add_argument('--resolution_proc', type=float, nargs='+', help='Resolution of images in dataset')
    parser.add_argument('--rescale_factor', type=float, help='Rescale factor for images in dataset')
    
    
    argument_names = [action.dest for action in parser._actions if action.dest != 'help']
    
    default_params = get_default_params_original_script()
    
    # Remove keys from default_params that are present in argument_names
    for key in argument_names:
        if key in default_params:
            del default_params[key]
    
    add_dict_to_argparser(parser, default_params)
    
    args = parser.parse_args()
    
    return args


def get_configuration_arguments(model_type, train_type) -> tuple[dict, dict, dict, argparse.Namespace]:
    args_ns_ = preprocess_cmd_args()
    
    dataset_config = load_config(args_ns_.dataset_config_file)
    dataset_config = rewrite_config_arguments(dataset_config, args_ns_, 'dataset')
    
    model_config = load_config(args_ns_.model_config_file)
    model_config = rewrite_config_arguments(model_config, args_ns_, 'model')
    
    model_config[model_type] = rewrite_config_arguments(
        model_config[model_type], args_ns_, f'model, {model_type}')

    train_config = load_config(args_ns_.train_config_file)
    train_config = rewrite_config_arguments(train_config, args_ns_, 'train')
    
    train_config[train_type] = rewrite_config_arguments(
        train_config[train_type], args_ns_, f'train, {train_type}')
    
    return dataset_config, model_config, train_config, args_ns_


def main():
    
    logger.log(f'Running {__file__}')
    train_type = 'ddpm_oai'
    model_type = 'ddpm_unet_oai'
    
    # Loading general parameters
    # :=========================================================================:
    dataset_config, model_config, train_config, args = get_configuration_arguments(
        model_type, train_type)
    
    resume          = train_config['resume']
    wandb_log       = train_config['wandb_log']
    start_new_exp   = train_config['start_new_exp']
    logdir          = train_config[train_type]['logdir']
    wandb_project   = train_config[train_type]['wandb_project']
    
    # Write or load parameters to/from logdir, used if a run is resumed.
    # :=========================================================================:
    is_resumed = os.path.exists(os.path.join(logdir, 'params.yaml')) and resume
    logger.log(f'training resumed: {is_resumed}')

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

        dump_config(os.path.join(logdir, 'params.yaml'), params)

    # Set the dir where things will be logged
    os.environ['DIFFUSION_BLOB_LOGDIR'] = logdir

    dist_util.setup_dist()
    logger.configure()
    
    # Setup wandb logging
    # :=========================================================================:
    if wandb_log and dist.get_rank() == 0:
        wandb_dir = setup_wandb(params, logdir, wandb_project, start_new_exp)
        
    print_config(params, keys=['training', 'model'])
    
    
    # Import dataset and create dataloader 
    # :=========================================================================:
    logger.log("creating data loader...")
    dataset             = train_config[train_type]['dataset']
    n_classes           = dataset_config[dataset]['n_classes']
    batch_size          = train_config[train_type]['batch_size']
    norm_dir            = train_config[train_type]['norm_dir']   
    norm_device         = train_config[train_type]['norm_device']
    cpt_type            = 'checkpoint_best' if train_config[train_type]['load_best_cpt'] else 'checkpoint_last'
    
    # Load normalization model used in the segmentation network
    if train_config[train_type]['norm_with_nn_on_fly']:
        norm = load_norm_from_configs_and_cpt(
            model_params_norm=model_params_norm,
            cpt_fp=os.path.join(norm_dir, train_config[cpt_type]),
            device=norm_device
        )
    else: 
        norm = None

    train_split         = train_config[train_type]['split']
    val_split           = train_config[train_type]['split_val']
    use_original_imgs   = train_config[train_type]['use_original_imgs']
    one_hot_encode      = train_config[train_type]['one_hot_encode']
    normalize           = train_config[train_type]['normalize']
    image_size          = train_config[train_type]['image_size']
    aug_params          = train_config[train_type]['augmentation']
    bg_suppression_opts = train_config[train_type]['bg_suppression_opts']

    paths               = dataset_config[dataset]['paths_processed']
    paths_original      = dataset_config[dataset]['paths_original']
    resolution_proc     = dataset_config[dataset]['resolution_proc']
    dim_proc            = dataset_config[dataset]['dim']
    
    (train_data, val_data) = get_datasets(
        splits = [train_split, val_split],
        norm = norm,
        norm_device = norm_device,
        norm_neg_one_to_one = True,
        paths = paths,
        paths_normalized_h5 = None, # dataset_config[dataset]['paths_normalized_with_nn'],
        use_original_imgs = use_original_imgs,
        one_hot_encode = one_hot_encode,
        normalize = normalize,
        paths_original = paths_original,
        image_size = image_size,
        resolution_proc = resolution_proc,
        dim_proc = dim_proc,
        n_classes = n_classes,
        aug_params = aug_params,
        bg_suppression_opts = bg_suppression_opts,
        deformation     = None,
        load_original   = False,
        shard = MPI.COMM_WORLD.Get_rank(),
        num_shards = MPI.COMM_WORLD.Get_size(),
    )
        
    # Create model and diffusion object
    # :=========================================================================:
    logger.log("creating model and diffusion...")

    image_channels      = dataset_config[dataset]['image_channels'] 

    num_channels        = model_config[model_type]['num_channels']
    channel_mult        = model_config[model_type]['channel_mult']
    num_res_blocks      = model_config[model_type]['num_res_blocks']
    num_heads           = model_config[model_type]['num_heads']
    
    image_size          = train_config[train_type]['image_size'][-1]
    learn_sigma         = train_config[train_type]['learn_sigma']
    seg_cond            = train_config[train_type]['seg_cond']
    dropout             = train_config[train_type]['dropout']
    
    model = create_model_conditioned_on_seg_mask(
        image_size = image_size,
        image_channels = image_channels,
        seg_cond = seg_cond,
        num_channels = num_channels,
        channel_mult = channel_mult,
        learn_sigma = learn_sigma,
        n_classes = n_classes,
        num_res_blocks = num_res_blocks,
        num_heads=num_heads,
        dropout = dropout,
        **args_to_dict(args, model_defaults().keys())
    )
    
    diffusion_steps     = train_config[train_type]['diffusion_steps']
    noise_schedule      = train_config[train_type]['noise_schedule']  
    use_kl              = train_config[train_type]['use_kl']
    timestep_respacing  = train_config[train_type]['timestep_respacing']

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        timestep_respacing=timestep_respacing,
        **args_to_dict(args, diffusion_defaults().keys()))
    model.to(dist_util.dev())
    
    # Train model
    # :=========================================================================:
    logger.log("training...")
    train_num_steps     = train_config[train_type]['train_num_steps']
    learning_rate       = float(train_config[train_type]['learning_rate'])
    use_fp16            = train_config[train_type]['use_fp16']
    microbatch          = train_config[train_type]['microbatch']
    num_workers = train_config[train_type]['num_workers']
    
    global_batch_size = dist.get_world_size() * batch_size
    logger.log(f"Effective batch size of {global_batch_size}")
    
    schedule_sampler    = train_config[train_type]['schedule_sampler']
    schedule_sampler    = create_named_schedule_sampler(schedule_sampler, diffusion)
    
    log_interval        = train_config[train_type]['log_interval']
    save_interval       = train_config[train_type]['save_interval']
    use_ddim            = train_config[train_type]['use_ddim']   
    num_samples_for_metrics = train_config[train_type]['num_samples_for_metrics']
    if wandb_log:
        wandb.watch([model], log='all')
    
    OAICDDPMTrainer(
        model=model,
        diffusion=diffusion,
        train_data=train_data,
        train_num_steps=train_num_steps,
        batch_size=batch_size,
        microbatch=microbatch,
        schedule_sampler=schedule_sampler,
        lr=learning_rate,
        use_fp16=use_fp16,
        num_workers=num_workers,
        log_interval=log_interval,
        save_interval=save_interval,
        val_data=val_data,
        use_ddim=use_ddim,
        num_samples_for_metrics=num_samples_for_metrics,
        wandb_log=wandb_log,
        ema_rate=args.ema_rate,
        resume_checkpoint=args.resume_checkpoint,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def get_default_params_original_script() -> dict:
    defaults = dict(
        data_dir="",
        weight_decay=0.0,
        lr_anneal_steps=0,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        resume_checkpoint="",
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_defaults())
    defaults.update(diffusion_defaults())
    return defaults


if __name__ == "__main__":
    main()
