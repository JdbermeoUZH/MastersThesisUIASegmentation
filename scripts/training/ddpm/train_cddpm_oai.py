"""
Train a diffusion model on images.
"""
import os
import sys
import argparse
from itertools import chain

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_gaussian_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop

from torch.utils.data import DataLoader

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

from tta_uia_segmentation.src.dataset.dataset_in_memory_for_ddpm import get_datasets
from tta_uia_segmentation.src.train import OAICDDPMTrainer
from tta_uia_segmentation.src.utils.io import (
    load_config, rewrite_config_arguments, dump_config, print_config)
from tta_uia_segmentation.src.models.UNetModelOAI import create_model_conditioned_on_seg_mask
from tta_uia_segmentation.src.models.ConditionalGaussianDiffusionOAI import create_gaussian_diffusion
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
    parser.add_argument('--resume', type=lambda s: s.strip().lower() == 'true', 
                        help='Resume training from last checkpoint. Default: True.') 
    parser.add_argument('--logdir', type=str, 
                        help='Path to directory where logs and checkpoints are saved. Default: logs')  
    parser.add_argument('--wandb_log', type=lambda s: s.strip().lower() == 'true', 
                        help='Log training to wandb. Default: False.')

    # Model parameters
    # ----------------:
    #parser.add_argument('--channel_size', type=int, nargs='+', help='Number of feature maps for each block. Default: [16, 32, 64]')
    #parser.add_argument('--channels_bottleneck', type=int, help='Number of channels in bottleneck layer of model. Default: 128')
    
    # Training loop
    # -------------:    
    parser.add_argument('--epochs', type=int, help='Number of epochs to train. Default: 100')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer. Default: 1e-4')
    parser.add_argument('--batch_size', type=int, help='Batch size for training. Default: 4')
    parser.add_argument('--num_workers', type=int, help='Number of workers for dataloader. Default: 0')
    
    # Diffusion model
    parser.add_argument('--learn_sigma', type=parse_bool, help='Whether to learn sigma. Default: False')
    
    # Dataset and its transformations to use for training
    # ---------------------------------------------------:
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for training. Default: USZ')
    parser.add_argument('--n_classes', type=int, help='Number of classes in dataset. Default: 21')
    parser.add_argument('--image_size', type=int, nargs='+', help='Size of images in dataset. Default: [560, 640, 160]')
    parser.add_argument('--resolution_proc', type=float, nargs='+', help='Resolution of images in dataset. Default: [0.3, 0.3, 0.6]')
    parser.add_argument('--rescale_factor', type=float, help='Rescale factor for images in dataset. Default: None')
    
    argument_names = [action.dest for action in parser._actions if action.dest != 'help']
    
    default_params = get_default_params_original_script()
    
    # Remove keys from default_params that are present in argument_names
    for key in argument_names:
        if key in default_params:
            del default_params[key]
    
    add_dict_to_argparser(parser, default_params)
    
    args = parser.parse_args()
    
    return args


def get_configuration_arguments() -> tuple[dict, dict, dict, argparse.Namespace]:
    args_ns_ = preprocess_cmd_args()
    
    dataset_config = load_config(args_ns_.dataset_config_file)
    dataset_config = rewrite_config_arguments(dataset_config, args_ns_, 'dataset')
    
    model_config = load_config(args_ns_.model_config_file)
    model_config = rewrite_config_arguments(model_config, args_ns_, 'model')

    train_config = load_config(args_ns_.train_config_file)
    train_config = rewrite_config_arguments(train_config, args_ns_, 'train')
    
    train_config['ddpm'] = rewrite_config_arguments(
        train_config['ddpm'], args_ns_, 'train, ddpm')
    
    return dataset_config, model_config, train_config, args_ns_


def main():
    
    logger.log(f'Running {__file__}')
    train_type = 'ddpm_oai'
    model_type = 'ddpm_unet_oai'
    
    # Loading general parameters
    # :=========================================================================:
    dataset_config, model_config, train_config, args = get_configuration_arguments()
    
    resume          = train_config['resume']
    wandb_log       = train_config['wandb_log']
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

    print_config(params, keys=['training', 'model'])

    dist_util.setup_dist()
    logger.configure()
    
    # Import dataset and create dataloader 
    # :=========================================================================:
    logger.log("creating data loader...")
    dataset             = train_config[train_type]['dataset']
    n_classes           = dataset_config[dataset]['n_classes']
    batch_size          = train_config[train_type]['batch_size']
    norm_dir            = train_config[train_type]['norm_dir']   
    norm_device         = train_config[train_type]['norm_device']
    
    # Load normalization model used in the segmentation network
    if train_config[train_type]['norm_with_nn_on_fly']:
        norm = load_norm_from_configs_and_cpt(
            model_params_norm=model_params_norm,
            cpt_fp=os.path.join(norm_dir, train_params_norm['checkpoint_best']),
            device=norm_device
        )
    else: 
        norm = None

    (data,) = get_datasets(
        splits          = [train_config[train_type]['split']],
        norm            = norm,
        norm_device     = norm_device,  
        paths           = dataset_config[dataset]['paths_processed'],
        paths_normalized_h5 = dataset_config[dataset]['paths_normalized_with_nn'],
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
    
    num_workers = train_config[train_type]['num_workers']
    data = DataLoader(
            data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            drop_last=True
        )
    
    data = cycle(data)

    # Create model and diffusion object
    # :=========================================================================:
    logger.log("creating model and diffusion...")

    image_channels      = dataset_config[dataset]['image_channels'] 

    num_channels        = model_config[model_type]['num_channels']
    channel_mult        = model_config[model_type]['channel_mult']

    image_size          = train_config[train_type]['image_size'][-1]
    learn_sigma         = train_config[train_type]['learn_sigma']
    seg_cond            = train_config[train_type]['seg_cond']
    
    # args_not_to_use = ['image_channels', 'n_classes', 
    #                    'num_channels', 'channel_mult',
    #                    'image_size', 'learn_sigma', 'seg_cond']
    # args_to_use = list(model_and_diffusion_defaults().keys())
    # args_to_use = [arg for arg in args_to_use if arg not in args_not_to_use]
    
    model = create_model_conditioned_on_seg_mask(
        image_size = image_size,
        image_channels = image_channels,
        seg_cond = seg_cond,
        num_channels = num_channels,
        channel_mult = channel_mult,
        learn_sigma = learn_sigma,
        n_classes = n_classes,
        **args_to_dict(args, model_defaults().keys())
    )
    
    diffusion_steps     = train_config[train_type]['diffusion_steps']
    noise_schedule      = train_config[train_type]['noise_schedule']  
    use_kl              = train_config[train_type]['use_kl']
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        **args_to_dict(args, diffusion_defaults().keys()))
    model.to(dist_util.dev())
    
    # Train model
    # :=========================================================================:
    logger.log("training...")
    learning_rate = float(train_config[train_type]['learning_rate'])
    schedule_sampler = train_config[train_type]['schedule_sampler']
    schedule_sampler = create_named_schedule_sampler(schedule_sampler, diffusion)
    
    OAICDDPMTrainer(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        schedule_sampler=schedule_sampler,
        microbatch=args.microbatch,
        lr=learning_rate,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def get_default_params_original_script() -> dict:
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_defaults())
    defaults.update(diffusion_defaults())
    return defaults


def cycle(dl):
    while True:
        for data in dl:
            yield data


def model_defaults():
    """
    Defaults for image training.
    """
    return dict(
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )


def diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        sigma_small=False,
        timestep_respacing="",
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
    )

if __name__ == "__main__":
    main()
