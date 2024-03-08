"""
Train a diffusion model on images.
"""
import os
import sys
import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop

from torch.utils.data import DataLoader

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

from dataset.dataset_in_memory_for_ddpm import get_datasets
from tta_uia_segmentation.src.utils.io import (
    load_config, rewrite_config_arguments, dump_config, print_config)
from tta_uia_segmentation.src.models.io import load_norm_from_configs_and_cpt
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
    
    # Dataset and its transformations to use for training
    # ---------------------------------------------------:
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for training. Default: USZ')
    parser.add_argument('--n_classes', type=int, help='Number of classes in dataset. Default: 21')
    parser.add_argument('--image_size', type=int, nargs='+', help='Size of images in dataset. Default: [560, 640, 160]')
    parser.add_argument('--resolution_proc', type=float, nargs='+', help='Resolution of images in dataset. Default: [0.3, 0.3, 0.6]')
    parser.add_argument('--rescale_factor', type=float, help='Rescale factor for images in dataset. Default: None')
    
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
    
    train_config['ddpm'] = rewrite_config_arguments(
        train_config['ddpm'], args, 'train, ddpm')
    
    return dataset_config, model_config, train_config


def main():
    
    logger.log(f'Running {__file__}')
    train_type = 'ddpm_oai'
    
    # Loading general parameters
    # :=========================================================================:
    dataset_config, model_config, train_config = get_configuration_arguments()
    
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

    args = create_argparser().parse_args()
    
    dist_util.setup_dist()
    logger.configure()

    # Create model and diffusion object
    # :=========================================================================:
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # Import dataset and create dataloader 
    logger.log("creating data loader...")
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

    data = get_datasets(
        splits          = [train_config[train_type]['split']],
        norm            = norm,
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
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            drop_last=True
        )
    
    data = cycle(data)
    
    # Train model
    # :=========================================================================:
    logger.log("training...")
    learning_rate = float(train_config[train_type]['learning_rate'])
    
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=learning_rate,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
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
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def cycle(dl):
    while True:
        for data in dl:
            yield data


if __name__ == "__main__":
    main()
