import os
import sys
import argparse

import wandb
from torch.utils.data import DataLoader
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'tta_uia_segmentation', 'src')))

from dataset import DatasetInMemoryForDDPM
from models import ConditionalGaussianDiffusion
from train import CDDPMTrainer
from utils.io import (load_config, dump_config, print_config,
                      save_checkpoint, write_to_csv, rewrite_config_arguments)
from utils.utils import seed_everything, define_device
from utils.logging import setup_wandb



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
    parser.add_argument('--channel_size', type=int, nargs='+', help='Number of feature maps for each block. Default: [16, 32, 64]')
    parser.add_argument('--channels_bottleneck', type=int, help='Number of channels in bottleneck layer of model. Default: 128')
    parser.add_argument('--skips', type=lambda s: [val.strip().lower() == 'true' for val in s.split()], 
                        help='Whether to use skip connections on each block, specified as a space-separated list of booleans (True or False)'
                        'Default: True True True')
    parser.add_argument('--n_dimensions', type=int, help='Number of dimensions of the model, i.e: 1D, 2D, or 3D. Default: 3')  
    
    # Training loop
    # -------------:
    parser.add_argument('--epochs', type=int, help='Number of epochs to train. Default: 100')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer. Default: 1e-4')
    parser.add_argument('--batch_size', type=int, help='Batch size for training. Default: 4')
    parser.add_argument('--num_workers', type=int, help='Number of workers for dataloader. Default: 0')
    parser.add_argument('--validate_every', type=int, help='Validate every n epochs. Default: 1')
    parser.add_argument('--seed', type=int, help='Seed for random number generators. Default: 0')   
    parser.add_argument('--device', type=str, help='Device to use for training. Default cuda', )
    parser.add_argument('--checkpoint_last', type=str, help='Name of last checkpoint file. Default: checkpoint_last.pth')
    parser.add_argument('--checkpoint_best', type=str, help='Name of best checkpoint file. Default: checkpoint_best.pth')
    
    # Dataset and its transformations to use for training
    # ---------------------------------------------------:
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for training. Default: USZ')
    parser.add_argument('--n_classes', type=int, help='Number of classes in dataset. Default: 21')
    parser.add_argument('--image_size', type=int, nargs='+', help='Size of images in dataset. Default: [560, 640, 160]')
    parser.add_argument('--resolution_proc', type=float, nargs='+', help='Resolution of images in dataset. Default: [0.3, 0.3, 0.6]')
    parser.add_argument('--rescale_factor', type=float, help='Rescale factor for images in dataset. Default: None')
    
    parser.add_argument('--concatenate', type=lambda s: s.strip().lower() == 'true', 
                        help='Whether to concatenate images and labels. Default: True')
    parser.add_argument('--axis_to_concatenate', type=int, help='Axis to concatenate images and labels. Default: 0')
    
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



if __name__ == '__main__':

    print(f'Running {__file__}')
    train_type = 'ddpm'
    # Loading general parameters
    # :=========================================================================:
    dataset_config, model_config, train_config = get_configuration_arguments()
    
    params          = {'datset': dataset_config, 'model': model_config,
                       'training': train_config}
    resume          = train_config['resume']
    seed            = train_config['seed']
    device          = train_config['device']
    wandb_log       = train_config['wandb_log']
    logdir          = train_config[train_type]['logdir']
    wandb_project   = train_config[train_type]['wandb_project']
    
    # Write or load parameters to/from logdir, used if a run is resumed.
    # :=========================================================================:
    is_resumed = os.path.exists(os.path.join(logdir, 'params.yaml')) and resume
    print(f'training resumed: {is_resumed}')

    if is_resumed:
        params = load_config(os.path.join(logdir, 'params.yaml'))
        
    else:
        os.makedirs(logdir, exist_ok=True)
        dump_config(os.path.join(logdir, 'params.yaml'), params)

    print_config(params, keys=['training', 'model'])

    # Setup wandb logging
    # :=========================================================================:
    if wandb_log:
        wandb_dir = setup_wandb(params, logdir, wandb_project)
    
    # Define the dataset that is to be used for training
    # :=========================================================================:
    print('Defining dataset')
    seed_everything(seed)
    device              = define_device(device)
    dataset             = train_config[train_type]['dataset']
    n_classes           = dataset_config[dataset]['n_classes']
    batch_size          = train_config[train_type]['batch_size']
    #num_workers         = train_config[train_type]['num_workers']
    
    # Dataset definition
    train_dataset = DatasetInMemoryForDDPM(
        concatenate     = train_config[train_type]['concatenate'],
        axis_to_concatenate = train_config[train_type]['axis_to_concatenate'],
        paths           = dataset_config[dataset]['paths_processed'],
        paths_original  = dataset_config[dataset]['paths_original'],
        split          = 'train',
        image_size      = train_config[train_type]['image_size'],
        resolution_proc = dataset_config[dataset]['resolution_proc'],
        dim_proc        = dataset_config[dataset]['dim'],
        n_classes       = n_classes,
        aug_params      = train_config[train_type]['augmentation'],
        bg_suppression_opts = train_config[train_type]['bg_suppression_opts'],
        deformation     = None,
        load_original   = False,
    )
    print('Dataloaders defined')
    
    # Define the denoiser model diffusion pipeline
    # :=========================================================================:
    
    print(f'Using Device {device}')
    # Model definition
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True,
        channels= 1, 
        self_condition=True,
    )#.to(device)
    
    diffusion = ConditionalGaussianDiffusion(
        model,
        image_size = 256,
        timesteps = 1000    # number of steps
    )#.to(device)

    # Execute the training loop
    # :=========================================================================:
    print('Defining trainer: training loop, optimizer and loss')
    batch_size = train_config[train_type]['batch_size']
    gradient_accumulate_every = train_config[train_type]['gradient_accumulate_every']
    save_and_sample_every = train_config[train_type]['save_and_sample_every']
    train_lr = float(train_config[train_type]['learning_rate'])
    train_num_steps = train_config[train_type]['num_steps']
    save_and_sample_every = train_config[train_type]['save_and_sample_every']
    
    trainer = CDDPMTrainer(
            diffusion,
            train_dataset,
            train_batch_size = batch_size,
            train_lr = train_lr,
            train_num_steps = train_num_steps,# total training steps
            gradient_accumulate_every = gradient_accumulate_every,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            amp = True,                       # turn on mixed precision
            calculate_fid = False,            # whether to calculate fid during training 
            results_folder=logdir,
            save_and_sample_every = save_and_sample_every,
            )
    
    if wandb_log:
        #wandb.save(os.path.join(wandb_dir, trainer.get_last_checkpoint_name()), base_path=wandb_dir)
        #wandb.watch([diffusion, model], trainer.get_loss_function(), log='all')
        wandb.watch([diffusion, model], log='all')
        
    # Start training
    # :=========================================================================:
    trainer.train()
    
    # write_to_csv(
    #     path=os.path.join(logdir, 'training_statistics.csv'),
    #     data=np.stack([trainer.get_training_losses(),
    #                    trainer.get_validation_losses(validate_every=validate_every),
    #                    trainer.get_validation_scores(validate_every=validate_every)], 1),
    #     header=['training_losses', 'validation_losses', 'validation_scores'],
    # )
    
    if wandb_log:
        wandb.finish()
