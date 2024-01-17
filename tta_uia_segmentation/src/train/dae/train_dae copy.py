import os
import argparse

import wandb
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

from dataset.dataset import get_datasets
from models import UNet
from train.dae.DAETrainer import DAETrainer
from utils.loss import DiceLoss
from utils.io import load_config, rewrite_config_arguments, dump_config, print_config, save_checkpoint, write_to_csv, deep_get
from utils.utils import seed_everything, define_device
from utils.logging import setup_wandb


def preprocess_cmd_args() -> argparse.Namespace:
    """_
    Parse command line arguments and return them as a Namespace object.
    
    All options specified will overwrite whaterver is specified in the config files.
    
    """
    parser = argparse.ArgumentParser(description="Train DAE")
    
    parser.add_argument('model_config_file', type=str, help='Path to yaml config file with parameters that define the model.')
    parser.add_argument('train_config_file', type=str, help='Path to yaml config file with parameters for training.')
    
    # Training parameters. If provided, overrides default parameters from config file.
    # :================================================================================================:
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from last checkpoint. Default: True.') 
    parser.add_argument('--logdir', type=str, 
                        help='Path to directory where logs and checkpoints are saved. Default: logs')  
    parser.add_argument('--wandb_log', action='store_true', 
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
    parser.add_argument('--seed', type=int, help='Seed for random number generators.', default=0)   
    parser.add_argument('--device', type=str, help='Device to use for training.', default='gpu')
    parser.add_argument('--checkpoint_last', type=str, help='Name of last checkpoint file. Default: checkpoint_last.pth')
    parser.add_argument('--checkpoint_best', type=str, help='Name of best checkpoint file. Default: checkpoint_best.pth')
    
    # Dataset and its transformations to use for training
    # ---------------------------------------------------:
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for training. Default: USZ')
    parser.add_argument('--n_classes', type=int, help='Number of classes in dataset. Default: 21')
    parser.add_argument('--image_size', type=int, nargs='+', help='Size of images in dataset. Default: [560, 640, 160]')
    parser.add_argument('--resolution_proc', type=float, nargs='+', help='Resolution of images in dataset. Default: [0.3, 0.3, 0.6]')
    parser.add_argument('--rescale_factor', type=float, help='Rescale factor for images in dataset. Default: None')
    
    # Augmentations
    parser.add_argument('--da_ratio', type=float, help='Ratio of images to apply DA to. Default: 0.25')
    parser.add_argument('sigma', type=float, help='augmentation. Default: 20') #TODO: specify what this is
    parser.add_argument('alpha', type=float, help='augmentation. Default: 1000') #TODO: specify what this is
    parser.add_argument('trans_min', type=float, help='Minimum value for translation augmentation. Default: -10')
    parser.add_argument('trans_max', type=float, help='Maximum value for translation augmentation. Default: 10')
    parser.add_argument('rot_min', type=float, help='Minimum value for rotation augmentation. Default: -10')
    parser.add_argument('rot_max', type=float, help='Maximum value for rotation augmentation. Default: 10') 
    parser.add_argument('scale_min', type=float, help='Minimum value for zooming augmentation. Default: 0.9') 
    parser.add_argument('scale_max', type=float, help='Maximum value for zooming augmentation. Default: 1.1')
    parser.add_argument('gamma_min', type=float, help=' augmentation. Default: 1.0') #TODO: specify what this is
    parser.add_argument('gamma_max', type=float, help=' augmentation. Default: 1.0') #TODO: specify what this is
    parser.add_argument('brightness_min', type=float, help='Minimum value for brightness augmentation. Default: 0.0') 
    parser.add_argument('brightness_max', type=float, help='Maximum value for brightness augmentation. Default: 0.0')   
    parser.add_argument('noise_mean', type=float, help='Mean value for noise augmentation. Default: 0.0')
    parser.add_argument('noise_std', type=float, help='Standard deviation value for noise augmentation. Default: 0.0') 
    
    # Deformation or Corruptions
    parser.add_argument('--mask_type', type=str, help='Type of mask to use for deformation. Default: squares_jigsaw',
                        choices=['squares_jigsaw', 'zeros', 'jigsaw', 'random_labels'])
    parser.add_argument('--mask_radius', type=int, help='Radius of mask for deformation. Default: 10') 
    parser.add_argument('--mask_squares', type=int, help='Number of squares for mask. Default: 200') 
    parser.add_argument('--is_num_masks_fixed', type=lambda s: s.strip().lower() == 'true', help='Whether to use jigsaw mask. Default: False')
    parser.add_argument('-is_size_masks_fixed', type=lambda s: s.strip().lower() == 'true', help='Whether to use jigsaw mask. Default: False')
    
    args = parser.parse_args()
    
    return args


def get_configuration_arguments() -> tuple[dict, dict]:
    args = preprocess_cmd_args()
    
    model_config = load_config(args.model_config_file)
    model_config = rewrite_config_arguments(model_config, args, 'model')

    train_config = load_config(args.train_config_file)
    train_config = rewrite_config_arguments(train_config, args, 'train')
    
    train_config['dae'] = rewrite_config_arguments(
        train_config['dae'], args, 'train, dae')
    
    train_config['dae']['augmentation'] = rewrite_config_arguments(
        train_config['dae']['augmentation'], args, 'train, dae, augmentation')
    
    train_config['dae']['deformation'] = rewrite_config_arguments(
        train_config['dae']['deformation'], args, 'train, dae, deformation')
    
    return model_config, train_config


def save_atlas(train_dataset, num_workers, logdir):
    train_dataset.set_augmentation(False)
    atlas_dataloader = DataLoader(dataset=train_dataset, batch_size=1,
        shuffle=False, num_workers=num_workers, drop_last=False)
    
    atlas = None
    for _, y, _, _, _ in atlas_dataloader:
        if atlas is None:
            atlas = y.clone()
        else:
            atlas += y

    atlas = atlas.float()
    atlas /= len(atlas_dataloader)

    save_checkpoint(
        path=os.path.join(logdir, 'atlas.h5py'),
        atlas=atlas,
    )


if __name__ == '__main__':

    print(f'Running {__file__}')
    
    # Loading general parameters
    # :=========================================================================:
    model_config, train_config = get_configuration_arguments()
    
    params          = {'model': model_config, 'training': train_config}
    resume          = train_config['resume']
    seed            = train_config['seed']
    device          = train_config['device']
    logdir          = train_config['logdir']
    wandb_log       = train_config['wandb_log']
    wandb_project   = train_config['wandb_project']
    
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
    device      = define_device(device)
    dataset     = train_config['dae']['dataset']
    n_classes   = train_config['dae']['n_classes']
    batch_size      = deep_get(params, 'training', 'dae', 'batch_size')
    num_workers     = deep_get(params, 'training', 'dae', 'num_workers', default=0)
    
    # Dataset definition
    train_dataset, val_dataset = get_datasets(
        paths           = deep_get(params, 'datasets', dataset, 'paths_processed'),
        paths_original  = deep_get(params, 'datasets', dataset, 'paths_original'),
        splits          = ['train', 'val'],
        image_size      = deep_get(params, 'training', 'dae', 'image_size'),
        resolution_proc = deep_get(params, 'datasets', dataset, 'resolution_proc'),
        rescale_factor  = deep_get(params, 'training', 'dae', 'rescale_factor', default=None),
        dim_proc        = deep_get(params, 'datasets', dataset, 'dim'),
        n_classes       = n_classes,
        aug_params      = deep_get(params, 'training', 'dae', 'augmentation', default=None),
        deformation     = deep_get(params, 'training', 'dae', 'deformation', default=None),
        load_original   = False,
    )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True)
    
    val_dataset = ConcatDataset([val_dataset] * train_config['dae']['validate_every'])
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, drop_last=False)

    print('Dataloaders defined')

    
    # Define DAE model that is to be trained
    # :=========================================================================:
    print('Defining model')
    dae = UNet(
        in_channels             = n_classes,
        n_classes               = n_classes,
        channels                = model_config['channel_size'],
        channels_bottleneck     = model_config['channels_bottleneck'],
        skips                   = model_config['skips'],
        n_dimensions            = model_config['n_dimensions']
    ).to(device)

    # Define the Trainer that will be used to train the model
    # :=========================================================================:
    print('Defining trainer: training loop, optimizer and loss')
    dae_trainer = DAETrainer(
        model                   = dae,
        learning_rate           = train_config['dae']['learning_rate'],
        device                  = device,
        loss_func               = DiceLoss(),
        is_resumed              = is_resumed,
        checkpoint_last         = train_config['checkpoint_last'],
        checkpoint_best         = train_config['checkpoint_best'],
        logdir                  = logdir,
        device                  = device,
        wandb_log               = wandb_log,
    )
    
    if wandb_log:
        wandb.save(os.path.join(wandb_dir, dae_trainer.get_last_checkpoint_name()),
                   base_path=wandb_dir)
        wandb.watch([dae], dae_trainer.get_loss_function(), log='all')

   
    # Start training
    # :=========================================================================:
    dae_trainer.train(
        train_dataloader        = train_dataloader,
        val_dataloader          = val_dataloader,
        epochs                  = train_config['dae']['epochs'],
        validate_every          = train_config['dae']['validate_every']
    )
    
    # Save/Log results
    # :=========================================================================:
    save_atlas(train_dataset, num_workers, logdir)

    write_to_csv(
        path=os.path.join(logdir, 'training_statistics.csv'),
        data=np.stack([dae_trainer.get_training_losses()], 1),
        header=['training_losses'],
    )
    
    write_to_csv(
        path=os.path.join(logdir, 'validation_statistics.csv'),
        data=np.stack([dae_trainer.get_validation_losses(),
                       dae_trainer.get_validation_scores()], 1),
        header=['validation_losses', 'validation_scores'],
    )

    if wandb_log:
        wandb.finish()
