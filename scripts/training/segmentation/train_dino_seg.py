import os
import sys
import argparse

import wandb
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..')))

from tta_uia_segmentation.src.dataset.dataset_in_memory import get_datasets
from tta_uia_segmentation.src.models.io import define_and_possibly_load_dino_seg
from tta_uia_segmentation.src.utils.loss import DiceLoss
from tta_uia_segmentation.src.utils.io import (
    load_config, dump_config, print_config, 
    write_to_csv, rewrite_config_arguments)
from tta_uia_segmentation.src.utils.utils import seed_everything, define_device, parse_bool
from tta_uia_segmentation.src.utils.logging import setup_wandb


train_type = 'segmentation_dino'


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
    parser.add_argument('--resume', type=parse_bool, 
                        help='Resume training from last checkpoint. Default: True.') 
    parser.add_argument('--logdir', type=str, 
                        help='Path to directory where logs and checkpoints are saved. Default: logs')  
    parser.add_argument('--wandb_log', type=parse_bool, 
                        help='Log training to wandb. Default: False.')

    # Model parameters
    # ----------------:
    parser.add_argument('--dino_model', type=str, help='Name of DINO model to use. Default: "large"')
    parser.add_argument('--svd_path', type=str, help='Path to SVD model. Default: None')

    # Training loop
    # -------------:
    parser.add_argument('--epochs', type=int, help='Number of epochs to train. Default: 100')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer. Default: 1e-4')
    parser.add_argument('--batch_size', type=int, help='Batch size for training. Default: 4')
    parser.add_argument('--gradient_accumulation', type=int, help='Number of batches to accumulate gradients over. Default: 1')
    parser.add_argument('--num_workers', type=int, help='Number of workers for dataloader. Default: 0')
    parser.add_argument('--validate_every', type=int, help='Validate every n epochs. Default: 1')
    parser.add_argument('--seed', type=int, help='Seed for random number generators. Default: 0')   
    parser.add_argument('--device', type=str, help='Device to use for training. Default cuda', )
    parser.add_argument('--checkpoint_last', type=str, help='Name of last checkpoint file. Default: checkpoint_last.pth')
    parser.add_argument('--checkpoint_best', type=str, help='Name of best checkpoint file. Default: checkpoint_best.pth')
    
    # Loss function
    parser.add_argument('--smooth', type=float, help='Smoothing factor for dice loss. Default: 1e-5')
    parser.add_argument('--epsilon', type=float, help='Epsilon factor for dice loss. Default: 1e-10')
    parser.add_argument('--fg_only', type=parse_bool, help='Whether to calculate dice loss only on foreground. Default: True')
    parser.add_argument('--debug_mode', type=parse_bool, help='Whether to run in debug mode. Default: False')

    # Dataset and its transformations to use for training
    # ---------------------------------------------------:
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for training. Default: USZ')
    parser.add_argument('--n_classes', type=int, help='Number of classes in dataset. Default: 21')
    parser.add_argument('--image_size', type=int, nargs='+', help='Size of images in dataset. Default: [560, 640, 160]')
    parser.add_argument('--resolution_proc', type=float, nargs='+', help='Resolution of images in dataset. Default: [0.3, 0.3, 0.6]')
    parser.add_argument('--rescale_factor', type=float, help='Rescale factor for images in dataset. Default: None')
    
    # Augmentations
    parser.add_argument('--da_ratio', type=float, help='Ratio of images to apply DA to. Default: 0.25')
    parser.add_argument('--sigma', type=float, help='augmentation. Default: 20') 
    parser.add_argument('--alpha', type=float, help='augmentation. Default: 1000') 
    parser.add_argument('--trans_min', type=float, help='Minimum value for translation augmentation. Default: -10')
    parser.add_argument('--trans_max', type=float, help='Maximum value for translation augmentation. Default: 10')
    parser.add_argument('--rot_min', type=float, help='Minimum value for rotation augmentation. Default: -10')
    parser.add_argument('--rot_max', type=float, help='Maximum value for rotation augmentation. Default: 10') 
    parser.add_argument('--scale_min', type=float, help='Minimum value for zooming augmentation. Default: 0.9') 
    parser.add_argument('--scale_max', type=float, help='Maximum value for zooming augmentation. Default: 1.1')
    parser.add_argument('--gamma_min', type=float, help=' augmentation. Default: 1.0') 
    parser.add_argument('--gamma_max', type=float, help=' augmentation. Default: 1.0') 
    parser.add_argument('--brightness_min', type=float, help='Minimum value for brightness augmentation. Default: 0.0') 
    parser.add_argument('--brightness_max', type=float, help='Maximum value for brightness augmentation. Default: 0.0')   
    parser.add_argument('--noise_mean', type=float, help='Mean value for noise augmentation. Default: 0.0')
    parser.add_argument('--noise_std', type=float, help='Standard deviation value for noise augmentation. Default: 0.0') 
    
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
    
    train_config[train_type] = rewrite_config_arguments(
        train_config[train_type], args, f'train, {train_type}')
    
    train_config[train_type]['augmentation'] = rewrite_config_arguments(
        train_config[train_type]['augmentation'], args, f'train, {train_type}, augmentation')
    
    train_config[train_type]['bg_suppression_opts'] = rewrite_config_arguments(
        train_config[train_type]['bg_suppression_opts'], args, f'train, {train_type}, bg_suppression_opts',
        prefix_to_remove='bg_supression_')
    
    return dataset_config, model_config, train_config



if __name__ == '__main__':

    print(f'Running {__file__}')
    
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
    wandb_dir = setup_wandb(params, logdir, wandb_project) if wandb_log else None
    
    # Define the dataset that is to be used for training
    # :=========================================================================:
    print('Defining dataset')
    seed_everything(seed)
    device              = define_device(device)
    dataset_name        = train_config[train_type]['dataset']
    n_classes           = dataset_config[dataset_name]['n_classes']
    batch_size          = train_config[train_type]['batch_size']
    num_workers         = train_config[train_type]['num_workers']
    bg_suppression_opts = train_config[train_type]['bg_suppression_opts']

    # Dataset definition
    train_dataset, val_dataset = get_datasets(
        dataset_name    = dataset_name,
        paths           = dataset_config[dataset_name]['paths_processed'],
        paths_original  = dataset_config[dataset_name]['paths_original'],
        splits          = ['train', 'val'],
        image_size      = train_config[train_type]['image_size'],
        resolution_proc = dataset_config[dataset_name]['resolution_proc'],
        dim_proc        = dataset_config[dataset_name]['dim'],
        n_classes       = n_classes,
        aug_params      = train_config[train_type]['augmentation'],
        deformation     = None,
        load_original   = False,
        bg_suppression_opts = bg_suppression_opts
    )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True)    
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, drop_last=False)

    print('Dataloaders defined')
    
    # Define the 2D segmentation model
    # :=========================================================================:
    dino_seg = define_and_possibly_load_dino_seg(
        train_dino_seg_cfg = train_config[train_type],
        n_classes       = n_classes,
        cpt_fp          = train_config[train_type]['cpt_fp'],
        device          = device,
    )

    # Define the Trainer that will be used to train the model
    # :=========================================================================:
    print('Defining trainer: training loop, optimizer and loss')

    dice_loss = DiceLoss(
        smooth=train_config[train_type]['smooth'],
        epsilon=train_config[train_type]['epsilon'],
        debug_mode=train_config[train_type]['debug_mode'],
        fg_only=train_config[train_type]['fg_only']
    )

    trainer = NormSegTrainer(
        norm                = norm,
        seg                 = seg,
        learning_rate       = train_config[train_type]['learning_rate'],
        loss_func           = dice_loss,
        is_resumed          = is_resumed,
        checkpoint_best     = train_config['checkpoint_best'],
        checkpoint_last     = train_config['checkpoint_last'],
        device              = device,
        logdir              = logdir,
        wandb_log           = wandb_log,
        wandb_dir           = wandb_dir,
        bg_suppression_opts = bg_suppression_opts,
        with_bg_supression  = train_config[train_type]['with_bg_supression']
    )

    if wandb_log:
        wandb.save(os.path.join(wandb_dir, trainer.get_last_checkpoint_name()), base_path=wandb_dir)
        wandb.watch([dino_seg.decoder], trainer.get_loss_function(), log='all')
        
    # Start training
    # :=========================================================================:
    epochs                  = train_config[train_type]['epochs']
    validate_every          = train_config[train_type]['validate_every']

    if trainer.with_bg_supression:
        print(f'Using background suppression with {trainer.bg_suppression_opts["type"]} type')
    
    trainer.train(
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        epochs = epochs,
        validate_every = validate_every,
    )
    
    write_to_csv(
        path=os.path.join(logdir, 'training_statistics.csv'),
        data=np.stack([trainer.get_training_losses(),
                       trainer.get_validation_losses(validate_every=validate_every),
                       trainer.get_validation_scores(validate_every=validate_every)], 1),
        header=['training_losses', 'validation_losses', 'validation_scores'],
    )
    
    if wandb_log:
        wandb.finish()
