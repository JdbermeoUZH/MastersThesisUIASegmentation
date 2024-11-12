import os
import sys
import shutil
import argparse

import h5py
import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'tta_uia_segmentation', 'src')))

from dataset.dataset_in_memory import get_datasets
from models import Normalization 
from utils.io import (
    load_config, dump_config, print_config, rewrite_config_arguments)
from utils.utils import seed_everything, define_device



def preprocess_cmd_args() -> argparse.Namespace:
    """_
    Parse command line arguments and return them as a Namespace object.
    
    All options specified will overwrite whaterver is specified in the config files.
    
    """
    parser = argparse.ArgumentParser(description="Train Segmentation Model (with shallow normalization module)")
    
    parser.add_argument('dataset_config_file', type=str, help='Path to yaml config file with parameters that define the dataset.')
    parser.add_argument('preprocessing_config_file', type=str, help='Path to yaml config file with parameters to preprocess the dataset.')
    
    parser.add_argument('--logdir', type=str, help='Path to directory where logs and checkpoints are saved. Default: logs')  
    
    # Model parameters
    # ----------------:
    parser.add_argument('--channel_size', type=int, nargs='+', help='Number of feature maps for each block. Default: [16, 32, 64]')
    parser.add_argument('--channels_bottleneck', type=int, help='Number of channels in bottleneck layer of model. Default: 128')
    parser.add_argument('--skips', type=lambda s: [val.strip().lower() == 'true' for val in s.split()], 
                        help='Whether to use skip connections on each block, specified as a space-separated list of booleans (True or False)'
                        'Default: True True True')
    parser.add_argument('--n_dimensions', type=int, help='Number of dimensions of the model, i.e: 1D, 2D, or 3D. Default: 3')  
    parser.add_argument('--checkpoint', type=str, help='Name of last checkpoint file. Default: checkpoint_last.pth')
    
    # Evaluation loop
    # -------------:
    parser.add_argument('--batch_size', type=int, help='Batch size for training. Default: 4')
    parser.add_argument('--num_workers', type=int, help='Number of workers for dataloader. Default: 0')
    parser.add_argument('--device', type=str, help='Device to use for training. Default cuda', )

    
    # Dataset and its transformations to use for training
    # ---------------------------------------------------:
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for training. Default: USZ')
    parser.add_argument('--n_classes', type=int, help='Number of classes in dataset. Default: 21')
    parser.add_argument('--image_size', type=int, nargs='+', help='Size of images in dataset. Default: [560, 640, 160]')
    parser.add_argument('--resolution_proc', type=float, nargs='+', help='Resolution of images in dataset. Default: [0.3, 0.3, 0.6]')
    parser.add_argument('--rescale_factor', type=float, help='Rescale factor for images in dataset. Default: None')
    
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
    
    preproc_config = load_config(args.preprocessing_config_file)
    preproc_config = rewrite_config_arguments(preproc_config, args, 'preprocessing')
    
    preproc_config['normalization_params'] = rewrite_config_arguments(
        preproc_config['normalization_params'], args, 'preprocessing, normalization_params')
    
    preproc_config['normalization_params']['bg_suppression_opts'] = rewrite_config_arguments(
        preproc_config['normalization_params']['bg_suppression_opts'], args,
        'preprocessing, normalization_params, bg_suppression_opts',
        prefix_to_remove='bg_supression_'
        )
    
    return dataset_config, preproc_config


if __name__ == '__main__':

    print(f'Running {__file__}')
    
    print('----------------Loading parameters and setup----------------')
    # Loading general parameters
    # :=========================================================================:
    dataset_config, preproc_config = get_configuration_arguments()
    
    params          = {'dataset': dataset_config, 'preprocessing': preproc_config}
    seed            = preproc_config['seed']
    device          = preproc_config['device']
    logdir          = preproc_config['normalization_params']['logdir']
    
    # Write or load parameters to logdir
    # :=========================================================================:
    os.makedirs(logdir, exist_ok=True)
    dump_config(os.path.join(logdir, 'params.yaml'), params)

    print_config(params, keys=['dataset', 'preprocessing'])
    
    # Define the dataset that is to be used for training
    # :=========================================================================:
    print('Defining dataset')
    seed_everything(seed)
    normalization_params = preproc_config['normalization_params']
    
    device              = define_device(device)
    splits              = preproc_config['splits']
    dataset             = preproc_config['dataset']
    n_classes           = dataset_config[dataset]['n_classes']
    batch_size          = normalization_params['batch_size']
    num_workers         = normalization_params['num_workers']
    bg_suppression_opts = normalization_params['bg_suppression_opts']

    # Dataset and dataloader definition
    datasets = get_datasets(
        paths           = dataset_config[dataset]['paths_processed'],
        paths_original  = dataset_config[dataset]['paths_original'],
        splits          = splits,
        image_size      = normalization_params['image_size'],
        resolution_proc = dataset_config[dataset]['resolution_proc'],
        dim_proc        = dataset_config[dataset]['dim'],
        n_classes       = n_classes,
        aug_params      = normalization_params['augmentation'],
        deformation     = None,
        load_original   = False,
        bg_suppression_opts = bg_suppression_opts
    )

    dataloaders = {
        split: DataLoader(dataset=dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers, drop_last=False) 
        for split, dataset in zip(splits, datasets)
    }
    
    print(f'Dataloaders for dataset {dataset} defined')
    
    # Define the 2D normalization model 
    # :=========================================================================:
    
    print(f'Using Device {device}')
    # Model definition
    norm_model_config   = preproc_config['normalization_params']['model']
      
    norm = Normalization(
        n_layers        = norm_model_config['n_layers'],
        image_channels  = norm_model_config['image_channels'],
        channel_size    = norm_model_config['channel_size'],
        kernel_size     = norm_model_config['kernel_size'],
        activation      = norm_model_config['activation'], 
        batch_norm      = norm_model_config['batch_norm'],
        residual        = norm_model_config['residual'],
        n_dimensions    = norm_model_config['n_dimensions'] 
    ).to(device)

    # Load checkpoint weights
    norm_checkpoint_path = norm_model_config['checkpoint']
    checkpoint = torch.load(norm_checkpoint_path, map_location=device)    
    norm.load_state_dict(checkpoint['norm_state_dict'])
        

    # Loop over each dataloder and obtain the normalized images
    # :=========================================================================:
    print('-----------------------Normalizing images-----------------------')
    norm.eval()
    
    with torch.no_grad():
                        
        for split, dataloader in dataloaders.items():
            # Create new h5file to store normalized images, which is a copy of the original h5file
            orig_h5_path = dataset_config[dataset]['paths_processed'][split]
            filename = os.path.basename(orig_h5_path).strip('.h5')
            new_h5_path = os.path.join(logdir, f'{filename}_normalized_with_nn.h5')
            shutil.copy(orig_h5_path, new_h5_path)
            
            print(f'Normalizing images in split: {split}')
            images = []
            for x, _, _, _, _ in tqdm(dataloader):
                images.append(norm(x.to(device)).cpu().numpy().squeeze())
            
            images = np.concatenate(images, 0)
            
            # Load the copied h5file and write the normalized images
            print(f'Writing normalized images to {new_h5_path}')
            with h5py.File(new_h5_path, 'r+') as f:
                del f['images']
                f.create_dataset('images', data=images)
                images_h5 = images
                
            # Check the values are indeed written
            with h5py.File(new_h5_path, 'r') as f:  
                assert np.allclose(f['images'][:], images), 'Images were not written correctly'
    
    print('-----------------------Done-----------------------')
            