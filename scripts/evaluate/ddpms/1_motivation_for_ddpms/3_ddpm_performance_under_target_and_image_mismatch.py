"""
Assess how does denoising performance degrade when there is a mismatch between segmentation mask and the image 

We do this in three ways:
  1. for a given image, use another image from the same patient but from very different cut
  2. for a given image, find the segmentation masks from another patient around -5, +5 cuts. Most images are registered to the same space, so the shift will be due to patient's anatomy
  3. for a given image, find the segmentation masks for the same patient around -5, +5 cuts  

We set the baseline or the ceiling at the same performance as the original image and the original segmentation mask.

"""

import os
import argparse

import numpy as np
import torch

from dataset import DatasetInMemoryForDDPM
from models import ConditionalGaussianDiffusion
from utils.io import load_config


def preprocess_cmd_args() -> argparse.Namespace: 
  parser = argparse.ArgumentParser(description="Train Segmentation Model (with shallow normalization module)")
  
  parser.add_argument('logdir', type=str, help='Path to the directory where the parameters of the trained model and its checkpoints are stored')
  parser.add_argument('dataset', type=str, help='Name of the dataset to use')
  parser.add_argument('--split', type=str, help='Name of the split to use', default='train', options=['train', 'val', 'test'])
  
  
  args = parser.parse_args()
  
  return args



if __name__  ==  '__main__':
  args = preprocess_cmd_args()
  
  # Load the configuration parameters
  assert os.path.exists(os.path.join(args.logdir, 'params.yaml')), "No parameters file found"
  params = load_config(os.path.join(logdir, 'params.yaml'))
  dataset_config = params['dataset']
  model_config = params['model']
  train_config = params['training']
  
  # Load the dataset
  dataset = get_datasets(
    splits          = [args.split],
    paths           = dataset_config[dataset]['paths_processed'],
    paths_original  = dataset_config[dataset]['paths_original'],
    image_size      = train_config[train_type]['image_size'],
    resolution_proc = dataset_config[dataset]['resolution_proc'],
    dim_proc        = dataset_config[dataset]['dim'],
    n_classes       = dataset_config[dataset]['n_classes'],
    aug_params      = None,
    bg_suppression_opts = None,
    deformation     = None,
    load_original   = False,
  )
  
  # Load the model
  dim                 = model_config['ddpm_unet']['dim']
  dim_mults           = model_config['ddpm_unet']['dim_mults']
  channels            = model_config['ddpm_unet']['channels']
  
  timesteps           = train_config[train_type]['timesteps']
  sampling_timesteps  = train_config[train_type]['sampling_timesteps']
  
  print(f'Using Device {device}')
  # Model definition
  model = Unet(
      dim=dim,
      dim_mults=dim_mults,   
      flash_attn=True,
      channels=channels, 
      self_condition=True,
  )
  
  diffusion = ConditionalGaussianDiffusion(
      model,
      image_size=train_config[train_type]['image_size'][-1],
      timesteps=timesteps,    # Range of steps in diffusion process
      sampling_timesteps = sampling_timesteps 
  )
