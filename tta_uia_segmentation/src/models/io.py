import os
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import nn as nn
from ema_pytorch import EMA
from denoising_diffusion_pytorch import Unet

from tta_uia_segmentation.src.utils.utils import assert_in
from tta_uia_segmentation.src.models import Normalization, UNet
from tta_uia_segmentation.src.models.ConditionalGaussianDiffusion import ConditionalGaussianDiffusion


def load_ddpm_from_configs_and_cpt(
    train_ddpm_cfg: dict,
    model_ddpm_cfg: dict,
    cpt_fp: str,
    img_size: int,
    device: torch.device,
    sampling_timesteps: Optional[int]) -> ConditionalGaussianDiffusion:

    timesteps           = train_ddpm_cfg['timesteps']
    sampling_timesteps  = train_ddpm_cfg['sampling_timesteps'] \
        if sampling_timesteps is None else sampling_timesteps

    # Model definition
    model = Unet(
        dim = model_ddpm_cfg['dim'],
        dim_mults = model_ddpm_cfg['dim_mults'],   
        flash_attn = True,
        channels=model_ddpm_cfg['channels'], 
        self_condition=True,
    ).to(device)

    ddpm = ConditionalGaussianDiffusion(
        model,
        image_size = img_size,
        timesteps = timesteps,    # Range of steps in diffusion process
        sampling_timesteps = sampling_timesteps 
    ).to(device)
    
    cpt = torch.load(cpt_fp, map_location=device)
    ema_update_every = 10
    ema_decay = 0.995
    ddpm_ema = EMA(ddpm, beta = ema_decay, update_every = ema_update_every)
    ddpm_ema.load_state_dict(cpt['ema'])
    
    return ddpm_ema.model


def load_norm_from_configs_and_cpt(
    model_params_norm: dict,
    cpt_fp: str,
    device: torch.device,
)-> Normalization:
    norm = Normalization(
        n_layers                = model_params_norm['n_layers'],
        image_channels          = model_params_norm['image_channels'],
        channel_size            = model_params_norm['channel_size'],
        kernel_size             = model_params_norm['kernel_size'],
        activation              = model_params_norm['activation'], 
        batch_norm              = model_params_norm['batch_norm'],
        residual                = model_params_norm['residual'],
        n_dimensions            = model_params_norm['n_dimensions'] 
    ).to(device)
    
    checkpoint = torch.load(cpt_fp, map_location=device)
    norm.load_state_dict(checkpoint['norm_state_dict'])
    
    return norm
    
def load_norm_and_seg_from_configs_and_cpt(
    n_classes: int,
    model_params_norm: dict,
    model_params_seg: dict,
    cpt_fp: str,
    device: torch.device,
    return_norm_state_dict: bool = False
)-> Union[tuple[Normalization, UNet, dict], tuple[Normalization, UNet]]:
    norm = Normalization(
        n_layers                = model_params_norm['n_layers'],
        image_channels          = model_params_norm['image_channels'],
        channel_size            = model_params_norm['channel_size'],
        kernel_size             = model_params_norm['kernel_size'],
        activation              = model_params_norm['activation'], 
        batch_norm              = model_params_norm['batch_norm'],
        residual                = model_params_norm['residual'],
        n_dimensions            = model_params_norm['n_dimensions'] 
    ).to(device)

    seg = UNet(
        in_channels             = model_params_seg['image_channels'],
        n_classes               = n_classes,
        channels                = model_params_seg['channel_size'],
        channels_bottleneck     = model_params_seg['channels_bottleneck'],
        skips                   = model_params_seg['skips'],
        n_dimensions            = model_params_seg['n_dimensions'] 
    ).to(device)
    
    checkpoint = torch.load(cpt_fp, map_location=device)
    norm.load_state_dict(checkpoint['norm_state_dict'])
    seg.load_state_dict(checkpoint['seg_state_dict'])
    norm_state_dict = checkpoint['norm_state_dict']
    
    if return_norm_state_dict:
        return norm, seg, norm_state_dict   
    else:
        return norm, seg
    
def load_dae_and_atlas_from_configs_and_cpt(
    n_classes: int,
    model_params_dae: dict,
    cpt_fp: str,
    device: torch.device,
)-> Union[nn.Module, torch.Tensor]:
    
    dae = UNet(
        in_channels             = n_classes,
        n_classes               = n_classes,
        channels                = model_params_dae['channel_size'],
        channels_bottleneck     = model_params_dae['channels_bottleneck'],
        skips                   = model_params_dae['skips'],
        n_dimensions            = model_params_dae['n_dimensions']
    ).to(device)
    
    checkpoint = torch.load(cpt_fp, map_location=device)
    dae.load_state_dict(checkpoint['dae_state_dict'])

    dae_dir = os.path.dirname(cpt_fp)
    checkpoint = torch.load(
        os.path.join(dae_dir, 'atlas.h5py'), map_location=device)
    atlas = checkpoint['atlas']
    
    return dae, atlas