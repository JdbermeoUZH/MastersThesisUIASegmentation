import os
import json
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import nn as nn
from tdigest import TDigest
from ema_pytorch import EMA

from tta_uia_segmentation.src.models import Normalization, ConditionalUnet, UNet
from tta_uia_segmentation.src.models.ConditionalGaussianDiffusion import ConditionalGaussianDiffusion
from tta_uia_segmentation.src.models.UNetModelOAI import create_model_conditioned_on_seg_mask, model_defaults
from tta_uia_segmentation.src.models.ConditionalGaussianDiffusionOAI import ConditionalGaussianDiffusionOAI, diffusion_defaults
from tta_uia_segmentation.src.models import DomainStatistics
from improved_diffusion.script_util import create_gaussian_diffusion
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.respace import SpacedDiffusion


def load_icddpm_from_configs_and_cpt(
    train_ddpm_cfg: dict,
    model_ddpm_cfg: dict,
    n_classes: int,
    image_channels: int,
    cpt_fp: str,
    sampling_timesteps: str,
    device: torch.device,
) -> tuple[ConditionalUnet, SpacedDiffusion, nn.Module]:
    
    unet_model = create_model_conditioned_on_seg_mask(
        image_size = train_ddpm_cfg['image_size'][-1],
        image_channels = image_channels,
        seg_cond = train_ddpm_cfg['seg_cond'],
        num_channels = model_ddpm_cfg['num_channels'],
        channel_mult = model_ddpm_cfg['channel_mult'],
        learn_sigma = train_ddpm_cfg['learn_sigma'],
        n_classes = n_classes,
        num_res_blocks = model_ddpm_cfg['num_res_blocks'],
        num_heads=model_ddpm_cfg['num_heads'],
        dropout = train_ddpm_cfg['dropout'],
        **model_defaults()
    ).to(device)
    # Load parameterers of the Unet model
    unet_model.load_state_dict(torch.load(cpt_fp, map_location=device))

    timestep_respacing = str(sampling_timesteps) if sampling_timesteps is not None\
        else train_ddpm_cfg['timestep_respacing']

    diffusion = create_gaussian_diffusion(
        steps=train_ddpm_cfg['diffusion_steps'],
        learn_sigma=train_ddpm_cfg['learn_sigma'],
        noise_schedule=train_ddpm_cfg['noise_schedule'],
        use_kl=train_ddpm_cfg['use_kl'],
        timestep_respacing=timestep_respacing,
        **diffusion_defaults()
    )
    
    # Define a schedule sampler
    schedule_sampler = train_ddpm_cfg['schedule_sampler']
    schedule_sampler = create_named_schedule_sampler(schedule_sampler, diffusion)
    
    return ConditionalGaussianDiffusionOAI(
        model=unet_model,
        ddpm=diffusion,
        schedule_sampler=schedule_sampler,
        device=device
    )
    
    
def load_cddpm_from_configs_and_cpt(
    train_ddpm_cfg: dict,
    model_ddpm_cfg: dict,
    n_classes: int,
    cpt_fp: str,
    img_size: int,
    device: torch.device,
    sampling_timesteps: Optional[int] = None,
    unconditional_rate: Optional[float] = None ) -> ConditionalGaussianDiffusion:

    timesteps           = train_ddpm_cfg['timesteps']
    sampling_timesteps  = train_ddpm_cfg['sampling_timesteps'] \
        if sampling_timesteps is None else sampling_timesteps

    # Model definition
    model = ConditionalUnet(
        dim = model_ddpm_cfg['dim'],
        dim_mults = model_ddpm_cfg['dim_mults'],   
        n_classes = n_classes,
        flash_attn = True,
        image_channels=model_ddpm_cfg['channels'], 
        condition_by_concat=not train_ddpm_cfg['condition_by_mult'],
    ).to(device)

    ddpm = ConditionalGaussianDiffusion(
        model=model,
        image_size = img_size,
        timesteps = timesteps,    # Range of steps in diffusion process
        sampling_timesteps = sampling_timesteps,
        also_unconditional = train_ddpm_cfg['also_unconditional'],
        unconditional_rate = train_ddpm_cfg['unconditional_rate'], 
    ).to(device)
    
    if unconditional_rate is not None:
        ddpm.unconditional_rate = unconditional_rate 

    cpt = torch.load(cpt_fp, map_location=device)
    ema_update_every = 10
    ema_decay = 0.995
    ddpm_ema = EMA(ddpm, beta = ema_decay, update_every = ema_update_every)
    ddpm_ema = ddpm_ema.to(device)
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
    

def load_domain_statistiscs(
    cpt_fp: str,
    frozen: bool,
    momentum: float = 0.96,
    min_max_clip_q: tuple[float, float] = (0.025, 0.975)
) -> DomainStatistics:
    # Load the statistics and quantiles dict
    checkpoint_filepath_prefix = cpt_fp.replace('.pth', '')
    
    stats_dict = json.load(
        open(f'{checkpoint_filepath_prefix}_moments_and_quantiles.json', 'r'))
    
    # Load the TDigest object to calculate quantiles
    digest_dict = json.load(
        open(f'{checkpoint_filepath_prefix}_serialized_TDigest' + 
             '_to_calculate_quantiles.json', 'r')
    )
    quantile_cal = TDigest().update_from_dict(digest_dict)
    
    ds = DomainStatistics(
        mean=stats_dict['mean'],
        std=stats_dict['std'],
        quantile_cal = quantile_cal,
        precalculated_quantiles = stats_dict['quantiles'],
        momentum = momentum,
        frozen = frozen
    )
    
    ds.min = ds.get_quantile(min_max_clip_q[0])
    ds.max = ds.get_quantile(min_max_clip_q[1])
        
    return ds  
    
    
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