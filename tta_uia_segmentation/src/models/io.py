import os
import json
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import nn as nn
from tdigest import TDigest
from ema_pytorch import EMA
from diffusers import AutoencoderKL, UNet2DModel, UNet2DConditionModel

from tta_uia_segmentation.src.models import (
    Normalization,
    ConditionalUnet,
    UNet,
    ConditionalGaussianDiffusion,
    ConditionalLatentGaussianDiffusion
)
from tta_uia_segmentation.src.models import DomainStatistics

    
def load_cddpm_from_configs_and_cpt(
    train_ddpm_cfg: dict,
    model_ddpm_cfg: dict,
    n_classes: int,
    cpt_fp: str,
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

    image_size = train_ddpm_cfg['image_size'][-1] * train_ddpm_cfg['rescale_factor'][-1]
    ddpm = ConditionalGaussianDiffusion(
        model=model,
        image_size = image_size,
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


def define_and_possibly_load_lcddpm(
    train_config: dict,
    model_config: dict,
    n_classes: int,
    device: Optional[torch.device | str] = None, 
    cpt_fp: Optional[str] = None,
    return_ema_model: bool = False,
    ema_kwargs: dict = dict()
    ) -> ConditionalLatentGaussianDiffusion:

    vae_path = train_config['vae_path']
    if train_config['vae_pretrained_on_nat_images']:
        vae = AutoencoderKL.from_pretrained(vae_path, subfolder='vae')
    else:
        vae = AutoencoderKL.from_pretrained(vae_path)

    # Define Denoising Unet    
    dim                 = model_config['dim']
    dim_mults           = model_config['dim_mults']
    use_x_attention     = model_config['use_x_attention']
    cond_type           = model_config['cond_type']
    time_embedding_dim  = model_config['time_embedding_dim']

    num_downsamples     = len(dim_mults) - 1
    train_image_size    = train_config['image_size'][-1] * train_config['rescale_factor'][-1]
    img_latent_dim      = train_image_size / (2 ** num_downsamples)
    block_out_channels  = [dim * m for m in dim_mults]
    in_channels         = vae.config.latent_channels 
    in_channels         *= 2 if cond_type == 'concat' else 1
    out_channels        = vae.config.latent_channels
    down_attn_type_blocks = 'CrossAttnDownBlock2D' if use_x_attention else 'AttnDownBlock2D'
    up_attn_type_blocks = 'CrossAttnUpBlock2D' if use_x_attention else 'AttnUpBlock2D'
    unet_cls            = UNet2DConditionModel if use_x_attention else UNet2DModel

    down_block_types = [down_attn_type_blocks] * num_downsamples + ['DownBlock2D']
    up_block_types = ['UpBlock2D'] + [up_attn_type_blocks] * num_downsamples 
     
    unet_kwargs = dict(
        sample_size = img_latent_dim,
        in_channels = in_channels,
        out_channels = out_channels,
        block_out_channels = block_out_channels,
        down_block_types = down_block_types,
        up_block_types = up_block_types,
    )

    if use_x_attention:
        unet_kwargs['cross_attention_dim'] = time_embedding_dim
        unet_kwargs['time_embedding_dim'] = time_embedding_dim

    unet = unet_cls(**unet_kwargs)
    
    # Define the diffusion model    
    objective           = train_config['objective']
    timesteps           = train_config['timesteps']
    sampling_timesteps  = train_config['sampling_timesteps']
    
    unconditional_rate  = train_config['unconditional_rate']

    fit_emb_for_cond_img= train_config['fit_emb_for_cond_img']
    cond_type           = model_config['cond_type']

    snr_weighting_gamma = train_config['snr_weighting_gamma']
    rescale_betas_zero_snr = train_config['rescale_betas_zero_snr']

    ddpm = ConditionalLatentGaussianDiffusion(
        vae=vae,
        unet=unet,
        train_image_size=train_image_size,
        cond_img_channels=n_classes,
        objective=objective,
        forward_type='train_loss',
        num_train_timesteps=timesteps,
        num_sample_timesteps=sampling_timesteps,
        unconditional_rate=unconditional_rate,
        fit_emb_for_cond_img=fit_emb_for_cond_img,
        cond_type=cond_type,
        snr_weighting_gamma=snr_weighting_gamma,
        rescale_betas_zero_snr=rescale_betas_zero_snr,
    )

    if cpt_fp is not None:
        cpt = torch.load(cpt_fp, map_location=device)

        if return_ema_model:
            ema_kwargs.setdefault('update_every', 10)
            ema_kwargs.setdefault('beta', 0.995)
            ddpm_ema = EMA(ddpm, **ema_kwargs)
            ddpm_ema = ddpm_ema.to(device) if device is not None else ddpm_ema
            ddpm_ema.load_state_dict(cpt['ema'])
        else:
            ddpm = ddpm.to(device) if device is not None else ddpm
            ddpm.load_state_dict(cpt['model'])
    
    return ddpm


def define_and_possibly_load_cddpm(
    img_channels: int,
    train_config: dict,
    model_config: dict,
    n_classes: int,
    device: Optional[torch.device | str] = None, 
    cpt_fp: Optional[str] = None,
    return_ema_model: bool = False,
    ema_kwargs: dict = dict()
    ) -> ConditionalGaussianDiffusion:

    # Define Denoising Unet    
    dim                 = model_config['dim']
    dim_mults           = model_config['dim_mults']
    use_x_attention     = model_config['use_x_attention']
    cond_type           = model_config['cond_type']
    time_embedding_dim  = model_config['time_embedding_dim']

    num_downsamples     = len(dim_mults) - 1
    train_image_size    = train_config['image_size'][-1] * train_config['rescale_factor'][-1]
    img_latent_dim      = train_image_size / (2 ** num_downsamples)
    block_out_channels  = [dim * m for m in dim_mults]
    in_channels         = img_channels 
    in_channels         *= 2 if cond_type == 'concat' else 1
    out_channels        = img_channels
    down_attn_type_blocks = 'CrossAttnDownBlock2D' if use_x_attention else 'AttnDownBlock2D'
    up_attn_type_blocks = 'CrossAttnUpBlock2D' if use_x_attention else 'AttnUpBlock2D'
    unet_cls            = UNet2DConditionModel if use_x_attention else UNet2DModel

    down_block_types = [down_attn_type_blocks] * num_downsamples + ['DownBlock2D']
    up_block_types = ['UpBlock2D'] + [up_attn_type_blocks] * num_downsamples 
     
    unet_kwargs = dict(
        sample_size = img_latent_dim,
        in_channels = in_channels,
        out_channels = out_channels,
        block_out_channels = block_out_channels,
        down_block_types = down_block_types,
        up_block_types = up_block_types,
    )

    if use_x_attention:
        unet_kwargs['cross_attention_dim'] = time_embedding_dim
        unet_kwargs['time_embedding_dim'] = time_embedding_dim

    unet = unet_cls(**unet_kwargs)
    
    # Define the diffusion model    
    objective           = train_config['objective']
    timesteps           = train_config['timesteps']
    sampling_timesteps  = train_config['sampling_timesteps']
    
    unconditional_rate  = train_config['unconditional_rate']
    cond_type           = model_config['cond_type']

    snr_weighting_gamma = train_config['snr_weighting_gamma']
    rescale_betas_zero_snr = train_config['rescale_betas_zero_snr']

    ddpm = ConditionalGaussianDiffusion(
        unet=unet,
        train_image_size=train_image_size,
        img_channels=img_channels,
        cond_img_channels=n_classes,
        objective=objective,
        forward_type='train_loss',
        num_train_timesteps=timesteps,
        num_sample_timesteps=sampling_timesteps,
        unconditional_rate=unconditional_rate,
        cond_type=cond_type,
        snr_weighting_gamma=snr_weighting_gamma,
        rescale_betas_zero_snr=rescale_betas_zero_snr,
    )

    if cpt_fp is not None:
        cpt = torch.load(cpt_fp, map_location=device)

        if return_ema_model:
            ema_kwargs.setdefault('update_every', 10)
            ema_kwargs.setdefault('beta', 0.995)
            ddpm_ema = EMA(ddpm, **ema_kwargs)
            ddpm_ema = ddpm_ema.to(device) if device is not None else ddpm_ema
            ddpm_ema.load_state_dict(cpt['ema'])
        else:
            ddpm = ddpm.to(device) if device is not None else ddpm
            ddpm.load_state_dict(cpt['model'])
    
    return ddpm


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
    device: torch.device
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
    norm_state_dict = checkpoint['norm_state_dict']
    seg_state_dict = checkpoint['seg_state_dict']
    
    if 'norm_state_dict' in norm_state_dict:
        norm_state_dict = norm_state_dict['norm_state_dict']

    norm.load_state_dict(norm_state_dict)
    seg.load_state_dict(seg_state_dict)

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