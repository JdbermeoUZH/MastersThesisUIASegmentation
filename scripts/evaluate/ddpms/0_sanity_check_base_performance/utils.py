import os
import re
import glob
from typing import Literal

import torch
import numpy as np
from ema_pytorch import EMA
import matplotlib.pyplot as plt
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    multiscale_structural_similarity_index_measure
)
from torchmetrics.functional.regression import mean_absolute_error, mean_squared_error

from tta_uia_segmentation.src.dataset import DatasetInMemoryForDDPM 
from tta_uia_segmentation.src.models import ConditionalGaussianDiffusion, ConditionalUnet


metrics_to_log_default = {
    'PSNR': peak_signal_noise_ratio,
    'SSIM': structural_similarity_index_measure,
    'MSSIM': multiscale_structural_similarity_index_measure,
    'MAE': mean_absolute_error,
    'MSE': mean_squared_error,
}

metric_preferences = {
    'PSNR': 'max',
    'SSIM': 'max',
    'MSSIM': 'max',
    'MAE': 'min',
    'MSE': 'min',
}


def is_new_value_better(old_value: float, new_value: float, preference: Literal['min', 'max']) -> bool:
    if preference == 'min':
        return new_value < old_value
    elif preference == 'max':
        return new_value > old_value
    else:
        raise ValueError(f"Unknown preference: {preference}, valid values are 'min' or 'max'")


def get_last_milestone(logdir: str) -> int:
    pattern = r'model-(\d+)\.pt'
    checkpoints_fps = glob.glob(os.path.join(logdir, 'model-*.pt'))
    
    assert len(checkpoints_fps) > 0, "No milestone checkpoints found" 
    
    checkpoints_fns = [os.path.basename(fps) for fps in checkpoints_fps]
    
    milestones = [int(re.search(pattern, fn).group(1)) if re.search(pattern, fn) else -1
                  for fn in checkpoints_fns]
    
    last_milestone = max(milestones)
    assert last_milestone != -1, "Could not find the last milestone"
    
    return last_milestone


def load_dataset_from_configs(dataset_cfg, training_cfg, **kwargs) -> DatasetInMemoryForDDPM:
    paths_type = 'paths_normalized_with_nn' if training_cfg['use_nn_normalized'] else 'paths_processed'
    
    dataset = DatasetInMemoryForDDPM(
        one_hot_encode  = training_cfg['one_hot_encode'], 
        normalize       = training_cfg['normalize'],
        paths           = dataset_cfg[paths_type],
        paths_original  = dataset_cfg['paths_original'],
        image_size      = training_cfg['image_size'],
        resolution_proc = dataset_cfg['resolution_proc'],
        dim_proc        = dataset_cfg['dim'],
        n_classes       = dataset_cfg['n_classes'],
        **kwargs
    )
    
    return dataset


def load_ddpm_from_configs_and_cpt(
    train_ddpm_cfg: dict,
    model_ddpm_cfg: dict,
    cpt_fp: str,
    n_classes: int,
    device: torch.device) -> ConditionalGaussianDiffusion:

    timesteps           = train_ddpm_cfg['timesteps']
    sampling_timesteps  = train_ddpm_cfg['sampling_timesteps']

    # Model definition
    model = ConditionalUnet(
        dim=model_ddpm_cfg['dim'],
        dim_mults=model_ddpm_cfg['dim_mults'],
        n_classes=n_classes,   
        flash_attn=True,
        image_channels=model_ddpm_cfg['channels'], 
        condition_by_concat=train_ddpm_cfg['condition_by_mult']
    ).to(device)

    ddpm = ConditionalGaussianDiffusion(
        model,
        image_size = train_ddpm_cfg['image_size'][-1],
        timesteps = timesteps,    # Range of steps in diffusion process
        sampling_timesteps = sampling_timesteps 
    ).to(device)
    
    cpt = torch.load(cpt_fp, map_location=device)
    ema_update_every = 10
    ema_decay = 0.995
    ddpm_ema = EMA(ddpm, beta = ema_decay, update_every = ema_update_every)
    ddpm_ema.load_state_dict(cpt['ema'])
    
    return ddpm_ema.model
    


def plot_denoising_progress(
    img_gt: torch.Tensor, seg_gt: torch.Tensor,
    img_w_noise: torch.Tensor, ex_img_w_noise_ti: torch.Tensor,
    img_denoist_at_ti_plus_1: torch.Tensor, img_denoised_sampled: torch.Tensor, img_denoised: torch.Tensor,
    noise_T: int, t_i: int,
    return_fig: bool = False
    ):
    
    fig, ax = plt.subplots(1, 7, figsize=(35, 5))
    ax[0].imshow(img_gt.squeeze().cpu().numpy(), cmap='gray')
    ax[0].set_title('Original')

    ax[1].imshow(seg_gt.squeeze().cpu().numpy(), cmap='viridis')
    ax[1].set_title('Segmentation GT')

    ax[2].imshow(img_w_noise.squeeze().cpu().numpy(), cmap='gray')
    ax[2].set_title(f'Starting Noised img @ T={noise_T}')
    
    ax[3].imshow(ex_img_w_noise_ti.squeeze().cpu().numpy(), cmap='gray')
    ax[3].set_title(f'Example of Noised img @ t={t_i}')

    ax[4].imshow(img_denoist_at_ti_plus_1.squeeze().cpu().numpy(), cmap='gray')
    ax[4].set_title(f'Denoised - sampled est of t={t_i} (Current input)')

    ax[5].imshow(img_denoised_sampled.squeeze().cpu().numpy(), cmap='gray')
    ax[5].set_title(f'Denoised - sampled est. of t={t_i -1}')

    ax[6].imshow(img_denoised.squeeze().cpu().numpy(), cmap='gray')
    ax[6].set_title(f'Denoised img est. of t=0 @ t={t_i}')
    
    if return_fig:
        return fig
    
    plt.show()
    
    
def fig_to_np(fig: plt.Figure) -> torch.Tensor:
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data
