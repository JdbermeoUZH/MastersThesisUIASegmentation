import os
import sys
from tqdm import tqdm

import h5py
import yaml
import torch
import random
import numpy as np
from PIL import Image
from ema_pytorch import EMA
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    multiscale_structural_similarity_index_measure
)
from torchmetrics.functional.regression import mean_absolute_error, mean_squared_error

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..')))#, 'tta_uia_segmentation', 'src')))

from tta_uia_segmentation.src.models import ConditionalGaussianDiffusion

metrics_to_log_default = {
    'PSNR': peak_signal_noise_ratio,
    'SSIM': structural_similarity_index_measure,
    'MSSIM': multiscale_structural_similarity_index_measure,
    'MAE': mean_absolute_error,
    'MSE': mean_squared_error
}


def plot_denoising_progress(img_gt, seg_gt, img_w_noise, ex_img_w_noise_ti, img_denoist_at_ti_plus_1, img_denoised_sampled, img_denoised, noise_T, t_i):
    fig, ax = plt.subplots(1, 7, figsize=(35, 5))
    ax[0].imshow(img_gt.squeeze().numpy(), cmap='gray')
    ax[0].set_title('Original')

    ax[1].imshow(seg_gt.squeeze().numpy(), cmap='viridis')
    ax[1].set_title('Segmentation GT')

    ax[2].imshow(img_w_noise.squeeze().numpy(), cmap='gray')
    ax[2].set_title(f'Starting Noised img @ T={noise_T}')
    
    ax[3].imshow(ex_img_w_noise_ti.squeeze().numpy(), cmap='gray')
    ax[3].set_title(f'Example of Noised img @ t={t_i}')

    ax[4].imshow(img_denoist_at_ti_plus_1.squeeze().numpy(), cmap='gray')
    ax[4].set_title(f'Denoised - sampled est of t={t_i} (Current input)')

    ax[5].imshow(img_denoised_sampled.squeeze().numpy(), cmap='gray')
    ax[5].set_title(f'Denoised - sampled est. of t={t_i -1}')

    ax[6].imshow(img_denoised.squeeze().numpy(), cmap='gray')
    ax[6].set_title(f'Denoised img est. of t=0 @ t={t_i}')
    
    plt.show()
    
    
def evaluate_linear_denoising(
    img: np.ndarray,
    seg: np.ndarray,
    n_classes: int,
    t: int,
    ddpm: GaussianDiffusion,
    plot_every: int = 50,
    measure_metrics_every: int = 5,
    metrics: dict = metrics_to_log_default
) -> dict[str, dict]: 
    
    # Normalize image
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    img = ddpm.normalize(img)
    
    # Normalize segmentation
    x_cond = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).float()
    x_cond = x_cond / (n_classes - 1)
    x_cond = ddpm.normalize(x_cond)

    # Generate noised verion of the image
    t_tch = torch.full((1,), t)
    noise = torch.randn_like(img)
    noised_img = ddpm.q_sample(img, t_tch, noise)

    img_denoised = noised_img
    metrics_logs = {k: list() for k in metrics.keys()}

    for t_i in tqdm(reversed(range(0, t)), desc = 'sampling loop time step', total = t):
        img_denoised_at_ti, img_denoised_at_t0 = ddpm.p_sample(img_denoised, t_i, x_cond)
        
        example_noised_img_at_ti = ddpm.q_sample(img, torch.full((1,), t_i), torch.randn_like(img))
        
        if t_i % plot_every == 0 or t_i == 0:
            plot_denoising_progress(img, x_cond, noised_img, example_noised_img_at_ti, img_denoised,
                                    img_denoised_at_ti, img_denoised_at_t0, t, t_i)
            
        if t_i % measure_metrics_every == 0 or t_i == 0:
            for metric_name, metric_log in metrics_logs.items():
                metric_log.append((
                    t - t_i,
                    metrics_to_log_default[metric_name](img, img_denoised_at_ti).item(),
                    metrics_to_log_default[metric_name](img, img_denoised_at_t0).item()
                ))
                
    return metrics_logs


def store_results_sampling(
    img: torch.Tensor, seg: torch.Tensor, sampling_progress: list[: torch.Tensor],
    output_dir: str, metrics=metrics_to_log_default
    ):
    
    os.makedirs(output_dir, exist_ok=True)

    
    final_sampled_img = sampling_progress[:, -1, ...]
    
    # Plot the img, gt and sampled img side by side
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img.squeeze().detach().numpy(), cmap='gray')
    ax[1].imshow(seg.squeeze().detach().numpy(), cmap='viridis')
    ax[2].imshow(final_sampled_img.squeeze().detach().numpy(), cmap='gray')
    plt.savefig(os.path.join(output_dir, '0_img_gt_sampled_img_ddim.png'))
    
    # Store a gif of the sampling progress
    sampled_img_progress_np = sampling_progress.squeeze().cpu().numpy()
    sampled_img_progress_np = (sampled_img_progress_np * 255).astype(np.uint8)
    sampled_img_progress = [Image.fromarray(sampled_img_progress_np[img_i])
                            for img_i in range(sampled_img_progress_np.shape[0])]
    sampled_img_progress[0].save(os.path.join(output_dir, '1_sampling_progress.gif'), 
                                 save_all=True, append_images=sampled_img_progress[1:], 
                                 optimize=False, duration=150, loop=0)
     
    # Store metrics of final sampled img
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        for metric_name, metric_fun in metrics.items():
            f.write(f'{metric_name}: {metric_fun(img_unnorm, final_sampled_img).item()} \n')
            

if __name__ == '__main__':
    # Output dir
    out_dir = '/scratch_net/biwidl319/jbermeo/results/ddpm/sanity_checks/w_gpu'
    exp_name = 'cddpm_64_128_256_512_100k_steps_b_16'
    out_dir = os.path.join(out_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Script params
    img_size        = 256
    n_classes       = 15
    seed            = 1234 
    run_params      = yaml.safe_load(open('/scratch_net/biwidl319/jbermeo/logs/brain/cddpm/params.yaml', 'r'))
    cpt_path        = '/scratch_net/biwidl319/jbermeo/logs/brain/cddpm/model-20.pt'
    device          = 'cuda'
    ddim_steps      = 100
    
    np.random.seed(1234)
    random.seed(1234)
    
    # Load the data
    # :===============================================================:
    ds_fp = '/scratch_net/biwidl319/jbermeo/logs/brain/preprocessing/hcp_t1/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_20.hdf_normalized_with_nn.h5'
    ds_h5 = h5py.File(ds_fp, 'r')
    
    # Load trandom image
    vol_idx = random.randint(0, int(ds_h5['images'].shape[0]/img_size))
    slice_idx = random.randint(int(0.2 * img_size), int(0.8 * img_size))
    rand_img = ds_h5['images'][vol_idx * img_size + slice_idx, :, :]
    seg = ds_h5['labels'][vol_idx * img_size + slice_idx, :, :]
        
    # Save img of the slice
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rand_img, cmap='gray')
    ax[1].imshow(seg, cmap='viridis')
    plt.savefig(os.path.join(out_dir, '0_gt_rand_slice.png'))
    
    # Load the model
    # :===============================================================:
    train_config = run_params['training']['ddpm']

    timesteps           = train_config['timesteps']
    sampling_timesteps  = train_config['sampling_timesteps']

    # Model definition
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),   
        flash_attn = True,
        channels=1, 
        self_condition=True,
    ).to(device)

    ddpm = ConditionalGaussianDiffusion(
        model,
        image_size = 256,
        timesteps = timesteps,    # Range of steps in diffusion process
        sampling_timesteps = sampling_timesteps 
    ).to(device)
    
    cpt = torch.load(cpt_path, map_location=device)
    ema_update_every = 10
    ema_decay = 0.995
    ddpm_ema = EMA(ddpm, beta = ema_decay, update_every = ema_update_every)
    ddpm_ema.load_state_dict(cpt['ema'])
    
    # Prepare images for sampling
    # :===============================================================:
    img = torch.from_numpy(rand_img).unsqueeze(0).unsqueeze(0).float()
    img_norm = ddpm.normalize(img)
    img_unnorm = (img - img.min()) / (img.max() - img.min())

    x_cond = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).float()
    x_cond = x_cond / (n_classes - 1)
    x_cond_norm = ddpm.normalize(x_cond)
    
    # Sampling from scratch
    # :===============================================================:print('Sampling with DDIM')
    ddim_samp_dir = os.path.join(out_dir, '2_ddim_sampling_x_cond_not_normalized')
    sampling_steps_old = ddpm_ema.model.sampling_timesteps 
    ddpm_ema.model.sampling_timesteps = ddim_steps
    sampled_img_progress = ddpm_ema.model.ddim_sample(x_cond, True)
    final_sampled_img = sampled_img_progress[:, -1, ...]
    store_results_sampling(img_unnorm, x_cond, sampled_img_progress, ddim_samp_dir)
    
    print('Sampling from noise with linear sampling')
    lin_samp_dir = os.path.join(out_dir, '1_linear_sampling')    
    sampled_img_progress = ddpm_ema.model.p_sample_loop(x_cond_norm, return_all_timesteps=True)
    final_sampled_img = sampled_img_progress[:, -1, ...]
    store_results_sampling(img_unnorm, x_cond, sampled_img_progress, lin_samp_dir)

    

    
