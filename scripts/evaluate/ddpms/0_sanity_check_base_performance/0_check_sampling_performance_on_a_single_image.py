import os
import sys
import argparse
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


sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..')))

from tta_uia_segmentation.src.models import ConditionalGaussianDiffusion
from tta_uia_segmentation.src.dataset import DatasetInMemoryForDDPM 
from ..utils import metrics_to_log_default


# Default args
out_dir         = '/scratch_net/biwidl319/jbermeo/results/ddpm/sanity_checks/'
exp_name        = 'default_exp_name'
params_fp       = '/scratch_net/biwidl319/jbermeo/logs/brain/cddpm/params.yaml'
cpt_fp          = '/scratch_net/biwidl319/jbermeo/logs/brain/cddpm/model-20.pt'
ddim_steps      = 100
dataset         = 'hcp_t1'
seed            = 1234 
device          = 'cpu' if not  torch.cuda.is_available() else 'cuda'
ddim_only       = False
split           = 'train'


def store_results_sampling(
    img: torch.Tensor, seg: torch.Tensor, sampling_progress: list[: torch.Tensor],
    output_dir: str, metrics=metrics_to_log_default
    ):
    
    os.makedirs(output_dir, exist_ok=True)

    
    final_sampled_img = sampling_progress[:, -1, ...]
    
    # Plot the img, gt and sampled img side by side
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img.squeeze().detach().cpu().numpy(), cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(seg.squeeze().detach().cpu().numpy(), cmap='viridis')
    ax[1].set_title('Segmentation GT')  
    ax[2].imshow(final_sampled_img.squeeze().detach().cpu().numpy(), cmap='gray')
    ax[2].set_title('Sampled Image from DDPM')
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
            f.write(f'{metric_name}: {metric_fun(img, final_sampled_img).item()} \n')
            


def get_cmd_args():
    parser = argparse.ArgumentParser(description="Check quality of DDPM sampling on a single image")

    parse_booleans = lambda x: x.lower() in ['true', '1']
    
    parser.add_argument('--out_dir', type=str, default=out_dir, help='Output directory')
    parser.add_argument('--exp_name', type=str, default=exp_name, help='Experiment name')
    parser.add_argument('--params_fp', type=str, default=params_fp, help='Path to the params file')
    parser.add_argument('--cpt_fp', type=str, default=cpt_fp, help='Path to the checkpoint file')
    parser.add_argument('--ddim_steps', type=int, default=ddim_steps, help='Number of steps for DDIM sampling')
    parser.add_argument('--dataset', type=str, default=dataset, help='Dataset to use')
    parser.add_argument('--split', type=str, default=split, help='Split to use', options=['train', 'val', 'test'])
    parser.add_argument('--seed', type=int, default=seed, help='Seed for reproducibility')
    parser.add_argument('--device', type=str, default=device, help='Device to use')
    parser.add_argument('--ddim_only', type=parse_booleans, default=ddim_only, help='Whether to only use DDIM sampling')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cmd_args()
    
    # Create output dir
    out_dir = os.path.join(out_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
        
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    run_params = yaml.safe_load(open(args.params_fp, 'r'))
    dataset_params = run_params['dataset'][args.dataset]
    training_params = run_params['training']['ddpm']
    
    dataset = DatasetInMemoryForDDPM(
        split           = args.split,
        one_hot_encode  = training_params['one_hot_encode'], 
        normalize       = training_params['normalize'],
        paths           = dataset_params['paths_normalized_with_nn'],
        paths_original  = dataset_params['paths_original'],
        image_size      = dataset_params['image_size'],
        resolution_proc = dataset_params['resolution_proc'],
        dim_proc        = dataset_params['dim_proc'],
        n_classes       = dataset_params['n_classes'],
        aug_params      = None,
        bg_suppression_opts = None,
        deformation     = None,
        load_original   = False,
    )
    
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
    img = torch.from_numpy(rand_img).unsqueeze(0).unsqueeze(0).float().to(device)
    img_norm = ddpm.normalize(img)
    img_unnorm = (img - img.min()) / (img.max() - img.min())

    x_cond = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).float().to(device)
    x_cond = x_cond / (n_classes - 1)
    #x_cond_norm = ddpm.normalize(x_cond)
    
    # Sampling from scratch
    # :===============================================================:
    print('Sampling with DDIM')
    ddim_samp_dir = os.path.join(out_dir, '2_ddim_sampling_x_cond_not_normalized')
    sampling_steps_old = ddpm_ema.model.sampling_timesteps 
    ddpm_ema.model.sampling_timesteps = ddim_steps
    sampled_img_progress = ddpm_ema.model.ddim_sample(x_cond, True)
    final_sampled_img = sampled_img_progress[:, -1, ...]
    store_results_sampling(img_unnorm, x_cond, sampled_img_progress, ddim_samp_dir)
    
    print('Sampling from noise with linear sampling')
    lin_samp_dir = os.path.join(out_dir, '1_linear_sampling')    
    sampled_img_progress = ddpm_ema.model.p_sample_loop(x_cond, return_all_timesteps=True)
    final_sampled_img = sampled_img_progress[:, -1, ...]
    store_results_sampling(img_unnorm, x_cond, sampled_img_progress, lin_samp_dir)

    

    
