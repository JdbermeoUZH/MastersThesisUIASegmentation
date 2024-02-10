import os
import sys
import argparse

import yaml
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..')))

from utils import (
    metrics_to_log_default,
    load_dataset_from_configs,
    load_ddpm_from_configs_and_cpt
)


# Default args
out_dir         = '/scratch_net/biwidl319/jbermeo/results/ddpm/sanity_checks/'
exp_name        = 'default_exp_name'
params_fp       = '/scratch_net/biwidl319/jbermeo/logs/brain/ddpm_old_exps/on_non_nn_normalized_imgs/cddpm/not_one_hot_128_base_filters/params.yaml'
cpt_fp          = '/scratch_net/biwidl319/jbermeo/logs/brain/ddpm_old_exps/on_non_nn_normalized_imgs/cddpm/not_one_hot_128_base_filters/model-2-step_10000.pt'
ddim_steps      = 100
dataset         = 'hcp_t1'
seed            = 1234 
device          = 'cpu' if not  torch.cuda.is_available() else 'cuda'
ddim_only       = False
split           = 'train'


def store_results_sampling(
    img: torch.Tensor, seg: torch.Tensor, sampling_progress: list[torch.Tensor],
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
    parser.add_argument('--split', type=str, default=split, help='Split to use', choices=['train', 'val', 'test'])
    parser.add_argument('--seed', type=int, default=seed, help='Seed for reproducibility')
    parser.add_argument('--device', type=str, default=device, help='Device to use')
    parser.add_argument('--ddim_only', type=parse_booleans, default=ddim_only, help='Whether to only use DDIM sampling')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cmd_args()
    
    # Create output dir
    out_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
        
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    run_params = yaml.safe_load(open(args.params_fp, 'r'))
    dataset_params = run_params['dataset'][args.dataset]
    training_params = run_params['training']['ddpm']
    paths_type          = 'paths_normalized_with_nn' if training_params['use_nn_normalized'] \
                            else 'paths_processed'
        
    # Load the data
    # :===============================================================:
    dataset = load_dataset_from_configs(
        split           = args.split,
        aug_params      = None,
        bg_suppression_opts = None,
        deformation     = None,
        dataset_cfg     = dataset_params,
        training_cfg    = training_params,
    )

    # Load trandom image
    img_size = dataset.image_size[-1]   # DHW
    vol_idx = random.randint(0, dataset.num_vols)
    slice_idx = random.randint(int(0.2 * img_size), int(0.8 * img_size))
    rand_img, rand_seg = dataset[vol_idx * img_size + slice_idx]
    rand_img = rand_img.unsqueeze(0).to(device)
    rand_seg = rand_seg.unsqueeze(0).to(device)
        
    # Save img of the slice
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rand_img.squeeze().cpu().numpy(), cmap='gray')
    ax[1].imshow(rand_seg.squeeze().cpu().numpy(), cmap='viridis')
    plt.savefig(os.path.join(out_dir, '0_gt_rand_slice.png'))
    
    # Load the model
    # :===============================================================:
    train_config = run_params['training']['ddpm']

    timesteps           = train_config['timesteps']
    sampling_timesteps  = train_config['sampling_timesteps']

    # Load trained ddpm 
    ddpm = load_ddpm_from_configs_and_cpt(
        train_ddpm_cfg=run_params['training']['ddpm'],
        model_ddpm_cfg=run_params['model']['ddpm_unet'],
        cpt_fp=args.cpt_fp, 
        device=device
        )
    
    # Check that the images are properly normalized between 0 and 1
    assert rand_img.min() >= 0 and rand_img.max() <= 1, 'Image is not normalized'
    assert rand_seg.min() >= 0 and rand_seg.max() <= 1, 'Segmentation is not normalized'
    
    # Sampling from scratch
    # :===============================================================:
    print('Sampling with DDIM')
    ddim_samp_dir = os.path.join(out_dir, '2_ddim_sampling_x_cond_not_normalized')
    sampling_steps_old = ddpm.sampling_timesteps 
    ddpm.model.sampling_timesteps = ddim_steps
    sampled_img_progress = ddpm.ddim_sample(rand_seg, True)
    store_results_sampling(rand_img, rand_seg, sampled_img_progress, ddim_samp_dir)
    
    if not args.ddim_only:
        print('Sampling from noise with linear sampling')
        lin_samp_dir = os.path.join(out_dir, '1_linear_sampling')    
        sampled_img_progress = ddpm.p_sample_loop(rand_seg, return_all_timesteps=True)
        store_results_sampling(rand_img, rand_seg, sampled_img_progress, lin_samp_dir)

    

    
