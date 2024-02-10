import os
import sys
import yaml
import argparse
from tqdm import tqdm

import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..')))

from tta_uia_segmentation.src.models import ConditionalGaussianDiffusion
from utils import (
    fig_to_np,
    metric_preference,
    metrics_to_log_default,
    plot_denoising_progress,
    load_dataset_from_configs,
    load_ddpm_from_configs_and_cpt
)


# Default args
out_dir                 = '/scratch_net/biwidl319/jbermeo/results/ddpm/sanity_checks/'
exp_name                = 'default_exp_name/linear_denoising'
params_fp               = '/scratch_net/biwidl319/jbermeo/logs/brain/ddpm_old_exps/on_non_nn_normalized_imgs/cddpm/not_one_hot_128_base_filters/params.yaml'
cpt_fp                  = '/scratch_net/biwidl319/jbermeo/logs/brain/ddpm_old_exps/on_non_nn_normalized_imgs/cddpm/not_one_hot_128_base_filters/model-2-step_10000.pt'
noise_timesteps         = [50, 100, 250, 500, 750, 999]
ddim_steps              = 100
dataset                 = 'hcp_t1'
seed                    = 1234 
device                  = 'cpu' if not  torch.cuda.is_available() else 'cuda'
l_sampling_only         = False
split                   = 'train'
save_plot_every         = 50
measure_metrics_every   = 5


def get_cmd_args():
    parser = argparse.ArgumentParser(description="Check quality of DDPM sampling on a single image")

    parse_booleans = lambda x: x.lower() in ['true', '1']
    
    parser.add_argument('--out_dir', type=str, default=out_dir, help='Output directory')
    parser.add_argument('--exp_name', type=str, default=exp_name, help='Experiment name')
    parser.add_argument('--params_fp', type=str, default=params_fp, help='Path to the params file')
    parser.add_argument('--cpt_fp', type=str, default=cpt_fp, help='Path to the checkpoint file')
    parser.add_argument('--noise_timesteps', type=int, nargs='+', default=noise_timesteps, help='Timesteps for noise sampling')
    parser.add_argument('--ddim_steps', type=int, default=ddim_steps, help='Number of steps for DDIM sampling')
    parser.add_argument('--dataset', type=str, default=dataset, help='Dataset to use')
    parser.add_argument('--split', type=str, default=split, help='Split to use', choices=['train', 'val', 'test'])
    parser.add_argument('--seed', type=int, default=seed, help='Seed for reproducibility')
    parser.add_argument('--device', type=str, default=device, help='Device to use')
    parser.add_argument('--l_sampling_only', type=parse_booleans, default=l_sampling_only, help='Whether to only use linear sampling')
    parser.add_argument('--save_plot_every', type=int, default=save_plot_every, help='Save plot every n steps')
    parser.add_argument('--measure_metrics_every', type=int, default=measure_metrics_every, help='Measure metrics every n steps')
    
    return parser.parse_args()


def evaluate_linear_denoising(
    img: torch.Tensor,
    seg: torch.Tensor,
    t: int,
    ddpm: ConditionalGaussianDiffusion,
    plot_every: int = 50,
    measure_metrics_every: int = 5,
    metrics: dict = metrics_to_log_default,
    output_dir: str = None,
    device: torch.device = 'cpu'
) -> dict[str, dict]: 
    
    progress_dir = None
    
    if output_dir is not None:
        output_dir = os.path.join(output_dir, f'linear_denoising__noised_to_t_{t}')
        os.makedirs(output_dir, exist_ok=True)
        
        progress_dir = os.path.join(output_dir, 'progress')
        os.makedirs(progress_dir, exist_ok=True)
        
        metrics_dir = os.path.join(output_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
    # Normalize the image and the segmentation between -1 and 1
    img = ddpm.normalize(img)
    seg = ddpm.normalize(seg)
    
    assert img.shape == seg.shape, 'img and seg must have the same shape'
    assert img.min() >= -1 and img.max() <= 1, 'Image is not normalized between -1 and 1'
    assert seg.min() >= -1 and seg.max() <= 1, 'Segmentation is not normalized between -1 and 1'
    
    # Generate noised verion of the image
    batch_size = img.shape[0]
    t_tch = torch.full((batch_size,), t).to(device)
    noise = torch.randn_like(img).to(device)
    noised_img = ddpm.q_sample(img, t_tch, noise)

    metrics_logs = {k: list() for k in metrics.keys()}

    progress_figs = []    
    img_denoised_at_ti = noised_img
    for t_i in tqdm(reversed(range(0, t)), desc = 'sampling loop time step', total = t):
        img_denoised_at_ti_plus_1 = img_denoised_at_ti
        # Take denoising step
        img_denoised_at_ti, img_denoised_at_t0 = ddpm.p_sample(img_denoised_at_ti_plus_1, t_i, seg)
        
        # Get a reference of how the original image looks when noised at t_i
        t_i_tch = torch.full((batch_size,), t_i).to(device)
        noise_t_i = torch.randn_like(img).to(device)
        example_noised_img_at_ti = ddpm.q_sample(img, t_i_tch, noise_t_i)
        
        # Plot the progress
        if t_i % plot_every == 0 or t_i == 0:
            progress_fig = plot_denoising_progress(
                img, seg, noised_img, example_noised_img_at_ti, img_denoised_at_ti_plus_1, 
                img_denoised_at_ti, img_denoised_at_t0, t, t_i, 
                return_fig=True
                )
            
            if progress_dir is not None:
                progress_fig.savefig(os.path.join(progress_dir, f'progress_t_{t_i}.png'))
                progress_figs.append(Image.fromarray(fig_to_np(progress_fig)))
                plt.close(progress_fig)
        
        # Measure metrics
        if t_i % measure_metrics_every == 0 or t_i == 0:
            for metric_name, metric_log in metrics_logs.items():
                metric_log.append((
                    t - t_i,
                    metrics_to_log_default[metric_name](img, img_denoised_at_ti).item(),
                    metrics_to_log_default[metric_name](img, img_denoised_at_t0).item()
                ))
            
    # Save the progress as a gif
    if progress_dir is not None:
        progress_figs[0].save(os.path.join(progress_dir, 'progress.gif'), 
                              save_all=True, append_images=progress_figs[1:], 
                              optimize=False, duration=150, loop=0)
        
    # Save the metrics into a single dataframe
    metrics_dfs_list = []
    for metric_name, metric_log in metrics_logs.items():
        metric_log = np.array(metric_log)
        metrics_dfs_list.append(
            pd.DataFrame(metric_log, columns=['t_i', f'{metric_name}_t_i', f'{metric_name}_t_0']).set_index('t_i')
            )
    metrics_df = pd.concat(metrics_dfs_list, axis=1)
    
    # Plot and save the metrics over the denoising steps
    if metrics_dir is not None:
        for metric_name, metric_log in metrics_logs.items():
            metric_log = np.array(metric_log)
            
            # Plot each metric
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(metric_log[:, 0], metric_log[:, 1], label=f'{metric_name} (est. t_i) ')
            ax.plot(metric_log[:, 0], metric_log[:, 2], label=f'{metric_name} (est. t_0)')
            ax.legend()
            ax.set_title(f'{metric_name} over denoising progress')
            plt.savefig(os.path.join(metrics_dir, f'{metric_name}_over_time.png'))
        
        # Save all metrics to a file
        metrics_df.to_csv(os.path.join(metrics_dir, 'metrics.csv'))
                    
    return metrics_df


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
        
    # Check that the images are properly normalized between 0 and 1
    assert rand_img.min() >= 0 and rand_img.max() <= 1, 'Image is not normalized'
    assert rand_seg.min() >= 0 and rand_seg.max() <= 1, 'Segmentation is not normalized'

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
        
    # Evaluate the linear denoising for each t
    # :===============================================================:
    metrics_per_t = {}
    for t in args.noise_timesteps:
        metrics_per_t[t] = evaluate_linear_denoising(
            rand_img, rand_seg, t, ddpm, 
            output_dir=out_dir, plot_every=args.save_plot_every,
            measure_metrics_every=args.measure_metrics_every,
            device=device
        )
    
    for metric_name in metrics_to_log_default.keys():
        choose_best_fun = np.min if metric_preference[metric_name] == 'min' else np.max
        find_best_pos_fun = np.argmin if metric_preference[metric_name] == 'min' else np.argmax
        
        last_ti_values = [metrics_df[f'{metric_name}_t_i'].iloc[-1] for metrics_df in metrics_per_t.values()]
        last_t0_values = [metrics_df[f'{metric_name}_t_0'].iloc[-1] for metrics_df in metrics_per_t.values()]
        
        best_ti_values = [choose_best_fun(metrics_df[f'{metric_name}_t_i']) for metrics_df in metrics_per_t.values()]
        best_t0_values = [choose_best_fun(metrics_df[f'{metric_name}_t_0']) for metrics_df in metrics_per_t.values()]
        
        best_t_ti_est = [metrics_df.index[find_best_pos_fun(metrics_df[f'{metric_name}_t_i'])] for metrics_df in metrics_per_t.values()]
        best_t_to_est = [metrics_df.index[find_best_pos_fun(metrics_df[f'{metric_name}_t_0'])] for metrics_df in metrics_per_t.values()]
                
        # Write the results to a file
        with open(os.path.join(out_dir, f'{metric_name}_results.txt'), 'w') as f:
            f.write(f'Last t_i values: {last_ti_values}\n')
            f.write(f'Last t_i values mean: {np.mean(last_t0_values)} +- {np.std(last_t0_values)}\n\n')
            
            f.write(f'Last t_0 values: {last_t0_values}\n')
            f.write(f'Last t_0 values mean: {np.mean(last_t0_values)} +- {np.std(last_t0_values)}\n\n')
            
            f.write(f'Best t_i values: {best_ti_values}\n')
            f.write(f'Best t_0 values mean: {np.mean(best_t0_values)} +- {np.std(best_t0_values)}\n\n')
            
            f.write(f'Best t_0 values: {best_t0_values}\n')
            f.write(f'Best t_i values mean: {np.mean(best_ti_values)} +- {np.std(best_ti_values)}\n\n')
            
            f.write(f't of the best t_i estimates: {best_t_ti_est}\n')
            f.write(f't of the best t_0 estimates: {best_t_to_est}\n')
                
