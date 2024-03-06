import os
import sys
import yaml
import argparse
from tqdm import tqdm
from multiprocessing import cpu_count


import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchmetrics.functional.regression import mean_squared_error

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..')))

from tta_uia_segmentation.src.models import ConditionalGaussianDiffusion
from tta_uia_segmentation.src.dataset.dataset_h5_for_ddpm import get_datasets

from utils import (
    fig_to_np,
    metric_preferences,
    metrics_to_log_default,
    is_new_value_better,
    plot_denoising_progress,
    load_ddpm_from_configs_and_cpt
)

metrics_to_log_default = {
    **metrics_to_log_default,
    'MSE_v': mean_squared_error,
    'MSE_noise': mean_squared_error,
}

metric_preferences = {
    **metric_preferences,
    'MSE_v': 'min',
    'MSE_noise': 'min'
}

# :================================================================================================:
# Default args
out_dir                 = '/scratch_net/biwidl319/jbermeo/results/ddpm/sanity_checks/'
exp_name                = 'default_exp_name/linear_denoising'
params_fp               = '/scratch_net/biwidl319/jbermeo/logs/brain/ddpm_old_exps/on_non_nn_normalized_imgs/cddpm/not_one_hot_128_base_filters/params.yaml'
cpt_fp                  = '/scratch_net/biwidl319/jbermeo/logs/brain/ddpm_old_exps/on_non_nn_normalized_imgs/cddpm/not_one_hot_128_base_filters/model-2-step_10000.pt'
num_img_samples         = 50 
frac_sample_range       = (0.2, 0.8)
batch_size              = 4
num_workers             = cpu_count() // 2   
noise_timesteps         = [50, 100, 250, 500, 750, 999]
ddim_steps              = 100
dataset                 = 'hcp_t1'
seed                    = 1234 
device                  = 'cpu' if not  torch.cuda.is_available() else 'cuda'
l_sampling_only         = False
split                   = 'train'
# :================================================================================================:


def get_cmd_args():
    parser = argparse.ArgumentParser(description="Check quality of DDPM sampling on multiple images and noising timesteps.")

    parse_booleans = lambda x: x.lower() in ['true', '1']
    
    parser.add_argument('--out_dir', type=str, default=out_dir, help='Output directory')
    parser.add_argument('--exp_name', type=str, default=exp_name, help='Experiment name')
    parser.add_argument('--params_fp', type=str, default=params_fp, help='Path to the params file')
    parser.add_argument('--cpt_fp', type=str, default=cpt_fp, help='Path to the checkpoint file')
    
    parser.add_argument('--num_img_samples', type=int, default=num_img_samples, help='Number of images to sample from the dataset')
    parser.add_argument('--frac_sample_range', type=float, nargs=2, default=frac_sample_range, help='Fraction of the image to sample from')
    parser.add_argument('--noise_timesteps', type=int, nargs='+', default=noise_timesteps, help='Timesteps for noise sampling')
    parser.add_argument('--ddim_steps', type=int, default=ddim_steps, help='Number of steps for DDIM sampling')
    
    parser.add_argument('--dataset', type=str, default=dataset, help='Dataset to use')
    parser.add_argument('--split', type=str, default=split, help='Split to use', choices=['train', 'val', 'test'])
    
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=num_workers, help='Number of workers for the dataloader')
    parser.add_argument('--seed', type=int, default=seed, help='Seed for reproducibility')
    parser.add_argument('--device', type=str, default=device, help='Device to use')
    parser.add_argument('--l_sampling_only', type=parse_booleans, default=l_sampling_only, help='Whether to only use linear sampling')
    
    return parser.parse_args()


def evaluate_linear_denoising_multiple_imgs(
    img: torch.Tensor,
    seg: torch.Tensor,
    t: int,
    ddpm: ConditionalGaussianDiffusion,
    batch_idx: int,
    batch_size: int,
    metrics: dict = metrics_to_log_default,
    metric_preferences: dict = metric_preferences,
    device: torch.device = 'cpu'
) -> dict[str, dict]: 
    """
    Evaluates the linear denoising of an image batch for a given noise level t
    
    Return a dictionary with the metrics for each denoised image with the following structure:
    {
        img_idx: {
            'metric_name': {
                'last_value_t_i': float,
                'best_value_t_0': float,
                't_of_best_value_t_0': int
                },
            ...
        },
        ...
    }

    """
           
    # Normalize the image and the segmentation between -1 and 1
    img = ddpm.normalize(img).to(device)
    seg = ddpm.normalize(seg).to(device)
    
    assert img.shape == seg.shape, 'img and seg must have the same shape'
    assert img.min() >= -1 and img.max() <= 1, 'Image is not normalized between -1 and 1'
    assert seg.min() >= -1 and seg.max() <= 1, 'Segmentation is not normalized between -1 and 1'
    
    # Generate noised verion of the image
    batch_size = img.shape[0]
    t_tch = torch.full((batch_size,), t).to(device)
    noise_gt = torch.randn_like(img).to(device)
    noised_img = ddpm.q_sample(img, t_tch, noise_gt)

    # Initialize base case for metrics and denoising progress
    metrics_log_per_img = {batch_idx * batch_size + i: {} for i in range(batch_size)}
    metrics_log_per_img = {
        img_idx: { 
            metric_name: { 
                'last_value_t_i': None, 
                'best_value_t_0': np.inf if metric_preferences[metric_name] == 'min' else -np.inf, 
                't_of_best_value_t_0': None 
                }
            for metric_name in metrics.keys() 
            } 
        for img_idx in metrics_log_per_img.keys()
        }
    
    img_denoised_at_ti = noised_img
        
    for t_i in tqdm(reversed(range(0, t)), desc = 'sampling loop time step', total = t):
        img_denoised_at_ti_plus_1 = img_denoised_at_ti
        
        # Calculate GT values at time t_i
        t_i_tch = torch.full((batch_size,), t).to(device)
        v_t_i_gt = ddpm.predict_v(x_start=img, t=t_i_tch, noise=noise_gt)
        
        # Take denoising step
        img_denoised_at_ti, img_denoised_at_t0 = ddpm.p_sample(img_denoised_at_ti_plus_1, t_i, seg)

        # Get estimates for the starting noise and v
        noise_est = ddpm.predict_noise_from_start(x_t=img_denoised_at_ti_plus_1, t=t_i_tch, x0=img_denoised_at_t0)
        with torch.inference_mode():
            v_est = ddpm.model(img_denoised_at_ti_plus_1, t_i_tch, seg)
            
        # Measure metrics   
        for metric_name, metric_fun in metrics.items():
            if '_v' in metric_name:
                metric_values_t0 = [metric_fun(v_est[i, ...], v_t_i_gt[i, ...]).item() for i in range(batch_size)]
                metric_values_ti = [metric_values_t0[i] if t_i == 0 else None for i in range(batch_size)]
            elif '_noise' in metric_name:
                metric_values_t0 = [metric_fun(noise_est[i, ...], noise_gt[i, ...]).item() for i in range(batch_size)]
                metric_values_ti = [metric_values_t0[i] if t_i == 0 else None for i in range(batch_size)]                
            else:
                metric_values_ti = [metric_fun(img[i: i+1, ...], img_denoised_at_ti[i: i+1, ...]).item() 
                                    if t_i == 0 else None for i in range(batch_size)]
                metric_values_t0 = [metric_fun(img[i: i+1, ...], img_denoised_at_t0[i: i+1, ...]).item()
                                    for i in range(batch_size)]
                
            for idx, img_idx in enumerate(list(metrics_log_per_img.keys())):
                metrics_log_per_img[img_idx][metric_name]['last_value_t_i'] = metric_values_ti[idx]
                
                new_value_better = is_new_value_better(
                    old_value=metrics_log_per_img[img_idx][metric_name]['best_value_t_0'],
                    new_value=metric_values_t0[idx],
                    preference=metric_preferences[metric_name]
                    )
                
                if new_value_better:
                    metrics_log_per_img[img_idx][metric_name]['best_value_t_0'] = metric_values_t0[idx]
                    metrics_log_per_img[img_idx][metric_name]['t_of_best_value_t_0'] = t_i
                    
    return metrics_log_per_img


if __name__ == '__main__':
    args = get_cmd_args()
    
    # Create output dir
    out_dir = os.path.join(args.out_dir, args.exp_name, '3_performance_measured_over_multiple_volumes')
    os.makedirs(out_dir, exist_ok=True)
        
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    run_params = yaml.safe_load(open(args.params_fp, 'r'))
    dataset_params = run_params['dataset'][args.dataset]
    training_params = run_params['training']['ddpm']
    
    # Load the data
    # :===============================================================:
    (dataset,) = get_datasets(
        norm            = None,
        paths           = dataset_params['paths_processed'],
        paths_original  = dataset_params['paths_original'],
        paths_normalized_h5 = dataset_params['paths_normalized_with_nn'],
        splits          = [split],
        image_size      = training_params['image_size'],
        resolution_proc = dataset_params['resolution_proc'],
        dim_proc        = dataset_params['dim'],
        n_classes       = dataset_params['n_classes'],
        aug_params      = None,
        deformation     = None,
        load_original   = True,
        bg_suppression_opts = None,
        one_hot_encode  = False,
    )
    
    # Draw a sample of images from the dataset
    dataset = dataset.sample_slices(sample_size=args.num_img_samples, range_=args.frac_sample_range)   
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
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
        metrics_per_t[t] = {}
        
        for batch_idx, (img, seg) in enumerate(dataloader):
            # Get the metrics of the minbatch
            metrics_minibatch = evaluate_linear_denoising_multiple_imgs(
                img, seg, t, ddpm,
                batch_idx=batch_idx,
                batch_size=args.batch_size, 
                device=device
            )
            
            # Add the new metrics to the dictionary
            metrics_per_t[t] = {
                **metrics_minibatch,
                **metrics_per_t[t],
            }
        
        # Convert the metrics to a pandas dataframe
        metrics_t_df = pd.DataFrame.from_dict(metrics_per_t[t], orient='index')
        metrics_t_df = pd.concat([metrics_t_df[col_name].apply(pd.Series).add_prefix(f'{col_name}.') for col_name in metrics_t_df.columns], axis=1)
        metrics_t_df.sort_index().to_csv(os.path.join(out_dir, f'4_metrics_t_{t}.csv'))

        # Keep the mean and std of the metrics
        metrics_per_t[t] = metrics_t_df.agg(['mean', 'std'], axis=0).unstack()
        
    # Plot the metrics 
    summary_metrics_df = pd.DataFrame.from_dict(metrics_per_t, orient='index')
    summary_metrics_df.to_csv(os.path.join(out_dir, '1_summary_metrics.csv'))
    
    # Store the average of the metrics over the different t
    cols_to_summarize = ['last_value_t_i', 'best_value_t_0']
    cols_to_summarize = [col_tuple for col_tuple in summary_metrics_df.columns if col_tuple[0].split('.')[1] in cols_to_summarize]
    summary_over_t_df = pd.DataFrame.from_dict(
        {'mean': summary_metrics_df[cols_to_summarize].xs('mean', level=1, axis=1).mean(), 
         'std': summary_metrics_df[cols_to_summarize].xs('std', level=1, axis=1).agg(lambda x: np.linalg.norm(x)/np.sqrt(len(x)))},
        orient='index')
    summary_over_t_df.transpose().to_csv(os.path.join(out_dir, '0_summary_over_t.csv'))
    
    for metric_name in metrics_to_log_default.keys():
        # Plot the summary values for the last_ti_value
        jitter_range = 10
        jitter = np.random.uniform(-jitter_range, jitter_range, len(summary_metrics_df.index))
        col_name = f'{metric_name}.last_value_t_i'
        plt.errorbar(
            x=summary_metrics_df.index.values + jitter,
            y=summary_metrics_df[col_name]['mean'],
            yerr=summary_metrics_df[col_name]['std'], 
            fmt='-o',
            label='last_ti_value'
            )
        x_min, x_max = plt.xlim()
        # Plot the mean of the best_t0_value with confidence interval
        plt.axhline(y=summary_over_t_df[col_name]['mean'], color='r', linestyle='--', label='mean - last_value_t_i')
        # x_min, x_max = plt.xlim()
        # lower_est = summary_over_t_df[col_name]['mean'] - summary_over_t_df[col_name]['std']
        # upper_est = summary_over_t_df[col_name]['mean'] + summary_over_t_df[col_name]['std']
        # plt.fill_between(
        #     [x_min, x_max], 
        #     [lower_est] * 2, 
        #     [upper_est] * 2, 
        #     alpha=0.2, color='r'
        #     )
        
        # Plot the summary values for the best_t0_value
        jitter = np.random.uniform(-jitter_range, jitter_range, len(summary_metrics_df.index))
        # Set t with respect to 0 and not the noising step t
        col_name = f'{metric_name}.best_value_t_0'
        plt.errorbar(
            x=summary_metrics_df.index + jitter,
            y=summary_metrics_df[col_name]['mean'],
            yerr=summary_metrics_df[col_name]['std'], 
            fmt='-x',
            label='best_t0_value'
            )
        
        # Plot the mean of the best_t0_value with confidence interval
        plt.axhline(y=summary_metrics_df[col_name]['mean'].mean(), color='g', linestyle='--', label='mean - best_value_t_0')
        # x_min, x_max = plt.xlim()
        # lower_est = summary_over_t_df[col_name]['mean'] - summary_over_t_df[col_name]['std']
        # upper_est = summary_over_t_df[col_name]['mean'] + summary_over_t_df[col_name]['std']
        # plt.fill_between(
        #     [x_min, x_max], 
        #     [lower_est] * 2, 
        #     [upper_est] * 2, 
        #     alpha=0.2, color='r'
        #     )
        
        
        plt.title(f'{metric_name} summary')
        plt.xlabel('Noised level t')
        plt.ylabel(metric_name)
        plt.legend()
        plt.savefig(os.path.join(out_dir, f'2_{metric_name}_summary.png'))    
        plt.close()
        
        # Set t with respect to 0 and not the noising step t
        y = summary_metrics_df.index - summary_metrics_df[f'{metric_name}.t_of_best_value_t_0']['mean'] - 1
        plt.errorbar(
            x=summary_metrics_df.index,
            y=y,
            yerr=summary_metrics_df[f'{metric_name}.t_of_best_value_t_0']['std'], 
            fmt='x',
            label='best_t0_value'
            )
        
        plt.title(f'{metric_name} summary of at which t was the best estimate t_0 found')
        plt.xlabel('Noised level t')
        plt.ylabel('t_of_best_value_t_0')
        plt.savefig(os.path.join(out_dir, f'3_{metric_name}_t_of_best_value_t_0_summary.png'))
        plt.close()