import os
import sys
import yaml
import argparse
from tqdm import tqdm
from collections import defaultdict

import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..')))

from tta_uia_segmentation.src.dataset.dataset_in_memory_for_ddpm import get_datasets

from utils import load_ddpm_from_configs_and_cpt


# Default args
# :===============================================================:
out_dir                 = '/scratch_net/biwidl319/jbermeo/results/ddpm/sanity_checks/'
exp_name                = 'default_exp_name'
params_fp               = '/scratch_net/biwidl319/jbermeo/logs/brain/ddpm/not_one_hot_64_base_filters_with_aug_except_noise/params.yaml'
cpt_fp                  = '/scratch_net/biwidl319/jbermeo/logs/brain/ddpm/not_one_hot_64_base_filters_with_aug_except_noise/model-9.pt'
batch_size              = 4   
frac_sample_range       = (0.0, 1.0)
dataset                 = 'hcp_t2'
seed                    = 1234 
device                  = 'cpu' if not  torch.cuda.is_available() else 'cuda'
split                   = 'train'
mismatch_mode           = 'none' #'same_patient_very_different_labels'  # 'same_patient_very_different_labels', 'same_patient_similar_labels', 'different_patient_similar_labels', 'none'
n_mismatches            = 6
num_t_samples_per_img   = 20
num_iterations          = 500
# :===============================================================:


def get_cmd_args():
    parser = argparse.ArgumentParser(description="Check how the DDPM loss behaves for different timesteps and shifts at the images or labels.")
    
    parser.add_argument('--out_dir', type=str, default=out_dir, help='Output directory')
    parser.add_argument('--exp_name', type=str, default=exp_name, help='Experiment name')
    parser.add_argument('--params_fp', type=str, default=params_fp, help='Path to the params file')
    parser.add_argument('--cpt_fp', type=str, default=cpt_fp, help='Path to the checkpoint file')
    
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size')
    parser.add_argument('--num_iterations', type=int, default=num_iterations, help='Number of images to sample from the dataset')
    parser.add_argument('--n_mismatches', type=int, default=n_mismatches, help='Number of mismatches to sample for the label mismatch mode')
    parser.add_argument('--num_t_samples_per_img', type=int, default=num_t_samples_per_img, help='Number of timesteps to sample for each image')
    parser.add_argument('--frac_sample_range', type=float, nargs=2, default=frac_sample_range, help='Fraction of the image to sample from')
    parser.add_argument('--mismatch_mode', type=str, default=mismatch_mode, help='Mode for label mismatch', choices=['same_patient_very_different_labels', 'same_patient_similar_labels', 'different_patient_similar_labels', 'none'])
    
    parser.add_argument('--dataset', type=str, default=dataset, help='Dataset to use')
    parser.add_argument('--split', type=str, default=split, help='Split to use', choices=['train', 'val', 'test'])
    parser.add_argument('--seed', type=int, default=seed, help='Seed for reproducibility')
    parser.add_argument('--device', type=str, default=device, help='Device to use')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_cmd_args()
    
    # Create output dir
    out_dir = os.path.join(args.out_dir, args.exp_name, 
                           '4_ddpm_loss_multiple_t_and_imgs',
                           args.dataset, args.split,
                           args.mismatch_mode
                           )
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
        splits          = [args.split],
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
    
    # Start the evaluation
    # :===============================================================:
    noise_timesteps = np.linspace(0, timesteps - 1, args.num_t_samples_per_img, dtype=int)
    metrics_per_t = defaultdict(list)
            
    for i in tqdm(range(args.num_iterations)):
        # Sample random image from a random volume in a given range
        img_size = dataset.image_size[-1]   # DHW
        vol_idx = random.randint(0, dataset.num_vols - 1)
        slice_idx = random.randint(int(args.frac_sample_range[0] * img_size),
                                   int(args.frac_sample_range[1] * img_size))
        img, _ = dataset[dataset.vol_and_z_idx_to_idx(vol_idx, slice_idx)]    
        img = img.to(device)
        
        # Iterate over volumes and calculate the denoising performanc     
        # Get the mismatching images
        # :===============================================================:
        mismatch_ds = dataset.get_related_images(
            vol_idx=vol_idx,
            z_idx=slice_idx,
            mode=args.mismatch_mode,
            n=args.n_mismatches
        )
        
        mismatch_dl = DataLoader(
            mismatch_ds, 
            batch_size=args.batch_size if args.mismatch_mode != 'none' else 1,
            shuffle=False
            )
        
        # Evaluate the linear denoising for each t
        # :===============================================================:
        for t in noise_timesteps:
            ddpm_loss_sample = 0
            
            for _, seg_mismatch in mismatch_dl:    
                seg_mismatch = seg_mismatch.to(device)
                b = seg_mismatch.shape[0]
                
                img_b = img.repeat(b, 1, 1, 1)
                
                # Calculate the ddpm_loss
                t_tch = torch.full((b,), t, device=device).long()
                img_b = ddpm.normalize(img_b)
                seg_mismatch = ddpm.normalize(seg_mismatch)
                
                ddpm_loss_sample += (1 / len(mismatch_dl)) * ddpm.p_losses_conditioned_on_img(img_b, t_tch, seg_mismatch)
            
            metrics_per_t[t].append(ddpm_loss_sample.item())
            
    # Save the results
    # :===============================================================:
    loss_per_t_df = pd.DataFrame(metrics_per_t)
    loss_per_t_df.to_csv(os.path.join(out_dir, f'loss_per_t_{args.mismatch_mode}.csv'))
    
    # Save plot
    fig, ax = plt.subplots()
    summary_metrics_df = loss_per_t_df.describe()
    plt.errorbar(
            x=summary_metrics_df.loc['mean'].index,
            y=summary_metrics_df.loc['mean'],
            yerr=summary_metrics_df.loc['std'], 
            fmt='-o',
            label='last_ti_value'
            )
    plt.yscale('log')
    
    # Plot the mean of the best_t0_value with confidence interval
    plt.axhline(y=summary_metrics_df.loc['mean'].mean(),
                color='r', linestyle='--', label='mean loss')

    plt.savefig(os.path.join(out_dir, f'loss_per_t_{args.mismatch_mode}.png'))