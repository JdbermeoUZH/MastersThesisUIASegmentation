import os
import sys
import yaml
import argparse
from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
from typing import Literal

import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..')))

from tta_uia_segmentation.src.dataset.dataset_in_memory_for_ddpm import get_datasets
from tta_uia_segmentation.src.models.io import load_norm_from_configs_and_cpt, load_cddpm_from_configs_and_cpt
from tta_uia_segmentation.src.utils.io import load_config, dump_config


# Default args
# :===============================================================:
out_dir                 = '/scratch_net/biwidl319/jbermeo/results/wmh/ddpm/'
exp_name                = 'batch_size_130_dim_64_dim_mults_1_2_2_2_cond_by_concatenation_with_unconditional_training_rate_0.75'
ddpm_dir                = '/scratch_net/biwidl319/jbermeo/logs/brain/ddpm/hcp_t1/normalized_imgs/3x3_norm_filters/4_26/batch_size_130_dim_64_dim_mults_1_2_2_2_cond_by_concatenation_with_unconditional_training_rate_0.75' 
cpt_fn                  = 'model-9.pt' #'/scratch_net/biwidl319/jbermeo/logs/wmh/ddpm/normalized_imgs/no_bg_supp_3x3_conv/3_19/batch_size_128_cond_by_concat_multi_gpu/model-19.pt'
batch_size              = 16   
frac_sample_range       = (0.25, 0.75) # (0.3, 0.6):= usually has labels, (0.6, 0.9):= usually has no labels
dataset_sd              = 'hcp_t1' # umc
split_sd                = 'train'
dataset_td              = 'hcp_t1' #'nuhs'
split_td                = 'val'
seed                    = 1234 
device                  = 'cpu' if not  torch.cuda.is_available() else 'cuda'
mismatch_mode           = 'none' #'same_patient_very_different_labels'  # 'same_patient_very_different_labels', 'same_patient_similar_labels', 'different_patient_similar_labels', 'none'
n_mismatches            = 10
num_t_samples_per_img   = 20 # 20
num_iterations          = 100 # 500
with_augmentation       = False
weights_cond_free_guidance = [0.1, 1, 10, 30, 100]
min_max_quantile        = (0.001, 0.999)
# :===============================================================:

mismatch_args_per_dataset_type = {
    'wmh': {
        'same_patient_very_different_labels': {
            'min_dist_z_frac': 0.2, 'max_dice_score_threshold': 0.3},
        'same_patient_similar_labels': {
            'max_dist_z_frac': 0.1, 'min_dice_score_threshold': 0.5},
        'different_patient_similar_labels': {
            'max_dist_z_frac': 0.5, 'min_dice_score_threshold': 0.4}
    },
    'brain': {
        'same_patient_very_different_labels': {
            'min_dist_z_frac': 0.2, 'max_dice_score_threshold': 0.3},
        'same_patient_similar_labels': {
            'max_dist_z_frac': 0.2, 'min_dice_score_threshold': 0.7},
        'different_patient_similar_labels': {
            'max_dist_z_frac': 0.2, 'min_dice_score_threshold': 0.55}
    }
}

map_dataset_to_dataset_type = {
    'umc': 'wmh',
    'umc_w_synthseg_labels': 'wmh',
    'nuhs': 'wmh',
    'nuhs_w_synthseg_labels': 'wmh',
    'vu': 'wmh',
    'vu_w_synthseg_labels': 'wmh',
    'hcp_t1': 'brain',
    'hcp_t2': 'brain',
    'abide_caltech': 'brain',
}


def check_btw_0_1(*args: torch.Tensor, margin_error = 1e-2):
    for tensor in args:
        assert tensor.min() >= 0 - margin_error and tensor.max() <= 1 + margin_error, 'tensor values should be between 0 and 1'    


def rescale_volume(
        x: torch.Tensor,
        rescale_factor: tuple[float, float, float],
        how: Literal['up', 'down'] = 'down',
        return_dchw: bool = True
    ) -> torch.Tensor:
        # By define the recale factor as the proportion between
        #  the Atlas or DAE volumes and the processed volumes 
        rescale_factor = list((1 / np.array(rescale_factor))) \
            if how == 'up' else rescale_factor
            
        x = x.permute(1, 0, 2, 3).unsqueeze(0)
        x = F.interpolate(x, scale_factor=rescale_factor, mode='trilinear')
        
        if return_dchw:
            x = x.squeeze(0).permute(1, 0, 2, 3)
        
        return x


def get_cmd_args():
    parser = argparse.ArgumentParser(description="Check how the DDPM loss behaves for different timesteps and shifts at the images or labels.")
    parse_bool = lambda x: x.lower() in ['true', '1']
    
    parser.add_argument('--out_dir', type=str, default=out_dir, help='Output directory')
    parser.add_argument('--exp_name', type=str, default=exp_name, help='Experiment name')
    parser.add_argument('--ddpm_dir', type=str, default=ddpm_dir, help='Path to the params file')
    parser.add_argument('--cpt_fn', type=str, default=cpt_fn, help='Path to the checkpoint file')
    
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size')
    parser.add_argument('--num_iterations', type=int, default=num_iterations, help='Number of images to sample from the dataset')
    parser.add_argument('--n_mismatches', type=int, default=n_mismatches, help='Number of mismatches to sample for the label mismatch mode')
    parser.add_argument('--num_t_samples_per_img', type=int, default=num_t_samples_per_img, help='Number of timesteps to sample for each image')
    parser.add_argument('--frac_sample_range', type=float, nargs=2, default=frac_sample_range, help='Fraction of the image to sample from')
    parser.add_argument('--mismatch_mode', type=str, default=mismatch_mode, help='Mode for label mismatch', choices=['same_patient_very_different_labels', 'same_patient_similar_labels', 'different_patient_similar_labels', 'none'])
    parser.add_argument('--with_augmentation', type=parse_bool, default=with_augmentation, help='Whether to use data augmentation')
    parser.add_argument('--output_dir_suffix', type=str, default='', help='Suffix to add to the output directory')
      
    parser.add_argument('--dataset_sd', type=str, default=dataset_sd, help='Source domain dataset to use as baseline')
    parser.add_argument('--split_sd', type=str, default=split_sd, help='Split to use of the source domain', choices=['train', 'val', 'test'])
    parser.add_argument('--dataset_td', type=str, default=dataset_td, help='Target domain dataset to compare against')
    parser.add_argument('--split_td', type=str, default=split_sd, help='Split to use of the target domain', choices=['train', 'val', 'test'])
    parser.add_argument('--weights_cond_free_guidance', type=float, nargs='+', default=weights_cond_free_guidance, help='Weights for the conditional free guidance')
    parser.add_argument('--seed', type=int, default=seed, help='Seed for reproducibility')
    parser.add_argument('--device', type=str, default=device, help='Device to use')

    return parser.parse_args()


def plot_losses(
    baseline_info_df, shift_info_df,
    baseline_label, shift_label,
    output_path,
    title='DDPM Loss vs noise t', 
    figsize=(15, 5)
    ):
    baseline_summary_df = baseline_info_df.describe()
    shift_summary_df = shift_info_df.describe()
    
    plt.figure(figsize=figsize)
    plt.title(title)

    # Baseline
    # =========
    plt.errorbar(
        x=baseline_summary_df.loc['mean'].index,
        y=baseline_summary_df.loc['mean'],
        yerr=baseline_summary_df.loc['std'], 
        fmt='-o',
        label=baseline_label
        )
    # Plot the mean loss over all t
    print(f'Mean loss over all t - {baseline_label}: {baseline_summary_df.loc["mean"].mean()}')
    plt.axhline(y=baseline_summary_df.loc['mean'].mean(),
            color='b', linestyle='--', label=f'mean loss - {baseline_label}')
    
    # Shift
    # =========
    plt.errorbar(
        x=shift_summary_df.loc['mean'].index,
        y=shift_summary_df.loc['mean'],
        yerr=shift_summary_df.loc['std'], 
        fmt='-o',
        label=shift_label
        )
    
    # Plot the mean loss over all t
    print(f'Mean loss over all t - {shift_label}: {shift_summary_df.loc["mean"].mean()}')
    plt.axhline(y=shift_summary_df.loc['mean'].mean(),
            color='orange', linestyle='--', label=f'mean loss - {shift_label}')

    # Display legends
    plt.legend()    
    plt.yscale('log')

    plt.savefig(output_path)


if __name__ == '__main__':
    args = get_cmd_args()
    
    # Print the arguments
    pprint(args.__dict__)
    
    # Create output dir
    cpt_name = 'cpt_' + args.cpt_fn.split('.')[0].split('-')[-1]
    out_dir = os.path.join(args.out_dir, args.exp_name, cpt_name,
                           f'{args.dataset_sd}_{args.split_sd}',
                           f'{args.dataset_td}_{args.split_td}',
                           args.mismatch_mode + args.output_dir_suffix
                           )
    os.makedirs(out_dir, exist_ok=True)
    dump_config(os.path.join(out_dir, 'params.yaml'), args.__dict__)
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load
    run_params = yaml.safe_load(open(os.path.join(args.ddpm_dir, 'params.yaml'), 'r'))
    
    dataset_params_sd = run_params['dataset'][args.dataset_sd]
    dataset_params_td = run_params['dataset'][args.dataset_td]
    train_ddpm_cfg = run_params['training']['ddpm']
    model_ddpm_cfg = run_params['model']['ddpm_unet']
    
    # Load the normalization model 
    norm_dir = train_ddpm_cfg['norm_dir']
    params_norm = load_config(os.path.join(train_ddpm_cfg['norm_dir'], 'params.yaml'))

    model_params_norm = params_norm['model']['normalization_2D']
    train_params_norm = params_norm['training']
    
    norm = load_norm_from_configs_and_cpt(
        model_params_norm=model_params_norm,
        cpt_fp=os.path.join(norm_dir, train_params_norm['checkpoint_best']),
        device=device    
    )     
    
    # Load the data
    # :===============================================================:    
    n_classes           = dataset_params_sd['n_classes']

    (dataset_sd,) = get_datasets(
        splits          = [args.split_sd],
        norm            = norm,
        paths           = dataset_params_sd['paths_processed'],
        paths_original  = dataset_params_sd['paths_original'], 
        paths_normalized_h5 = None,
        use_original_imgs = train_ddpm_cfg['use_original_imgs'],
        one_hot_encode  = train_ddpm_cfg['one_hot_encode'],
        normalize       = train_ddpm_cfg['normalize'],
        image_size      = train_ddpm_cfg['image_size'],
        resolution_proc = dataset_params_sd['resolution_proc'],
        rescale_factor  = train_ddpm_cfg['rescale_factor'],
        dim_proc        = dataset_params_sd['dim'],
        n_classes       = dataset_params_sd['n_classes'],
        aug_params      = train_ddpm_cfg['augmentation'] if args.with_augmentation else None,
        deformation     = None,
        load_original   = True,
        bg_suppression_opts = None,
        norm_q_range    = min_max_quantile
    )
    
    (dataset_td,) = get_datasets(
        splits          = [args.split_td],
        norm            = norm,
        paths           = dataset_params_td['paths_processed'],
        paths_original  = dataset_params_td['paths_original'], 
        paths_normalized_h5 = None,
        use_original_imgs = train_ddpm_cfg['use_original_imgs'],
        one_hot_encode  = train_ddpm_cfg['one_hot_encode'],
        normalize       = train_ddpm_cfg['normalize'],
        norm_q_range    = train_ddpm_cfg['norm_q_range'],
        image_size      = train_ddpm_cfg['image_size'],
        resolution_proc = dataset_params_td['resolution_proc'],
        rescale_factor  = train_ddpm_cfg['rescale_factor'],
        dim_proc        = dataset_params_td['dim'],
        n_classes       = dataset_params_td['n_classes'],
        aug_params      = train_ddpm_cfg['augmentation'] if args.with_augmentation else None,
        deformation     = None,
        load_original   = True,
        bg_suppression_opts = None,
    )
       
    # Load trained ddpm
    # :===============================================================:
    timesteps           = train_ddpm_cfg['timesteps']
    img_size            = dataset_sd.image_size[-1]
    
    ddpm = load_cddpm_from_configs_and_cpt(
        train_ddpm_cfg=train_ddpm_cfg,
        model_ddpm_cfg=model_ddpm_cfg,
        n_classes=n_classes,  
        cpt_fp=os.path.join(args.ddpm_dir, args.cpt_fn),
        device=device
        )
    ddpm.eval()
    
    rescale_factor = np.array([1, ddpm.image_size , ddpm.image_size]) / np.array(train_ddpm_cfg['image_size'])
    #breakpoint()
    
    # Get the relevant mismatch args
    if args.mismatch_mode != 'none':
        dataset_type = map_dataset_to_dataset_type[args.dataset_td]
        mismatch_args = mismatch_args_per_dataset_type[dataset_type]
        mismatch_args = mismatch_args[args.mismatch_mode]
    else:
        mismatch_args = {}
    
    # Start the evaluation
    # :===============================================================:
    noise_timesteps = np.linspace(0, timesteps - 1, args.num_t_samples_per_img, dtype=int)
    weights_cond_free_guidance = args.weights_cond_free_guidance
    metrics_per_t_sd = {
        'conditional': defaultdict(list),
        'unconditional': defaultdict(list),
        **{f'conditional_w_cfg_{w}': defaultdict(list) for w in weights_cond_free_guidance},
    } 
    metrics_per_t_td = {
        'conditional': defaultdict(list),
        'unconditional': defaultdict(list),
        **{f'conditional_w_cfg_{w}': defaultdict(list) for w in weights_cond_free_guidance},
    } 
    
    vol_depth = dataset_sd.dim_proc[0]     
    no_mismaches_found_count = 0   
    
    for i in tqdm(range(args.num_iterations)):
        # Sample random image from a random volume in a given range
        vol_idx_sd = random.randint(0, dataset_sd.num_vols - 1)
        vol_idx_td = vol_idx_sd if split_sd == split_td and dataset_sd.num_vols == dataset_td.num_vols  \
            else random.randint(0, dataset_td.num_vols - 1)
        slice_idx = random.randint(int(args.frac_sample_range[0] * vol_depth),
                                   min(int(args.frac_sample_range[1] * vol_depth), vol_depth - 1))
        
        img_sd, seg_sd, _ = dataset_sd[dataset_sd.vol_and_z_idx_to_idx(vol_idx_sd, slice_idx)]    
        img_sd = img_sd.to(device)
        seg_sd = seg_sd.to(device)
        seg_uncond_sd = ddpm._generate_unconditional_x_cond(batch_size=1, device=seg_sd.device)

        # Target Domain images and labels
        img_td, seg_td, _ = dataset_td[dataset_td.vol_and_z_idx_to_idx(vol_idx_td, slice_idx)]
        img_td = img_td.to(device)
        
        check_btw_0_1(img_sd, seg_sd, seg_uncond_sd, img_td)
        
        # Normalize the source images between -1 and 1
        img_sd = ddpm.normalize(img_sd.unsqueeze(0))
        seg_sd = ddpm.normalize(seg_sd.unsqueeze(0))
        seg_uncond_sd = ddpm.normalize(seg_uncond_sd)
        img_td = ddpm.normalize(img_td.unsqueeze(0))
        
        # Iterate over volumes and calculate the denoising performance     
        #   on the source and target domains
        # :===============================================================:
        mismatch_dataset = dataset_td
        
        mismatch_ds = mismatch_dataset.get_related_images(
            vol_idx=vol_idx_td,
            z_idx=slice_idx,
            mode=args.mismatch_mode,
            n=args.n_mismatches,
            **mismatch_args
        )
        
        if len(mismatch_ds) == 0 and args.mismatch_mode != 'none':
            print('Warning: No mismatching images found')
            no_mismaches_found_count += 1
            continue
        
        mismatch_dl = DataLoader(
            mismatch_ds, 
            batch_size=args.batch_size if args.mismatch_mode != 'none' else 1,
            shuffle=False
            )
        
        # Evaluate the linear denoising for each t
        # :===============================================================:
        for t in noise_timesteps:
                        
            # Calculate losses in source domain       
            t_sd = torch.full((1,), t, device=device).long()
            
            # Rescale the images and labels if needed
            if all(rescale_factor != [1, 1, 1]):
                #breakpoint()
                img_sd = rescale_volume(img_sd, rescale_factor, how='down')
                seg_sd = rescale_volume(seg_sd, rescale_factor, how='down').round()
                seg_uncond_sd = rescale_volume(seg_uncond_sd, rescale_factor, how='down').round()
                 
            with torch.no_grad():
                ddpm_loss_sample_sd_cond = ddpm.p_losses_conditioned_on_img(
                    img_sd, t_sd, seg_sd)
                ddpm_loss_sample_sd_uncond = ddpm.p_losses_conditioned_on_img(
                    img_sd, t_sd, seg_uncond_sd) if ddpm.also_unconditional else torch.tensor(0)
                
                for w_cfg in weights_cond_free_guidance:
                    metrics_per_t_sd[f'conditional_w_cfg_{w_cfg}'][t].append(
                        ddpm.p_losses_conditioned_on_img(
                            img_sd, t_sd, seg_sd, w_clf_free=w_cfg).item()
                    )
                    
            metrics_per_t_sd['conditional'][t].append(ddpm_loss_sample_sd_cond.item())
            metrics_per_t_sd['unconditional'][t].append(ddpm_loss_sample_sd_uncond.item())

            # Calculate loss in target domain and/or label mismatches
            ddpm_loss_sample_td_cond = 0
            ddpm_loss_sample_td_uncond = 0
            for _, seg_mismatch, _ in mismatch_dl:    
                seg_mismatch = seg_mismatch.to(device)
                seg_mismatch_uncond = ddpm._generate_unconditional_x_cond(seg_mismatch.shape[0], device=seg_sd.device)

                check_btw_0_1(seg_mismatch, seg_mismatch_uncond)
                seg_mismatch = ddpm.normalize(seg_mismatch)
                seg_mismatch_uncond = ddpm.normalize(seg_mismatch_uncond)
                
                # Make a bacth of the test image
                b = seg_mismatch.shape[0]                
                img_mismatch = img_td
                img_mismatch_b = img_mismatch.repeat(b, 1, 1, 1)

                # Rescale the images and labels if needed
                if all(rescale_factor != [1, 1, 1]):
                    img_mismatch_b = rescale_volume(img_mismatch_b, rescale_factor, how='down').round()
                    seg_mismatch = rescale_volume(seg_mismatch, rescale_factor, how='down').round()
                
                # Calculate the ddpm_loss
                t_tch = torch.full((b,), t, device=device).long()

                with torch.no_grad():                
                    ddpm_loss_sample_td_cond += (1 / len(mismatch_dl)) * \
                        ddpm.p_losses_conditioned_on_img(img_mismatch_b, t_tch, seg_mismatch)
                    ddpm_loss_sample_td_uncond += (1 / len(mismatch_dl)) * \
                        ddpm.p_losses_conditioned_on_img(img_mismatch_b, t_tch, seg_mismatch_uncond) if ddpm.also_unconditional else torch.tensor(0)
                        
                    for w_cfg in weights_cond_free_guidance:
                        metrics_per_t_td[f'conditional_w_cfg_{w_cfg}'][t].append(
                            ddpm.p_losses_conditioned_on_img(
                                img_mismatch_b, t_tch, seg_mismatch, w_clf_free=w_cfg).item()
                        )

            metrics_per_t_td['conditional'][t].append(ddpm_loss_sample_td_cond.item())
            metrics_per_t_td['unconditional'][t].append(ddpm_loss_sample_td_uncond.item())
                    
    print(f'No mismatches found: {no_mismaches_found_count}, '
          f'that is {no_mismaches_found_count/args.num_iterations:.2%} % of iterations')
    
    # Save the results
    # :===============================================================:
    for loss_type in metrics_per_t_sd:
        if not ddpm.also_unconditional and loss_type != 'conditional':
            continue
        
        loss_per_t_sd_df = pd.DataFrame(metrics_per_t_sd[loss_type])
        loss_per_t_td_df = pd.DataFrame(metrics_per_t_td[loss_type])
        
        sd_label = f'{args.dataset_sd}__{args.split_sd}__{loss_type}'
        td_label = f'{args.dataset_td}__{args.split_td}__{loss_type}'
        td_label += f"_mismatch_{args.mismatch_mode}" if args.mismatch_mode != 'none' else ''
        
        # Save csv
        loss_per_t_sd_df.to_csv(os.path.join(out_dir, f'loss_{loss_type}_per_t_sd_{sd_label}.csv'))
        loss_per_t_td_df.to_csv(os.path.join(out_dir, f'loss_{loss_type}_per_t_td_{td_label}.csv'))
    
        # Save plot
        plot_losses(
            loss_per_t_sd_df, loss_per_t_td_df,
            baseline_label=sd_label, 
            shift_label=td_label,
            title=f'DDPM Loss ({loss_type}) vs noise t',
            output_path=os.path.join(out_dir, f'loss_per_t_{loss_type}_{args.mismatch_mode}.png')
        )
        
        # Save average losses to a text file
        with open(os.path.join(out_dir, f'average_losses_{loss_type}_{args.mismatch_mode}.txt'), 'w') as f:
            f.write(f'Average loss over all timesteps - {sd_label}: {loss_per_t_sd_df.mean().mean()}\n')
            f.write(f'Average loss over all timesteps - {td_label}: {loss_per_t_td_df.mean().mean()}')
    
    print('Done!')
