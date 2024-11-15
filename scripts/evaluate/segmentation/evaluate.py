import os
import sys
import argparse
from typing import Optional, Union

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Subset
from torch.nn import functional as F
from torch.utils.data import  TensorDataset, DataLoader


sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

from tta_uia_segmentation.src.dataset.dataset_in_memory import get_datasets, DatasetInMemory
from tta_uia_segmentation.src.models.io import define_and_possibly_load_norm_seg
from tta_uia_segmentation.src.models.normalization import background_suppression
from tta_uia_segmentation.src.utils.io import (
    load_config, dump_config, print_config, save_nii_image,
    rewrite_config_arguments)
from tta_uia_segmentation.src.utils.utils import seed_everything, define_device
from tta_uia_segmentation.src.utils.loss import dice_score, onehot_to_class
from tta_uia_segmentation.src.utils.visualization import export_images


metrics = {
    'dice': lambda y_pred, y_true: dice_score(y_pred, y_true, soft=False, reduction='none', smooth=1e-5)
}


def preprocess_cmd_args() -> argparse.Namespace:
    """_
    Parse command line arguments and return them as a Namespace object.
    
    All options specified will overwrite whaterver is specified in the config files.
    
    """
    parser = argparse.ArgumentParser(description="Measure Segmentation Performance")
    
    parser.add_argument('dataset_config_file', type=str, help='Path to yaml config file with parameters that define the dataset.')
    parser.add_argument('evaluation_config_file', type=str, help='Path to yaml config file with parameters that define the evaluation.')
    
    
    # Evaluation parameters
    # ---------------------------------------------------:
    parser.add_argument('--logdir', type=str, help='Path to directory where logs are saved. Default: "logs"')  
    parser.add_argument('--seg_dir', type=str, help='Path to directory where segmentation model is saved. Default: "logs"')
    parser.add_argument('--cpt_fp', type=str, help='Path to checkpoint file of the segmentation model. Default: "logs"')
    parser.add_argument('--seed', type=int, help='Seed for reproducibility. Default: 0')
    parser.add_argument('--device', type=str, help='Device to use for training. Default: "cuda"')
    
    # Dataset to evaluate the model on
    # ---------------------------------------------------:
    parser.add_argument('--dataset', type=str, help='Name of dataset to use for tta. Default: USZ')
    parser.add_argument('--split', type=str, help='Name of split to use for tta. Default: test')
    parser.add_argument('--vol_idx', type=str, help='Index of volume to, can allso be set to "all"')
    parser.add_argument('--n_classes', type=int, help='Number of classes in dataset. Default: 21')
    parser.add_argument('--classes_of_interest', type=int, nargs='+', help='Classes to consider for evaluation. Default: [0, 1]')
    parser.add_argument('--image_size', type=int, nargs='+', help='Size of images in dataset. Default: [560, 640, 160]')
    parser.add_argument('--resolution_proc', type=float, nargs='+', help='Resolution of images in dataset. Default: [0.3, 0.3, 0.6]')
    parser.add_argument('--rescale_factor', type=float, help='Rescale factor for images in dataset. Default: None')
    parser.add_argument('--evaluate_also_bg_supp', type=lambda s: s.strip().lower() == 'true')
    parser.add_argument('--save_nii', type=lambda s: s.strip().lower() == 'true')
    args = parser.parse_args()
    
    return args


def get_configuration_arguments() -> tuple[dict, dict]:
    args = preprocess_cmd_args()
    
    dataset_config = load_config(args.dataset_config_file)
    dataset_config = rewrite_config_arguments(dataset_config, args, 'dataset')
    
    evaluation_config = load_config(args.evaluation_config_file)
    evaluation_config = rewrite_config_arguments(evaluation_config, args, 'evaluation')
    
    return dataset_config, evaluation_config

@torch.inference_mode()
def test_volume(
    norm: torch.nn.Module,
    seg: torch.nn.Module,
    volume_dataset: DatasetInMemory,
    dataset_name: str,
    n_classes: int,
    index: int, 
    batch_size: int,
    num_workers: int,
    appendix='',
    metrics: dict = metrics,
    bg_suppression_opts: Optional[dict] = None,
    seg_with_bg_supp: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    logdir: Optional[str] = None,
    export_seg_imgs: bool = False,
    evaluate_also_bg_supp: bool = True,
    classes_of_interest: Optional[list[int]] = None,
    save_nii: bool = False,
):
    # Get original images (they are still downscaled)
    # :=========================================================================:
    x_original, y_original, bg = volume_dataset.dataset.get_original_images(index)
    _, C, D, H, W = y_original.shape  # xyz = HWD

    x_ = x_original.permute(0, 2, 3, 1).unsqueeze(0)  # NCHWD (= NCxyz)
    y_ = y_original.permute(0, 1, 3, 4, 2)  # NCHWD
    bg_ = torch.from_numpy(bg).permute(1, 2, 0).unsqueeze(0).unsqueeze(0)  # NCHWD

    # Rescale original images to their original sizes 
    original_pix_size = volume_dataset.dataset.pix_size_original[:, index]
    target_pix_size = volume_dataset.dataset.resolution_proc  # xyz
    scale_factor = original_pix_size / target_pix_size
    scale_factor[-1] = 1

    y_ = y_.float()
    bg_ = bg_.float()

    output_size = (y_.shape[2:] * scale_factor).round().astype(int).tolist()
    x_ = F.interpolate(x_, size=output_size, mode='trilinear')
    y_ = F.interpolate(y_, size=output_size, mode='trilinear')
    bg_ = F.interpolate(bg_, size=output_size, mode='trilinear')

    y_ = y_.round().byte()
    bg_ = bg_.round().bool()

    x_ = x_.squeeze(0).permute(3, 0, 1, 2)  # DCHW
    y_ = y_.squeeze(0).permute(3, 0, 1, 2)  # DCHW
    bg_ = bg_.squeeze(0).permute(3, 0, 1, 2)  # DCHW

    # Get segmentation over the entire volume
    # :=========================================================================:
    volume_dataloader = DataLoader(
        TensorDataset(x_, y_, bg_),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    x_norm = []
    y_pred = []
    bg_mask_vol = []
    
    for x, _, bg_mask in volume_dataloader:
        x = x.to(device).float()
        
        x_norm_part = norm(x)
                
        if seg_with_bg_supp:
            bg_mask = bg_mask.to(device)
            x_norm_bg_supp = background_suppression(x_norm_part, bg_mask, bg_suppression_opts)
            y_pred_part, _ = seg(x_norm_bg_supp)
        else:
            y_pred_part, _ = seg(x_norm_part)
        
        x_norm.append(x_norm_part.cpu())
        y_pred.append(y_pred_part.cpu())
        bg_mask_vol.append(bg_mask.cpu())
        

    x_norm = torch.vstack(x_norm)
    y_pred = torch.vstack(y_pred)

    # Rescale x and y to the original resolution
    x_norm = x_norm.permute(1, 0, 2, 3).unsqueeze(0)  # convert to NCDHW (with N=1)
    y_pred = y_pred.permute(1, 0, 2, 3).unsqueeze(0)  # convert to NCDHW (with N=1)
    
    if evaluate_also_bg_supp:
        bg_mask_vol = torch.vstack(bg_mask_vol).permute(1, 0, 2, 3).unsqueeze(0)  # convert to NCDHW (with N=1)
        y_pred_bg_supp = y_pred * bg_mask_vol.logical_not().float()
        y_pred_bg_supp = F.interpolate(y_pred_bg_supp, size=(D, H, W), mode='trilinear') 

    x_norm = F.interpolate(x_norm, size=(D, H, W), mode='trilinear')
    y_pred = F.interpolate(y_pred, size=(D, H, W), mode='trilinear')
    
    ## Calculate metrics and save images
    if export_seg_imgs:
        export_images(
            x_original,
            x_norm,
            y_original,
            y_pred,
            n_classes=n_classes,
            output_dir=os.path.join(logdir, 'example_segmentations'),
            image_name=f'{dataset_name}_{split}_{index:03}_{appendix}.png'
        )
        
        if evaluate_also_bg_supp:
            export_images(
                x_original,
                x_norm,
                y_original,
                y_pred_bg_supp,
                n_classes=n_classes,
                output_dir=os.path.join(logdir, 'example_segmentations_bg_supp_at_eval_time'),
                image_name=f'{dataset_name}_{split}_{index:03}_{appendix}.png'
            )
            
    if save_nii:
        save_nii_image(
            dir=os.path.join(logdir, 'volume_segmentations_all_classes'),
            filename=f'{dataset_name}_{split}_{index:03}_{appendix}.nii.gz', 
            image=onehot_to_class(y_pred).squeeze().detach().cpu().numpy().astype(np.int8)
        )
        
        if classes_of_interest is not None:
            for cls in classes_of_interest:
                save_nii_image(
                    dir=os.path.join(logdir, 'volume_segmentations_classes_of_interest'),
                    filename=f'{dataset_name}_{split}_{index:03}_{appendix}_cls_{cls}.nii.gz', 
                    image=onehot_to_class(y_pred[:, [0, cls]]).squeeze().detach().round().byte().cpu().numpy().astype(np.int8)
                )
                
                save_nii_image(
                    dir=os.path.join(logdir, 'volume_segmentations_classes_of_interest_bg_supp'),
                    filename=f'{dataset_name}_{split}_{index:03}_{appendix}_cls_{cls}.nii.gz', 
                    image=onehot_to_class(y_pred_bg_supp[:, [0, cls]]).squeeze().detach().round().byte().cpu().numpy().astype(np.int8)
                )

    metrics_index = {}
    for metric_name, metric_fn in metrics.items():
        if metric_name == 'dice':
            dices, dices_fg = metric_fn(y_pred, y_original)
            metrics_index['dice'] = {'dices': dices, 'dices_fg': dices_fg}
            print(f'\t mean dice score foreground: {dices_fg.mean().item()}')
            
            if evaluate_also_bg_supp:
                dices_bg_supp, dices_fg_bg_supp = metric_fn(y_pred_bg_supp, y_original)
                metrics_index['dice_bg_supp'] = {'dices': dices_bg_supp, 'dices_fg': dices_fg_bg_supp}
                print(f'\t mean dice score foreground with bg suppression during evaluation: {dices_fg_bg_supp.mean().item()}')
    
    return metrics_index


if __name__ == '__main__':
    
    # Load Hyperparameters
    print(f'Running {__file__}')
    
    # Loading general parameters
    # :=========================================================================:
    dataset_config, evaluation_config = get_configuration_arguments()
    
    dataset                 = evaluation_config['dataset']
    seg_dir                 = evaluation_config['seg_dir']
    seed                    = evaluation_config['seed']
    device                  = evaluation_config['device']
    split                   = evaluation_config['split']
    logdir                  = os.path.join(evaluation_config['logdir'], f'pred_on_{dataset}', split)

    assert os.path.exists(seg_dir), f"Path to segmentation directory does not exist: {seg_dir}"
    os.makedirs(logdir, exist_ok=True)
    
    params_seg              = load_config(os.path.join(seg_dir, 'params.yaml'))
    model_params_norm       = params_seg['model']['normalization_2D']
    model_params_seg        = params_seg['model']['segmentation_2D']
    train_params_seg        = params_seg['training']
    
    params                  = { 
                               'dataset': dataset_config,
                               'model': {'norm': model_params_norm, 'seg': model_params_seg},
                               'training': {'seg': train_params_seg},
                               'evaluation': evaluation_config,
                               }
    
    dump_config(os.path.join(logdir, 'params.yaml'), params)
    print_config(params, keys=['dataset', 'model', 'training', 'evaluation'])

    # Define the dataset that is to be used to evaluate the model
    # :=========================================================================:
    seed_everything(seed)
    device                  = define_device(device)
    n_classes               = dataset_config[dataset]['n_classes']
    paths                   = dataset_config[dataset]['paths_processed']
    paths_original          = dataset_config[dataset]['paths_original']
    image_size              = evaluation_config['image_size']
    resolution_proc         = dataset_config[dataset]['resolution_proc']
    dim_proc                = dataset_config[dataset]['dim']
    bg_suppresion_opts      = train_params_seg['segmentation']['bg_suppression_opts']
    
    print(f'Loading dataset: {dataset} {split}')
    eval_dataset, = get_datasets(
        paths = paths,
        paths_original = paths_original,
        splits = [split],
        image_size = image_size,
        resolution_proc = resolution_proc,
        dim_proc = dim_proc,
        n_classes = n_classes,
        aug_params = None,
        deformation = None,
        load_original = True,
        bg_suppression_opts = bg_suppresion_opts,
    )    
    print('Dataset loaded')

    # Load the segmentation model
    # :=========================================================================:
    print('Loading segmentation model')
    checkpoint_name = train_params_seg['checkpoint_best' if evaluation_config['load_best_cpt']
                                       else 'checkpoint_last']
    cpt_fp = os.path.join(seg_dir, checkpoint_name) 
    
    norm, seg = define_and_possibly_load_norm_seg(
        n_classes = n_classes,
        model_params_norm = model_params_norm,
        model_params_seg = model_params_seg,
        cpt_fp = cpt_fp,
        device = device,
    )
    norm.eval()
    seg.eval()
    print('Segmentation model loaded')
    
    # Evaluate the model
    # :=========================================================================:
    print(f'Evaluating model')
    vol_idx = evaluation_config['vol_idx']
    save_nii = evaluation_config['save_nii']
    batch_size = evaluation_config['batch_size']
    num_workers = evaluation_config['num_workers']
    also_bg_supp = evaluation_config['evaluate_also_bg_supp']
    classes_of_interest  = evaluation_config['classes_of_interest']

    indices_per_volume = eval_dataset.get_z_idxs_for_volumes() 
    indices_per_volume = {i: indices for i, indices in enumerate(indices_per_volume)}
    
    if vol_idx != 'all':
        indices_per_volume = {int(vol_idx): indices_per_volume[int(vol_idx)]}
    
    metrics_dict = {}
    for vol_i, idxs in indices_per_volume.items():
        print(f'Processing volume {vol_i}/{len(indices_per_volume) - 1}')
        volume_dataset = Subset(eval_dataset, idxs)
        metrics_dict[vol_i] = test_volume(
            norm=norm,
            seg=seg,
            volume_dataset=volume_dataset,
            dataset_name=dataset,
            n_classes=n_classes,
            index=vol_i,
            batch_size=batch_size,
            num_workers=num_workers,
            appendix='',
            metrics=metrics,
            bg_suppression_opts=None,
            seg_with_bg_supp=False,
            device=device,
            logdir=logdir,
            export_seg_imgs=True,
            evaluate_also_bg_supp=also_bg_supp,
            save_nii=save_nii,
            classes_of_interest=classes_of_interest
        )
        
    # Save the mean of each metric over the entire volume
    # :=========================================================================:
    
    out_summary_file = open(os.path.join(logdir, f'metrics_{dataset}_{split}.txt'), 'w')
    
    print('Saving metrics')
    print(f'Classes of interest: {classes_of_interest}')
    for metric_name in metrics_dict[0].keys():
        
        if 'dice' in metric_name:
            print(f'Storing results for {metric_name}' + '\n' + '-'*50)
            dice_dict = [{'vol': i, 'dice': dices[metric_name]['dices'][0].cpu().numpy().tolist()}
                         for i, dices in metrics_dict.items()]            
            df = pd.DataFrame(dice_dict).set_index('vol').sort_index()
            
            # Expand the list of dices into columns
            df = pd.DataFrame(df['dice'].to_list(), index=df.index)
                       
            # Save the dataframe to csv
            df.to_csv(os.path.join(logdir, f'{metric_name}_{dataset}_{split}.csv'))
            
            # Dice of the foreground classes
            fg_cols = [col for col in df.columns if col != 0]
            mean_dice_fg_str = f'Mean {metric_name} over entire {split} set: ' + \
                  f'{df[fg_cols].mean(axis=1).mean():.3f} +/- {df[fg_cols].mean(axis=1).std():.3f}'
            print(mean_dice_fg_str)
            out_summary_file.write(mean_dice_fg_str + '\n')
            
            # Dice of the classes of interest
            if classes_of_interest is not None:
                cls_interest_cols = [col for col in df.columns if col in classes_of_interest]
                mean_dice_cls_interest_str = (f'Mean {metric_name} over entire {split} set for classes of interest: {classes_of_interest}'
                        f'{df[cls_interest_cols].mean(axis=1).mean():.3f} +/- {df[cls_interest_cols].mean(axis=1).std():.3f}')
                print(mean_dice_cls_interest_str)
                out_summary_file.write(mean_dice_cls_interest_str + '\n')
            out_summary_file.write('\n')

    out_summary_file.close()
            
            
            
            
            
        
    