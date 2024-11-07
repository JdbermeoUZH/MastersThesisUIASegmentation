import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join('..', '..', 'tta_uia_segmentation', 'src'))

from tta_uia_segmentation.src.dataset.dataset_in_memory import get_datasets
from tta_uia_segmentation.src.models.io import define_and_possibly_load_norm_seg
from tta_uia_segmentation.src.utils.io import load_config
from tta_uia_segmentation.src.train.NormSegTrainer import NormSegTrainer

split_def           = 'train'
device_def          = 'cuda' if torch.cuda.is_available() else 'cpu' 
frac_dataset_def    = 1.0
bach_size_def       = None
num_epochs_def      = 1   


def parse_args():
    parser = argparse.ArgumentParser(description='Estimating moments and quartiles of the normalized images')
    
    parser.add_argument('model_dir', type=str, help='Path to the trained model directory')
    parser.add_argument('dataset', type=str, help='Name of the dataset')
    parser.add_argument('--split', type=str, default='train', help='Split to use for estimating moments and quartiles')
    parser.add_argument('--device', type=str, default=device_def, help='Device to use for estimating moments and quartiles')
    parser.add_argument('--frac_dataset', type=float, default=frac_dataset_def, help='Fraction of the dataset to use for estimating moments and quartiles')
    parser.add_argument('--batch_size', type=int, default=bach_size_def, help='Batch size for estimating moments and quartiles')
    parser.add_argument('--num_epochs', type=int, default=num_epochs_def, help='Number of epochs to use for estimating moments and quartiles')
    
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
          
    # Load config  
    params              = load_config(os.path.join(args.model_dir, 'params.yaml'))
    dataset_params      = params['dataset'][args.dataset]
    model_params        = params['model']
    train_params        = params['training']
    train_seg_params    = train_params['segmentation']
    
    # Load dataset
    print('Loading dataset')
    (ds, )  = get_datasets(
        splits          = [args.split],
        paths           = dataset_params['paths_processed'],
        paths_original  = dataset_params['paths_original'], 
        image_size      = train_seg_params['image_size'],
        resolution_proc = dataset_params['resolution_proc'],
        dim_proc        = dataset_params['dim'],
        n_classes       = dataset_params['n_classes'],
        aug_params      = train_seg_params['augmentation'],
        deformation     = None,
        load_original   = True,
        bg_suppression_opts = train_seg_params['bg_suppression_opts']
    )
    
    batch_size = args.batch_size if args.batch_size is not None else train_seg_params['batch_size']
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    cpts = [train_params['checkpoint_best'], train_params['checkpoint_last']]
    
    for cpt_fn in cpts:
        # Load model
        print(f'Loading model from checkpoint {cpt_fn}')
        norm, seg = define_and_possibly_load_norm_seg(
            n_classes=dataset_params['n_classes'],
            model_params_norm=model_params['normalization_2D'],
            model_params_seg=model_params['segmentation_2D'],
            cpt_fp=os.path.join(args.model_dir, cpt_fn),
            device=args.device
        )
        norm.eval()
        
        # Load trainer object for segmentation task with normalizer
        trainer = NormSegTrainer(
            norm=norm,
            seg=seg,
            device=args.device,
            learning_rate       = train_seg_params['learning_rate'],
            is_resumed          = False,
            checkpoint_best     = 'delete me',
            checkpoint_last     = 'delete me',
            logdir              = args.model_dir,
            wandb_log           = False,
            wandb_dir           = None,
            bg_suppression_opts = train_seg_params['bg_suppression_opts']
        )
        
        trainer._save_normalized_images_moments_and_quantiles(
            fp_prefix=os.path.join(args.model_dir, cpt_fn).replace('.pth', ''),
            dataloader=dl, frac_dataset=args.frac_dataset, num_epochs=args.num_epochs
        )