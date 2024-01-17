"""
This script resamples the images to a desired resolution

It is used to resample the different TOF-MRA images to the same resolution
"""
import os
import sys
import tqdm
from typing import Optional
from datetime import datetime
import logging
import argparse

import numpy as np
import nibabel as nib


sys.path.append(os.path.normpath(os.path.dirname(__file__), '..', '..', 'tta_uia_segmentation', 'src'))
from preprocessing.resampling import resample
from preprocessing.utils import get_filepaths


#---------- paths & hyperparameters
voxel_size_default          = np.array([0.3, 0.3, 0.6]) # hyper parameters to be set
save_logs                   = True
path_to_logs                = '/scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/logs/preprocessing/resampling'
path_to_save_processed_data = '/scratch_net/biwidl319/jbermeo/data/preprocessed/1_resampled'
#----------

date_now = datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(path_to_logs, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    filename=os.path.join(path_to_logs, f'{date_now}_resampling.log'), filemode='w')
log = logging.Logger('Resampling')


def preprocess_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Resampling of images')
    parser.add_argument('--voxel_size', type=float, nargs='+', default=voxel_size_default)
    parser.add_argument('--order', type=int, default=3)  
   
    parser.add_argument('--preprocessed', action='store_true', default=False)
    parser.add_argument('--path_to_dir', type=str)
    
    parser.add_argument('--dataset', type=str, choices=['USZ', 'ADAM', 'Laussane', None])
    parser.add_argument('--path_to_tof_dir', type=str)
    parser.add_argument('--fp_pattern_tof', type=str, nargs='+')
    parser.add_argument('--path_to_seg_dir', type=str)
    parser.add_argument('--fp_pattern_seg', type=str, nargs='+')
    parser.add_argument('--level_of_dir_with_id', type=int, default=-2)
    parser.add_argument('--not_every_scan_has_seg', action='store_true', default=False)
    
    parser.add_argument('--path_to_save_processed_data', type=str, default=path_to_save_processed_data)   
    parser.add_argument('--path_to_logs', type=str, default=path_to_logs)   
        
    args = parser.parse_args()

    if args.preprocessed:
        if args.path_to_dir is None:
            parser.error('--path_to_dir is required when --preprocessed is specified')  
    
    else:
        if args.path_to_tof_dir is None:
            parser.error('--path_to_tof_dir is required when --preprocessed is not specified')
                    
        if args.path_to_seg_dir is not None:
            if args.fp_pattern_seg is None:
                parser.error('--fp_pattern_seg is required when --path_to_seg_dir is not None')
        
        if args.dataset is None:
            if args.fp_pattern_tof is None:
                parser.error('--fp_pattern_tof is required when --dataset is None')
            
            if args.path_to_seg_dir is not None and args.fp_pattern_seg is None:
                parser.error('--fp_pattern_seg is required when --path_to_seg_dir is not None and --dataset is None')
                    
        if args.dataset == 'Lausanne' and args.path_to_seg_dir is None:
            parser.error('--path_to_seg_dir is required when --dataset is Lausanne')        
        
    
    return args


def resample_image_and_segmentation_mask(
    scans_dict: dict[str, dict[str, str]],
    voxel_size: tuple[float, float, float],
    save_output: bool = False, 
    output_dir: Optional[str] = None
    ):
    if save_output: os.makedirs(output_dir, exist_ok=True)
    
    # For now let's do it sequentially. Later we can parallelize it
    for img_id, img_dict in tqdm.tqdm(scans_dict.items()):
        log.info(f"Resampling scan {img_id}")
        
        if save_output:
            img_output_dir = os.path.join(output_dir, img_id)
            os.makedirs(img_output_dir, exist_ok=True)
        
        # Load the TOF scan
        tof_scan = nib.load(img_dict['tof'])
        
        # Resample the TOF scan
        resampled_tof_scan = resample(tof_scan, voxel_size)
        
        # Save the resampled TOF scan
        if save_output:
            nib.save(resampled_tof_scan, os.path.join(img_output_dir, f'{img_id}_tof.nii.gz'))
        
        # If the scan has a segmentation mask, resample it
        if 'seg' in img_dict.keys():
            seg_mask = nib.load(img_dict['seg'])
            resampled_seg_mask = resample(seg_mask, voxel_size, is_segmenation=True)
            
            # Save the resampled segmentation mask
            if save_output:
                nib.save(resampled_seg_mask, os.path.join(img_output_dir, f'{img_id}_seg.nii.gz'))
            
        log.info(f"Scan {img_id} resampled")


if __name__ == '__main__':
    # path_to_USZ_dataset       = '/scratch_net/biwidl319/jbermeo/data/raw/USZ'
    # path_to_ADAM_dataset      = '/scratch_net/biwidl319/jbermeo/data/raw/ADAM'
    # path_to_Laussane_tof      = '/scratch_net/biwidl319/jbermeo/data/raw/Lausanne/original_images'
    # path_to_Laussane_seg      = '/scratch_net/biwidl319/jbermeo/data/raw/Lausanne/skull_stripped_and_aneurysm_mask'
    
    args = preprocess_cmd_args()

    # Get filepaths of the the dataset
    scans_dict = get_filepaths(
        preprocessed=args.preprocessed,
        path_to_dir=args.path_to_dir,
        dataset=args.dataset,
        path_to_tof_dir=args.path_to_tof_dir,
        fp_pattern_tof=args.fp_pattern_tof,
        path_to_seg_dir=args.path_to_seg_dir,
        fp_pattern_seg=args.fp_pattern_seg,
        level_of_dir_with_id=args.level_of_dir_with_id,
        every_scan_has_seg=not args.not_every_scan_has_seg
    )
    
    resample_image_and_segmentation_mask(
        scans_dict=scans_dict,
        voxel_size=args.voxel_size,
        save_output=True,
        output_dir=args.path_to_save_processed_data,
    )
    print('Done!')
    