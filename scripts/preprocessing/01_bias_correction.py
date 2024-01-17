import os
import sys
import tqdm
import logging
import argparse
import shutil
import multiprocessing
from typing import Optional
from datetime import datetime

import SimpleITK as sitk

sys.path.append(os.path.normpath(os.path.dirname(__file__), '..', '..', 'tta_uia_segmentation', 'src'))

from preprocessing.bias_correction import N4bias_correction_filter
from preprocessing.utils import get_filepaths


#---------- paths & hyperparameters
shrink_factor_default       = 1
max_num_iters_default       = [50, 50, 50] 
path_to_logs                = '/scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/logs/preprocessing/resampling'
path_to_save_processed_data = '/scratch_net/biwidl319/jbermeo/data/preprocessed/0_bias_corrected'
multi_proc_default          = True
n_threads_default           = 4
#----------


date_now = datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(path_to_logs, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    filename=os.path.join(path_to_logs, f'{date_now}_bias_correction.log'), filemode='w')
log = logging.Logger('Bias Correction')


def preprocess_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Bias correction of scans')
    
    parser.add_argument('--shrink_factor', type=int, default=shrink_factor_default)
    parser.add_argument('--maximum_number_of_iterations', type=int, nargs='+', default=max_num_iters_default)
    
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
    
    parser.add_argument('--multi_proc', action='store_true', default=multi_proc_default)
    parser.add_argument('--n_threads', type=int, default=n_threads_default)
    
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



def bias_correction_image_and_segmentation_mask_sequential(
    scans_dict: dict[str, dict[str, str]],
    shrink_factor: int = shrink_factor_default,
    max_num_iterations: list[int] = max_num_iters_default,
    save_output: bool = False, 
    output_dir: Optional[str] = None):
    
    if save_output: 
        os.makedirs(output_dir, exist_ok=True)
    
    # For now let's do it sequentially. Later we can parallelize it
    for img_id, img_dict in tqdm.tqdm(scans_dict.items()):
        log.info(f"N4 Bias correction for scan {img_id}")
        img_output_dir = os.path.join(output_dir, img_id)
        os.makedirs(img_output_dir, exist_ok=True)
        
        # Load the TOF scan
        img = sitk.ReadImage(img_dict['tof'], sitk.sitkFloat64)

        # N4 bias correction
        img_res, _ = N4bias_correction_filter(
            img,
            image_mask_flag    = True, 
            shrink_factor      = shrink_factor, 
            max_num_iterations = max_num_iterations
        )
        
        # Save the bias corrected TOF scan
        if save_output:
            sitk.WriteImage(img_res, os.path.join(img_output_dir, f'{img_id}_tof.nii.gz'))
            
            # Copy the segmentation mask to the img_output_dir
            if 'seg' in img_dict:
                shutil.copyfile(img_dict['seg'], os.path.join(img_output_dir, f'{img_id}_seg.nii.gz'))        
        
        log.info(f"Scan {img_id} bias corrected")


def bias_correction_multiprocess(
    scans_dict: dict[str, dict[str, str]],
    lock: multiprocessing.Lock,
    every_n: int = 4,
    start_i: int = 0,
    shrink_factor: int = shrink_factor_default,
    max_num_iterations: list[int] = max_num_iters_default,
    save_output: bool = True,
    output_dir: Optional[str] = './',
    ):
    
    with lock:
        os.makedirs(output_dir, exist_ok=True)
    
    img_ids = list(scans_dict.keys())
    
    for idx in range(start_i, len(img_ids), every_n):
        img_id = img_ids[idx]
        img_dict = scans_dict[img_id]
        log.info(f"N4 Bias correction for scan {img_id}")
        img_output_dir = os.path.join(output_dir, img_id)
        
        with lock:
            os.makedirs(img_output_dir, exist_ok=True)
        
        # Load the TOF scan
        img = sitk.ReadImage(img_dict['tof'], sitk.sitkFloat64)

        # N4 bias correction
        img_res, _ = N4bias_correction_filter(
            img,
            image_mask_flag    = True, 
            shrink_factor      = shrink_factor, 
            max_num_iterations = max_num_iterations
        )
        
        # Save the bias corrected TOF scan
        if save_output:
            sitk.WriteImage(img_res, os.path.join(img_output_dir, f'{img_id}_tof.nii.gz'))

            if 'seg' in img_dict:
                shutil.copyfile(img_dict['seg'], os.path.join(img_output_dir, f'{img_id}_seg.nii.gz'))   
        
        log.info(f"Scan {img_id} bias corrected")

    

if __name__ == '__main__':
    args = preprocess_cmd_args()
    
    # path_to_USZ_dataset         = '/scratch_net/biwidl319/jbermeo/data/raw/USZ'
    # path_to_ADAM_dataset        = '/scratch_net/biwidl319/jbermeo/data/raw/ADAM'
    # path_to_Laussane_tof        = '/scratch_net/biwidl319/jbermeo/data/raw/Lausanne/original_images'
    # path_to_Laussane_seg        = '/scratch_net/biwidl319/jbermeo/data/raw/Lausanne/skull_stripped_and_aneurysm_mask'
    
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
    
    # Iterate over the scans and perform bias correction
    if args.multi_proc:
        print('Starting bias correction with multiprocessing')
        manager    = multiprocessing.Manager()
        lock       = manager.Lock()
        split_dif  = args.n_threads
        start_from = 0
        split_id   = 0
        ps         = []

        for k in range(start_from + split_id * split_dif, start_from + split_dif * (split_id+1)):
            
            ps.append(
                multiprocessing.Process(
                    target=bias_correction_multiprocess,
                    args=(
                        scans_dict,
                        lock,
                        split_dif,
                        k,
                        args.shrink_factor,
                        args.maximum_number_of_iterations,
                        True,
                        args.path_to_save_processed_data
                    )
                )
            )
             
        for k in range(len(ps)):    ps[k].start()
        for k in range(len(ps)):    ps[k].join()
        
    else:
        bias_correction_image_and_segmentation_mask_sequential(
            scans_dict=scans_dict,
            shrink_factor=args.shrink_factor,
            max_num_iterations=args.maximum_number_of_iterations,
            save_output=True,
            output_dir=args.path_to_save_processed_data
        )
    
    print('Done!')
    
    