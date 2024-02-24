import os
import sys
import tqdm
import logging
import argparse
import multiprocessing as mp
<<<<<<< HEAD:scripts/preprocessing/03_registration.py
=======
from typing import Optional, Union
>>>>>>> started adding code for the DAE:tta_uia_segmentation/src/preprocessing/03_registration.py
from datetime import datetime

import numpy as np
import SimpleITK as sitk

sys.path.append(os.path.normpath(os.path.dirname(__file__), '..', '..', 'tta_uia_segmentation', 'src'))

from preprocessing.registration import rigid_registration
from preprocessing.utils import get_filepaths


#---------- paths & hyperparameters
mmi_n_bins_default              = 50
learning_rate_default           = 1.0
number_of_iterations_default    = 100
fixed_image_path_default        = '/scratch_net/biwidl319/jbermeo/data/preprocessed/1_resampled/USZ/10745241-MCA-new/10745241-MCA-new_tof.nii.gz'
path_to_logs                    = '/scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/logs/preprocessing/aligned'
path_to_save_processed_data     = '/scratch_net/biwidl319/jbermeo/data/preprocessed/2_aligned'
multi_proc_default              = True
n_threads_default               = 4
#----------


def preprocess_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Registratin of scans')
    
    parser.add_argument('--fixed_image_path', type=str, default=fixed_image_path_default)  
    parser.add_argument('--mmi_n_bins', type=int, default=mmi_n_bins_default)   
    parser.add_argument('--learning_rate', type=float, default=learning_rate_default)
    parser.add_argument('--number_of_iterations', type=int, default=number_of_iterations_default)
    parser.add_argument('--not_use_geometrical_center_mode', action='store_true', default=False)
        
    parser.add_argument('--preprocessed', action='store_true', default=False)
    parser.add_argument('--path_to_dir', type=str)
    
    parser.add_argument('--dataset', type=str, choices=['USZ', 'ADAM', 'Laussane', None])
    parser.add_argument('--path_to_tof_dir', type=str)
    parser.add_argument('--fp_pattern_tof', type=str, nargs='+')
    parser.add_argument('--path_to_seg_dir', type=str)
    parser.add_argument('--fp_pattern_seg', type=str, nargs='+')
    parser.add_argument('--level_of_dir_with_id', type=int, default=-2)
    parser.add_argument('--not_every_scan_has_seg', action='store_true', default=False)
    
    parser.add_argument('--sample_size_frac', type=float, default=1.0)
    
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


def rigid_registration_sequential(
    scans_dict: dict[str, dict[str, str]],
    fixed_image_path: str,
    use_geometrical_center_mode: bool = True,
    mmi_n_bins: int = mmi_n_bins_default,
    learning_rate: float = learning_rate_default,
    number_of_iterations: int = number_of_iterations_default,
    output_dir: str = path_to_save_processed_data
):
     # For now let's do it sequentially. Later we can parallelize it
    os.makedirs(output_dir, exist_ok=True)
        
    for img_id, img_dict in tqdm.tqdm(scans_dict.items()):
        log.info(f"Registering scan {img_id}")
        
        registered_tof_scan, registered_seg_mask = rigid_registration(
            fixed_image_path=fixed_image_path,
            moving_image_path=img_dict['tof'],
            image_segmentation_mask_path=img_dict['seg'] if 'seg' in img_dict.keys() else None,
            use_geometrical_center_mode=use_geometrical_center_mode,
            mmi_n_bins=mmi_n_bins,
            learning_rate=learning_rate,
            number_of_iterations=number_of_iterations
        )
        
        # Save the registered TOF scan and registered segmentation mask
        img_output_dir = os.path.join(output_dir, img_id)
        os.makedirs(img_output_dir, exist_ok=True)
        
        sitk.WriteImage(registered_tof_scan, os.path.join(img_output_dir, f'{img_id}_tof.nii.gz'))
        
        if 'seg' in img_dict.keys():
            sitk.WriteImage(registered_seg_mask, os.path.join(img_output_dir, f'{img_id}_seg.nii.gz'))
        
        log.info(f"Scan {img_id} registered") 
        
        
def rigid_registration_multiprocess(
    scans_dict: dict[str, dict[str, str]],
    fixed_image_path: str,
    lock: mp.Lock,
    every_n: int = 4,
    start_i: int = 0,
    use_geometrical_center_mode: bool = True,
    mmi_n_bins: int = mmi_n_bins_default,
    learning_rate: float = learning_rate_default,
    number_of_iterations: int = number_of_iterations_default,
    output_dir: str = path_to_save_processed_data
):
    
    with lock:
        os.makedirs(output_dir, exist_ok=True)
    
    img_ids = list(scans_dict.keys())
    
    for idx in range(start_i, len(img_ids), every_n):
        img_id = img_ids[idx]
        img_dict = scans_dict[img_id]
        
        log.info(f"Registering scan {img_id}")
        
        registered_tof_scan, registered_seg_mask = rigid_registration(
            fixed_image_path=fixed_image_path,
            moving_image_path=img_dict['tof'],
            image_segmentation_mask_path=img_dict['seg'] if 'seg' in img_dict.keys() else None,
            use_geometrical_center_mode=use_geometrical_center_mode,
            mmi_n_bins=mmi_n_bins,
            learning_rate=learning_rate,
            number_of_iterations=number_of_iterations
        )
        
        # Save the registered TOF scan and registered segmentation mask
        with lock:
            img_output_dir = os.path.join(output_dir, img_id)
            os.makedirs(img_output_dir, exist_ok=True)
        
        sitk.WriteImage(registered_tof_scan, os.path.join(img_output_dir, f'{img_id}_tof.nii.gz'))
        
        if 'seg' in img_dict.keys():
            sitk.WriteImage(registered_seg_mask, os.path.join(img_output_dir, f'{img_id}_seg.nii.gz'))
        
        log.info(f"Scan {img_id} registered") 
        


def rigid_registration_sequential(
    scans_dict: dict[str, dict[str, str]],
    fixed_image_path: str,
    use_geometrical_center_mode: bool = True,
    mmi_n_bins: int = mmi_n_bins_default,
    learning_rate: float = learning_rate_default,
    number_of_iterations: int = number_of_iterations_default,
    save_output: bool = False, 
    output_dir: Optional[str] = None
):
     # For now let's do it sequentially. Later we can parallelize it
    os.makedirs(args.path_to_save_processed_data, exist_ok=True)
        
    for img_id, img_dict in tqdm.tqdm(scans_dict.items()):
        log.info(f"Registering scan {img_id}")
        
        registered_tof_scan, registered_seg_mask = rigid_registration(
            fixed_image_path=args.fixed_image_path,
            moving_image_path=img_dict['tof'],
            image_segmentation_mask_path=img_dict['seg'] if 'seg' in img_dict.keys() else None,
            use_geometrical_center_mode=not args.not_use_geometrical_center_mode,
            mmi_n_bins=args.mmi_n_bins,
            learning_rate=args.learning_rate,
            number_of_iterations=args.number_of_iterations
        )
        
        # Save the registered TOF scan and registered segmentation mask
        img_output_dir = os.path.join(args.path_to_save_processed_data, img_id)
        os.makedirs(img_output_dir, exist_ok=True)
        
        sitk.WriteImage(registered_tof_scan, os.path.join(img_output_dir, f'{img_id}_tof.nii.gz'))
        
        if 'seg' in img_dict.keys():
            sitk.WriteImage(registered_seg_mask, os.path.join(img_output_dir, f'{img_id}_seg.nii.gz'))
        
        log.info(f"Scan {img_id} registered") 
        
        
def rigid_registration_multiprocess(
    scans_dict: dict[str, dict[str, str]],
    fixed_image_path: str,
    lock: mp.Lock,
    every_n: int = 4,
    start_i: int = 0,
    use_geometrical_center_mode: bool = True,
    mmi_n_bins: int = mmi_n_bins_default,
    learning_rate: float = learning_rate_default,
    number_of_iterations: int = number_of_iterations_default,
    save_output: bool = False, 
    output_dir: Optional[str] = None
):
    
    with lock:
        os.makedirs(output_dir, exist_ok=True)
    
    img_ids = list(scans_dict.keys())
    
    for idx in range(start_i, len(img_ids), every_n):
        img_id = img_ids[idx]
        img_dict = scans_dict[img_id]
        
        log.info(f"Registering scan {img_id}")
        
        registered_tof_scan, registered_seg_mask = rigid_registration(
            fixed_image_path=args.fixed_image_path,
            moving_image_path=img_dict['tof'],
            image_segmentation_mask_path=img_dict['seg'] if 'seg' in img_dict.keys() else None,
            use_geometrical_center_mode=not args.not_use_geometrical_center_mode,
            mmi_n_bins=args.mmi_n_bins,
            learning_rate=args.learning_rate,
            number_of_iterations=args.number_of_iterations
        )
        
        # Save the registered TOF scan and registered segmentation mask
        with lock:
            img_output_dir = os.path.join(args.path_to_save_processed_data, img_id)
            os.makedirs(img_output_dir, exist_ok=True)
        
        sitk.WriteImage(registered_tof_scan, os.path.join(img_output_dir, f'{img_id}_tof.nii.gz'))
        
        if 'seg' in img_dict.keys():
            sitk.WriteImage(registered_seg_mask, os.path.join(img_output_dir, f'{img_id}_seg.nii.gz'))
        
        log.info(f"Scan {img_id} registered") 
        

if __name__ == '__main__':
    args = preprocess_cmd_args()
    
    date_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.path_to_logs, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        filename=os.path.join(args.path_to_logs, f'{date_now}_registration.log'), filemode='w')
    log = logging.Logger('Registration')
    
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
    
    if args.sample_size_frac < 1.0:
        log.info(f'Using a sample of {args.sample_size_frac} of the dataset')
        scans = list(scans_dict.keys())
        scans = np.random.choice(scans, size=int(len(scans) * args.sample_size_frac), replace=False)
        scans_dict = {k: scans_dict[k] for k in scans}
    
    print(f'We have {len(scans_dict)} to register')
        
    if args.multi_proc:
        print('Starting bias correction with multiprocessing')
        manager    = mp.Manager()
        lock       = manager.Lock()
        split_dif  = args.n_threads
        start_from = 0
        split_id   = 0
        ps         = []

        for k in range(start_from + split_id * split_dif, start_from + split_dif * (split_id+1)):
            ps.append(
                mp.Process(
                    target=rigid_registration_multiprocess,
                    args=(
                        scans_dict,
                        args.fixed_image_path,
                        lock,
                        args.n_threads,
                        k,
                        not args.not_use_geometrical_center_mode,
                        args.mmi_n_bins,
                        args.learning_rate,
                        args.number_of_iterations,
<<<<<<< HEAD:scripts/preprocessing/03_registration.py
=======
                        True,
>>>>>>> started adding code for the DAE:tta_uia_segmentation/src/preprocessing/03_registration.py
                        args.path_to_save_processed_data
                    )
                )
            )
<<<<<<< HEAD:scripts/preprocessing/03_registration.py
        
        for k in range(len(ps)):    ps[k].start()
        for k in range(len(ps)):    ps[k].join()
        
=======
        
        for k in range(len(ps)):    ps[k].start()
        for k in range(len(ps)):    ps[k].join()
        
>>>>>>> started adding code for the DAE:tta_uia_segmentation/src/preprocessing/03_registration.py
    else:
        rigid_registration_sequential(
            scans_dict=scans_dict,
            fixed_image_path=args.fixed_image_path,
            use_geometrical_center_mode=not args.not_use_geometrical_center_mode,
            mmi_n_bins=args.mmi_n_bins,
            learning_rate=args.learning_rate,
            number_of_iterations=args.number_of_iterations,
<<<<<<< HEAD:scripts/preprocessing/03_registration.py
=======
            save_output=True,
>>>>>>> started adding code for the DAE:tta_uia_segmentation/src/preprocessing/03_registration.py
            output_dir=args.path_to_save_processed_data
        )
        
    log.info(f"Registration finished!")
        


