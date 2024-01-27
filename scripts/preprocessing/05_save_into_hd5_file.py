
import os
import sys
import json
import argparse

import h5py
import numpy as np
import nibabel as nib
from sklearn.model_selection import KFold, train_test_split

sys.path.append(os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'tta_uia_segmentation', 'src')))

from preprocessing.utils import get_filepaths

#---------- paths & hyperparameters
num_folds_default                   = 5
train_val_split_default             = 0.25
preprocessed_default                = True
level_of_dir_with_id_default        = -2
path_to_save_processed_data_default = '/scratch_net/biwidl319/jbermeo/data/preprocessed/UIA_segmentation'
diameter_threshold_default          = 4  # mm, separate the scans into two groups: < 4mm and >= 4mm. UIAs < 4mm are usually not treated
seed                                = 0
num_channels                        = 1
#----------

def preprocess_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Registratin of scans')
    
    parser.add_argument('dataset', type=str, choices=['USZ', 'ADAM', 'Lausanne', None])
    parser.add_argument('path_to_dir', type=str, help='Path to the directory with the scans')
    parser.add_argument('path_label_names_dict', type=str, help='Path to the dictionary that maps the label names to their corresponding class')
    parser.add_argument('--num_folds', type=int, default=num_folds_default, help='Number of folds for cross-validation')
    parser.add_argument('--train_val_split', type=float, default=train_val_split_default,
                        help='Percentage of the train set to use as validation')
    parser.add_argument('--diameter_threshold', type=float, default=diameter_threshold_default,
                        help='Diameter threshold to separate the scans into two groups: < 4mm and >= 4mm. UIAs < 4mm are usually not treated')
    parser.add_argument('--path_to_save_processed_data', type=str, default=path_to_save_processed_data_default)   
    parser.add_argument('--num_channels', type=int, default=num_channels, help='Number of channels of the scans')
        
    args = parser.parse_args()
    
    return args


def separate_filepaths_based_on_aneurysm_diameter(
    scan_fps: dict,
    diameter: float = diameter_threshold_default
    ):
    less_than_4mm = []
    for scan_id, image_fps in scan_fps.items():
        if 'seg' not in image_fps:
            continue
        seg_mask = nib.load(image_fps['seg'])
        seg_mask_arr = seg_mask.get_fdata()
        voxel_size = np.prod(seg_mask.header.get_zooms())
        
        # Calculate the approximate diameter of class 4, assuming it is a sphere
        uia_vol = seg_mask_arr[seg_mask_arr == 4].sum() * voxel_size
        
        if uia_vol == 0:
            continue
        
        uia_diameter = np.power(6 * uia_vol / np.pi, 1/3)
        
        if uia_diameter <= diameter:
            less_than_4mm.append(scan_id)    
            
    scan_fps_less_than_4mm = {scan_id: scan_fps[scan_id] for scan_id in less_than_4mm}
    scan_fps_greater_than_4mm = {scan_id: scan_fps[scan_id] for scan_id in scan_fps if scan_id not in less_than_4mm}    
    
    return scan_fps_less_than_4mm, scan_fps_greater_than_4mm    


def _verify_expected_channels(scan_data: np.ndarray, num_channels: int = 1):
    assert len(scan_data.shape) in [3, 4], \
    f'Expected a single channel or multichannel 3D scan, but got {len(scan_data.shape)}D'
    
    if num_channels == 1 and len(scan_data.shape) == 3:
        scan_data = np.expand_dims(scan_data, axis=0)
        
    elif num_channels == 1 and len(scan_data.shape) == 4:
        if scan_data.shape[0] == 1:
            scan_data = scan_data[0]
        else:
            raise ValueError(f'Expected a single channel scan, but got {scan_data.shape[0]} channels')
        
    return scan_data


def add_scans_to_group(scan_fps: dict, h5_fp: str, group_name: str, num_channels: int = 1, max_buffer_scans: int = 5):
    num_scans_written = 0
    h5f = h5py.File(h5_fp, 'a')
    H5Data = h5f.create_group(group_name)
    
    for scan_id, image_fps in scan_fps.items():
        num_scans_written += 1
        
        if num_scans_written % max_buffer_scans == 0:
            h5f.close()
            h5f = h5py.File(h5_fp, 'a')
            H5Data = h5f[group_name]
            
        H5Scan = H5Data.create_group(scan_id)
        
        # Store the tof scan
        scan = nib.load(image_fps['tof'])
        
        # Convert the scan from WHD to DHW
        scan_data = scan.get_fdata()
        scan_data = _verify_expected_channels(scan_data, num_channels=num_channels)
        
        scan_data = np.moveaxis(scan_data, -1, -3)
        scan_data = np.moveaxis(scan_data, -2, -1)
        scan_data = np.rot90(scan_data, k=2, axes=(-3, -2))
            
        
        H5Scan.create_dataset('tof', data=scan_data, dtype=np.float32)
        
        # Store original spacing
        px, py, pz = scan.header.get_zooms()
        H5Scan.create_dataset('px', data=px)
        H5Scan.create_dataset('py', data=py)
        H5Scan.create_dataset('pz', data=pz)
        
        # Store the segmentation, also in DHW
        if every_scan_has_seg:
            seg_data = nib.load(image_fps['seg']).get_fdata()
            seg_data = _verify_expected_channels(seg_data, num_channels=num_channels)
            seg_data = np.moveaxis(seg_data, -1, -3)
            seg_data = np.moveaxis(seg_data, -2, -1)
            seg_data = np.rot90(seg_data, k=2, axes=(-3, -2))
            H5Scan.create_dataset('seg', data=seg_data, dtype=np.uint8)
        else:
            H5Scan.create_dataset('seg', data=np.zeros(scan.shape))
        

if __name__ == '__main__':
    args = preprocess_cmd_args()
    
    every_scan_has_seg = False if args.dataset == 'Lausanne' else True
    
    # Get filepaths of all scans
    scan_fps = get_filepaths(
        path_to_dir=args.path_to_dir, 
        preprocessed=preprocessed_default, 
        every_scan_has_seg=False)
    
    # Filter out the scans that have UIAs of less than 4mm of diameter
    scan_fps_leq_4mm, scan_fps = separate_filepaths_based_on_aneurysm_diameter(
        scan_fps, diameter=args.diameter_threshold if args.dataset != 'USZ' else 0)
    
    # Create a hdf5 file to store the preprocessed data
    os.makedirs(args.path_to_save_processed_data, exist_ok=True)
    h5_fp = os.path.join(args.path_to_save_processed_data, f'{args.dataset}.h5')
    h5f = h5py.File(h5_fp, 'w')
    
    # Store general metadata of the dataset
    # :================================================================================================:
    scan_ids = list(scan_fps.keys())
    h5f.create_dataset('ids', data=scan_ids)
    
    label_names = json.load(open(args.path_label_names_dict, 'r'))
    h5f.create_dataset('label_names', data=json.dumps(label_names))
    
    # Create a folds group to store the indexes of each of the folds
    # :================================================================================================:
    H5Folds = h5f.create_group('folds')
    
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(kf.split(scan_ids)):
        # Create a group for each fold in the hdf5 file
        H5Fold = H5Folds.create_group(f'fold_{fold}')
        
        # Add the train and test indexes to the fold group
        H5Fold.create_dataset('train_idx', data=train_idx)
        H5Fold.create_dataset('test_idx', data=test_idx)
        
        # Create a list with the train-dev/val-dev folds
        train_dev_idx, val_dev_idx = train_test_split(
            train_idx, test_size=args.train_val_split, 
            shuffle=True, random_state=seed)    

        # Add the train-dev and val-dev indexes to the fold group
        H5Fold.create_dataset('train_dev_idx', data=train_dev_idx)
        H5Fold.create_dataset('val_dev_idx', data=val_dev_idx)
    
    h5f.close()
        
    # Create a data group to store the preprocessed data (each image index is a group)
    # :================================================================================================:
    add_scans_to_group(scan_fps, h5_fp, 'data')
            
    # Store the filepaths of the scans with UIAs of less than 4mm of diameter
    if len(scan_fps_leq_4mm) >= 0:
        add_scans_to_group(scan_fps_leq_4mm, h5_fp, 'data_leq_4mm')