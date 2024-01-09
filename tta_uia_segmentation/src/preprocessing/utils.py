"""
Utility functions for preprocessing    

It includes functions to:
 - Load filepaths scans and segmentations masks of different datasets 
"""


import os
import glob
from collections import OrderedDict
from typing import Optional


def get_USZ_filepaths(path_to_source_directory: str, include_segmentation_masks: bool = True) -> OrderedDict[str, dict[str, str]]:
    """
    Get the filepaths of the scans and the segmentation masks of the USZ dataset
    
    Parameters
    ----------
    path_to_source_directory : str
        Path to the folder where the scans are stored
    include_segmentation_masks : bool
        Whether to include the segmentation masks or not
    
    Returns
    -------
    OrderedDict[str, dict[str, str]]
        Dictionary with the filepaths of the scans and the segmentation masks (optional)
    
    """
    scans_dict = OrderedDict()
    
    # get the filepaths of the scans
    tof_scans_fps = glob.glob(os.path.join(path_to_source_directory, '*', '*_tof.nii.gz'))
    
    for scan_fp in tof_scans_fps:
        scan_name = os.path.basename(os.path.dirname(scan_fp))
        scans_dict[scan_name] = {'tof': scan_fp}
    
    if include_segmentation_masks:
        seg_masks_fps = glob.glob(os.path.join(path_to_source_directory, '*', '*_seg.nii.gz'))
        assert len(tof_scans_fps) == len(seg_masks_fps), \
            "The number of scans and segmentation masks is not the same"
            
        # add the segmentation masks to the dictionary
        for seg_mask_fp in seg_masks_fps:
            scan_name = os.path.basename(os.path.dirname(seg_mask_fp))
            
            assert scan_name in scans_dict.keys(), \
                f"The scan {scan_name} does not have a TOF scan"
            scans_dict[scan_name]['seg'] = seg_mask_fp
        
    return scans_dict


def get_ADAM_filepaths(path_to_source_directory: str, include_segmentation_masks: bool = True) -> OrderedDict[str, dict[str, str]]:
    """
    Get the filepaths of the scans and the segmentation masks of the ADAM dataset
    
    Parameters
    ----------
    path_to_source_directory : str
        Path to the folder where the scans are stored
    include_segmentation_masks : bool
        Whether to include the segmentation masks or not
    
    Returns
    -------
    OrderedDict[str, dict[str, str]]
        Dictionary with the filepaths of the scans and the segmentation masks (optional)
    
    """
    scans_dict = OrderedDict()
    
    # get the filepaths of the scans
    tof_scans_fps = glob.glob(os.path.join(path_to_source_directory, '*', '*_TOF.nii.gz'))
    
    for scan_fp in tof_scans_fps:
        scan_name = os.path.basename(os.path.dirname(scan_fp))
        scans_dict[scan_name] = {'tof': scan_fp}
    
    if include_segmentation_masks:
        seg_masks_fps = glob.glob(os.path.join(path_to_source_directory, '*', '*_aneurysms.nii.gz'))
        assert len(tof_scans_fps) == len(seg_masks_fps), \
            "The number of scans and segmentation masks is not the same"
            
        # add the segmentation masks to the dictionary
        for seg_mask_fp in seg_masks_fps:
            scan_name = os.path.basename(os.path.dirname(seg_mask_fp))
            
            assert scan_name in scans_dict.keys(), \
                f"The scan {scan_name} does not have a TOF scan"
            scans_dict[scan_name]['seg'] = seg_mask_fp
        
    return scans_dict


def get_Laussane_filepaths(path_to_tof_dir: str, path_to_segmentation_dir: Optional[str]) -> OrderedDict[str, dict[str, str]]:
    """
    Get the filepaths of the scans and the segmentation masks of the Laussane dataset
    
    Note scan 482 does not have a segmentation mask, as it does not have an aneurysm
    
    Parameters
    ----------
    path_to_tof_dir : str
        Path to the folder where the TOF scans are stored
    path_to_segmentation_dir : str
        Path to the folder where the segmentation masks are stored
        
    Returns
    -------
    OrderedDict[str, dict[str, str]]
        Dictionary with the filepaths of the scans and the segmentation masks (optional)
    
    """
    scans_dict = OrderedDict()
    
    # get the filepaths of the scans
    tof_scans_fps = glob.glob(os.path.join(path_to_tof_dir, '*', '*', '*', '*_angio.nii.gz'))
    
    for scan_fp in tof_scans_fps:
        scan_name = scan_fp.split('/')[-4]
        scans_dict[scan_name] = {'tof': scan_fp}
        
    if path_to_segmentation_dir is not None:
        seg_masks_fps = glob.glob(os.path.join(path_to_segmentation_dir, '*', '*', '*', '*Lesion_1_mask.nii.gz'))
            
        # add the segmentation masks to the dictionary
        for seg_mask_fp in seg_masks_fps:
            scan_name = seg_mask_fp.split('/')[-4]
            
            assert scan_name in scans_dict.keys(), \
                f"The scan {scan_name} does not have a TOF scan"
            scans_dict[scan_name]['seg'] = seg_mask_fp
            
    return scans_dict
    

if __name__ == '__main__':
    usz_fp = '/scratch_net/biwidl319/jbermeo/data/raw/USZ'
    SIZE_USZ = 62
    
    adam_fp = '/scratch_net/biwidl319/jbermeo/data/raw/ADAM'
    SIZE_ADAM = 113
    
    laussane_tof_fp = '/scratch_net/biwidl319/jbermeo/data/raw/Lausanne/original_images'
    laussane_seg_fp = '/scratch_net/biwidl319/jbermeo/data/raw/Lausanne/skull_stripped_and_aneurysm_mask'
    SIZE_LAUSSANE = 38
    
    # Check if USZ dataset filepaths are indexed correctly
    test_usz_filepaths = get_USZ_filepaths(usz_fp, include_segmentation_masks=True)
    assert len(test_usz_filepaths) == SIZE_USZ, \
        f"The number of scans is not the expected one. It should be 62, but it is {len(test_usz_filepaths)}"
        
    # Check if ADAM dataset filepaths are indexed correctly
    test_adam_filepaths = get_ADAM_filepaths(adam_fp, include_segmentation_masks=True)
    assert len(test_adam_filepaths) == SIZE_ADAM, \
        f"The number of scans is not the expected one. It should be 30, but it is {len(test_adam_filepaths)}"
        
    # Check if Laussane dataset filepaths are indexed correctly
    test_laussane_filepaths = get_Laussane_filepaths(laussane_tof_fp, laussane_seg_fp)
    assert len(test_laussane_filepaths) == SIZE_LAUSSANE, \
        f"The number of scans is not the expected one. It should be 38, but it is {len(test_laussane_filepaths)}"
        
    print('All tests passed!')
