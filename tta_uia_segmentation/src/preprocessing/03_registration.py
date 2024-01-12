import os
import tqdm
import logging
import argparse
from typing import Optional, Union
from datetime import datetime

import SimpleITK as sitk

from utils import get_filepaths

#---------- paths & hyperparameters
shrink_factor_default       = 1
max_num_iters_default       = [50, 50, 50] 
path_to_logs                = '/scratch_net/biwidl319/jbermeo/MastersThesisUIASegmentation/logs/preprocessing/resampling'
path_to_save_processed_data = '/scratch_net/biwidl319/jbermeo/data/preprocessed/0_bias_corrected'
#----------


date_now = datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(path_to_logs, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    filename=os.path.join(path_to_logs, f'{date_now}_registration.log'), filemode='w')
log = logging.Logger('Registration')


def preprocess_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Bias correction of scans')
    
    parser.add_argument('--shrink_factor', type=int, default=shrink_factor_default)
    parser.add_argument('--maximum_number_of_iterations', type=int, nargs='+', default=max_num_iters_default)
    
    parser.add_argument('--preprocessed', action='store_true', default=False)
    parser.add_argument('--path_to_dir', type=str)
    
    parser.add_argument('--dataset', type=str, choices=['USZ', 'ADAM', 'Laussane', None])
    parser.add_argument('--path_to_tof_dir', type=str, required=True)
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

import SimpleITK as sitk


def rigid_registration(
    fixed_image_path: str, 
    moving_image_path: str, 
    image_segmentation_mask_path: Optional[str] = None,
    use_geometrical_center_mode: bool = True
    ) -> Union[sitk.Image, tuple[sitk.Image, sitk.Image]]:
    """
    Perform rigid registration between a fixed image and a moving image using SimpleITK.

    Parameters
    ----------
    fixed_image_path : str
        Path to the fixed image.
    
    moving_image_path : str
        Path to the moving image.
        
    output_image_path : str, optional
        Path to save the registered image. If None, the image is not saved.
        
    Returns
    -------
    resampled_image : SimpleITK.Image
        The registered image.
    """

    # Read the images
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    
    # Read the segmentation mask, if provided
    if image_segmentation_mask_path is not None:
        image_segmentation_mask = sitk.ReadImage(image_segmentation_mask_path, sitk.sitkFloat32)
    else:
        image_segmentation_mask = None

    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Set the metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    # Set the optimizer
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Set the initial transformation
    centering_mode = sitk.CenteredTransformInitializerFilter.GEOMETRY if use_geometrical_center_mode \
        else sitk.CenteredTransformInitializerFilter.MOMENTS
    
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, 
                                                          sitk.Euler3DTransform(), 
                                                          centering_mode)
    registration_method.SetInitialTransform(initial_transform)

    # Set the interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Execute the registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Apply the final transformation
    resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    
    # Apply the final transformation to the segmentation mask, if provided
    #  The resampling should be of order zero, so that the segmentation mask is not interpolated
    if image_segmentation_mask is not None:
        resampled_image_segmentation_mask = sitk.Resample(
            image_segmentation_mask,
            fixed_image,
            final_transform,
            sitk.sitkNearestNeighbor,
            0.0,
            image_segmentation_mask.GetPixelID()
        )
        
        return resampled_image, resampled_image_segmentation_mask
    
    else:
        return resampled_image
