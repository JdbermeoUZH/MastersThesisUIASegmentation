from typing import Optional, Union

import SimpleITK as sitk


#---------- Default hyperparameters
mmi_n_bins_default              = 50
learning_rate_default           = 1.0
number_of_iterations_default    = 100
#----------


def rigid_registration(
    fixed_image_path: str, 
    moving_image_path: str, 
    image_segmentation_mask_path: Optional[str] = None,
    use_geometrical_center_mode: bool = True,
    mmi_n_bins: int = mmi_n_bins_default,
    learning_rate: float = learning_rate_default,
    number_of_iterations: int = number_of_iterations_default,
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
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=mmi_n_bins)

    # Set the optimizer
    registration_method.SetOptimizerAsGradientDescent(learningRate=learning_rate, numberOfIterations=number_of_iterations,
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
    resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkBSpline, 0.0, moving_image.GetPixelID())
    
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


