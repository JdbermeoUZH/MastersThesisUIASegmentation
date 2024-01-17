"""_
Contains functions for bias correction of MR images.

"""

import SimpleITK as sitk


def N4bias_correction_filter(
    img_init: sitk.Image, 
    image_mask_flag: bool          = True,
    shrink_factor: int             = 1,
    max_num_iterations: list[int]  = None
    )-> tuple[sitk.Image, sitk.Image]:
    '''
    following the official documentation:
    https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html
    '''
    # make sure image is 32bit float
    inputImg   = sitk.Cast(img_init, sitk.sitkFloat32)
    image      = inputImg
    image_mask = None
    if image_mask_flag == True: image_mask = sitk.OtsuThreshold(image, 0, 1)

    if shrink_factor > 1:
        image      = sitk.Shrink(inputImg, [shrink_factor] * inputImg.GetDimension())
        if image_mask != None:
            image_mask = sitk.Shrink(image_mask, [shrink_factor] * inputImg.GetDimension()) 
        
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetNumberOfControlPoints([4,4,4])
    corrector.SetConvergenceThreshold(1e-6)

    if max_num_iterations != None:
        corrector.SetMaximumNumberOfIterations(max_num_iterations)
    
    # This part actually executes the n4 bias correction    
    if image_mask != None: _ = corrector.Execute(image, image_mask)
    else:                  _ = corrector.Execute(image)
    
    log_bias_field  = corrector.GetLogBiasFieldAsImage(inputImg)
    log_bias_field  = sitk.Cast(log_bias_field, sitk.sitkFloat64)
    
    corrected_image_full = img_init / sitk.Exp(log_bias_field)
    
    return corrected_image_full, log_bias_field
