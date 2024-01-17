import numpy as np
import nibabel as nib
import skimage.transform as ski_trf


def resize_segmentation(segmentation:np.ndarray, new_shape: tuple[int, ...], order: int = 3):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    
    taken from: https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/utils.py#L22
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return ski_trf.resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = ski_trf.resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        
        return np.round(reshaped).astype(np.int8)


def resample(
    nii_img: nib.nifti1.Nifti1Image, 
    new_voxel_size: tuple[float, float, float],
    order: int = 3,
    is_segmenation: bool = False,
    ) -> nib.nifti1.Nifti1Image:
    """
    Resample a nifti image to a new voxel size with scikit-image

    Parameters
    ----------
    nii_img : nib.nifti1.Nifti1Image
        Image to resample
    new_voxel_size : tuple[float, float, float]
        New voxel size
    order : int, optional
        Order of the spline interpolation, by default 3
    is_segmenation : bool, optional
        Whether the image is a segmentation mask, by default False. If True, the interpolation is done either with
        nearest neighbor or each class is interpolated independently and only voxels with a probability > 0.5 are kept
    """
    old_dims       = nii_img.header.get_data_shape()
    old_voxel_size = nii_img.header.get_zooms()
  
    assert len(old_dims) == len(old_voxel_size) == len(new_voxel_size), \
        "New voxel size has to have the same dimension as the old voxel size"
    
    # New shape due to change is voxel size
    new_shape = [int(old_dims[i] * old_voxel_size[i]/new_voxel_size[i]) for i in range(len(new_voxel_size))]
  
    resize_fn = lambda x: resize_segmentation(x, new_shape, order) if is_segmenation \
        else ski_trf.resize(x, new_shape, order, mode='edge', anti_aliasing=False)
    
    img_array = nii_img.get_fdata()
    resized_img_array = resize_fn(img_array)
    
    new_nii_img = nib.Nifti1Image(resized_img_array, affine=nii_img.affine, header=nii_img.header)
        
    return new_nii_img
