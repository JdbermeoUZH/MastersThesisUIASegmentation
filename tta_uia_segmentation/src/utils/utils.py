import os
import gc
import random
from typing import Optional, Union, Literal, Tuple, Any

import torch
import torch.distributed
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import numpy as np
import nibabel as nib
import nibabel.processing as nibp


def define_device(device: str, print_device: bool = False) -> torch.device:
    if device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu') # type: ignore
        if print_device: print('No GPU available, using CPU instead')
    else:
        device = torch.device(device) # type: ignore

    if print_device: print(f'Using Device {device}')
    
    return device # type: ignore


def assert_in(value, name, possible_values):
    assert value in possible_values, \
        f'{name} must be in {possible_values} but is {value}'


def normalize_percentile(data, min_p=0, max_p=100):
    min = np.percentile(data, min_p)
    max = np.percentile(data, max_p)
    return normalize(data, min, max)


def normalize(data, min=None, max=None):
    if min is None:
        min = np.min(data)
    if max is None:
        max = np.max(data)

    if max == min:
        data = np.zeros_like(data)
    else:
        data = (data - min) / (max - min)
    data = np.clip(data, 0, 1)
    data = (255 * data).astype(np.uint8)
    return data


def get_seed():
    seed = random.randint(0, 2**64 - 1)  # Random 64 bit integer
    return seed

def from_dict_or_default(dict_, key, d):
    if key in dict_:
        return dict_[key]
    return d

def seed_everything(seed:int=0) -> None:
    """
    Seed method for PyTorch for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_generator(seed, library='torch'):
    assert_in(library, 'library', ['torch', 'numpy'])
    if library == 'torch':
        return torch.Generator().manual_seed(seed)
    elif library == 'numpy':
        return np.random.default_rng(seed)
    

def get_generators(num_generators=1, seed=None, library='torch'):
    assert_in(library, 'library', ['torch', 'numpy'])
    if num_generators == 0:
        return []
    if seed is None:
        seed = get_seed()

    generators = [
        get_generator(seed, library=library) for _ in range(num_generators)
    ]
    return generators


def uniform_interval_sampling(interval_starts, interval_ends, rng):

    assert interval_starts.shape == interval_ends.shape
    n_intervals = len(interval_starts)

    indices = np.arange(n_intervals)
    interval_lengths = interval_ends - interval_starts

    if all(interval_lengths == 0):
        interval_lengths = np.ones_like(interval_lengths)
    
    interval_lengths /= sum(interval_lengths)
    index = rng.choice(indices, p=interval_lengths)

    sample = rng.uniform(interval_starts[index], interval_ends[index])
    return sample


def random_select_from_tensor(tensor, dim):
    assert isinstance(tensor, torch.Tensor), f'tensor is of type {type(tensor)} but should be torch.Tensor'

    n_elements = tensor.shape[dim]
    index = np.random.randint(n_elements)
    return tensor.select(dim, index)


def resize_volume(
    x: torch.Tensor,
    target_pix_size: tuple[float, float, float] | torch.Tensor,
    current_pix_size: tuple[float, float, float] | torch.Tensor,
    target_img_size: Optional[tuple[int, int, int]] = None,
    input_format: Literal['NCDHW', 'DCHW', 'CDHW'] = 'NCDHW',
    output_format: Literal['NCDHW', 'DCHW'] = 'NCDHW',
    mode: Literal['trilinear', 'bilinear', 'nearest'] = 'trilinear',
    only_inplane_resample: bool = True,
    target_size_before_resample: Optional[tuple[int, int, int]] = None,
) -> torch.Tensor:
    """
    Resize a volume tensor to a target pixel size.

    Parameters
    ----------
    x : torch.Tensor
        Input volume tensor.
    target_pix_size : tuple[float, float, float] | torch.Tensor
        Target pixel size in (D, H, W) format.
    current_pix_size : tuple[float, float, float] | torch.Tensor
        Current pixel size in (D, H, W) format.
    target_img_size : tuple[int, int, int]
        Target image size in (D, H, W) format.
    vol_format : Literal['NCDHW', 'DCHW', 'CDHW'], optional
        Input volume format, by default 'NCDHW'.
    output_format : Literal['NCDHW', 'DCHW'], optional
        Desired output format, by default 'NCDHW'.
    mode : Literal['trilinear', 'nearest'], optional
        Interpolation mode, by default 'trilinear'.

    Returns
    -------
    torch.Tensor
        Resized volume tensor in the specified output format.
    """
    
    # Convert tensor NCDHW format
    # :===============================================:
    assert len(input_format) == len(x.shape), f"x has {len(x.shape)} dimensions ({x.shape}) " + \
        f"while format has {len(input_format)} ({input_format})"
    
    if input_format == 'DCHW':
        # Convert to NCDHW
        x = from_DCHW_to_NCDHW(x, N=1)
            
    elif input_format == 'CDHW':
        # Convert to NCDHW
        x = x.unsqueeze(0)

    elif input_format == 'NCDHW':
        pass
    
    else:
        raise ValueError(f"Unsupported volume format: {input_format}")
    
    # Crop or pad to intermediate size (if specified)
    # :===============================================:
    if target_size_before_resample is not None:
        crop_or_pad_necessary = tuple(x.shape[-3:]) != tuple(target_size_before_resample)
        x = crop_or_pad_to_size(x, target_size_before_resample)
    

    # Resample/resize volume
    # :===============================================:
    scale_factor = torch.Tensor(current_pix_size) / torch.Tensor(target_pix_size)       # (D, H, W)
    resampled_size = (torch.tensor(x.shape[-3:]) * scale_factor).round().int().tolist() # (D, H, W)

    if only_inplane_resample:
        if target_img_size is not None:
            assert x.shape[-3] == target_img_size[0], \
                "Target depth must match depth of 'x' for only_inplane_resize=True"
        assert mode in ['bilinear', 'nearest'], "Only 'bilinear' and 'nearest' modes are supported for only_inplane_resample=True" 
    
        n_spatial_dims = 2
        scale_factor = scale_factor[-n_spatial_dims:]
        resampled_size = resampled_size[-n_spatial_dims:]
        resample_necessary = (scale_factor != 1).any()
        
        if resample_necessary:
            x = from_NCDHW_to_DCHW(x)
            x = F.interpolate(x, size=resampled_size, mode=mode)
            x = from_DCHW_to_NCDHW(x, N=1)
    else:
        n_spatial_dims = 3
        resample_necessary = (scale_factor != 1).any()
        
        if resample_necessary:
            x = F.interpolate(x, size=resampled_size, mode=mode)

    # Crop or pad to target size
    # :===============================================:
    if target_img_size is not None:
        crop_or_pad_necessary = tuple(x.shape[-n_spatial_dims:]) != tuple(target_img_size[-n_spatial_dims:])
        
        if crop_or_pad_necessary:
            x = crop_or_pad_to_size(x, target_img_size)

    # Convert tensor to output format
    # :===============================================:
    if output_format == 'DCHW':
        x = from_NCDHW_to_DCHW(x)
    elif output_format == 'NCDHW':
        pass
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    return x


def resize_image(
    x: torch.Tensor,
    target_pix_size: tuple[float, float] | torch.Tensor,
    current_pix_size: tuple[float, float] | torch.Tensor,
    img_format: Literal['CHW', 'NCHW'] = 'NCHW',
    output_format: Literal['CHW', 'NCHW'] = 'NCHW',
    target_img_size: Optional[tuple[int, int]] = None,
    mode: Literal['bilinear', 'nearest'] = 'bilinear',
    target_size_before_resample: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Resize an image tensor to a target pixel size.

    Parameters
    ----------
    x : torch.Tensor
        Input image tensor.
    target_pix_size : tuple[float, float] | torch.Tensor
        Target pixel size in (H, W) format.
    current_pix_size : tuple[float, float] | torch.Tensor
        Current pixel size in (H, W) format.
    target_img_size : tuple[int, int]
        Target image size in (H, W) format.
    mode : Literal['bilinear', 'nearest'], optional
        Interpolation mode, by default 'bilinear'.

    Returns
    -------
    torch.Tensor
        Resized image tensor.
    """


    # Convert tensor to NCHW format
    # :===============================================:
    assert len(img_format) == len(x.shape), f"x has {len(x.shape)} dimensions ({x.shape}) " + \
        f"while format has {len(img_format)} ({img_format})"
    
    if img_format == 'CHW':
        # Convert to NCHW
        x = x.unsqueeze(0)

    elif img_format == 'NCHW':
        pass
    

    # Crop or pad to intermediate size (if specified)
    # :===============================================:
    if target_size_before_resample is not None:
        crop_or_pad_necessary = tuple(x.shape[-2:]) != tuple(target_size_before_resample)
        x = crop_or_pad_to_size(x, target_size_before_resample)

    # Resample/resize volume
    # :===============================================:
    scale_factor = torch.Tensor(current_pix_size) / torch.Tensor(target_pix_size)       
    resampled_size = (torch.tensor(x.shape[-2:]) * scale_factor).round().int().tolist() 

    resample_necessary = (scale_factor != 1).any()

    if resample_necessary:
        x = F.interpolate(x, size=resampled_size, mode=mode)

    # Crop or pad to target size
    # :===============================================:
    if target_img_size is not None:
        crop_or_pad_necessary = tuple(x.shape[-2:]) != tuple(target_img_size)
        
        if crop_or_pad_necessary:
            x = crop_or_pad_to_size(x, target_img_size)

    # Convert tensor to output format
    # :===============================================:
    if output_format == 'CHW':
        x = x.squeeze(0)
    elif output_format == 'NCHW':
        pass

    return x


def resize_and_resample_nibp(
    image: np.ndarray, 
    target_size: tuple[int, ...],
    original_voxel_size: tuple[float, ...], 
    target_voxel_size: tuple[float, ...],
    order: int
    ) -> np.ndarray:
    """
    Resizes and resamples a 3D image to a given voxel size and target size.
    """
    # Convert image to nibabel image
    affine_matrix = np.eye(4)
    affine_matrix[0,0] = original_voxel_size[0]
    affine_matrix[1,1] = original_voxel_size[1]
    affine_matrix[2,2] = original_voxel_size[2]
    
    # If the image is 3D with 3 dim array, just resample it
    if len(image.shape) == 3:
        return nibp.conform(
            nib.Nifti1Image(image, affine_matrix), target_size, target_voxel_size, order=order
        ).get_fdata()
        
    # If the image has multiple channels, resample each channel
    new_img_channels = []
    for channel_i in range(image.shape[0]):   
        nib_img = nib.Nifti1Image(image[channel_i], affine_matrix)
        new_img_channels.append(
            nibp.conform(nib_img, target_size, target_voxel_size, order=order).get_fdata()
        )
    
    return np.stack(new_img_channels, axis=0)


def crop_or_pad_to_size(data: torch.Tensor, target_size: Union[Tuple[int, int], Tuple[int, int, int]]) -> torch.Tensor:
    """
    Crops or pads batched data to a given size in 2D (NCHW) or 3D (NCDHW).\
    
    Residual from uneven cropping/padding is added to the right. 
    
    Parameters:
    data (torch.Tensor): Input data to be cropped or padded.
    target_size (tuple): Target size (H, W) for 2D or (D, H, W) for 3D.
    
    Returns:
    torch.Tensor: Cropped or padded data.
    """
    if not isinstance(data, torch.Tensor):
        raise ValueError("Input data must be a torch.Tensor")
    
    input_shape = data.shape
    ndim = len(input_shape)
    
    if ndim not in [4, 5]:
        raise ValueError("Input data must be 4D (NCHW) or 5D (NCDHW)")
    
    if len(target_size) != ndim - 2:
        raise ValueError("Target size must match the spatial dimensions of input data")
    
    spatial_shape = input_shape[2:]
    
    # Calculate padding or cropping for each dimension
    diff = [t - s for t, s in zip(target_size, spatial_shape)]
    pad_crop = [(d // 2, d - d // 2) if d > 0 else (d - d // 2, d // 2)  for d in diff]
    
    # Perform padding or cropping
    pad_crop = [(0, 0), (0, 0)] + pad_crop  # Add N and C dimensions
    padded_cropped = F.pad(data, [item for sublist in reversed(pad_crop) for item in sublist])

    assert tuple(padded_cropped.shape[-len(target_size):]) == tuple(target_size), f"Output shape {padded_cropped.shape} does not match target size {target_size}"
    
    return padded_cropped


def distribute_n_in_m_slots(n, m):
    elements_per_slot = n // m
    slots_with_extra_element = n % m
    equitably_dist_list = slots_with_extra_element * [elements_per_slot + 1] + \
        (m - slots_with_extra_element) * [elements_per_slot]
    np.random.shuffle(equitably_dist_list)

    return equitably_dist_list


def stratified_sampling(a, b, m, n, return_torch: bool = True) -> Union[np.ndarray|torch.Tensor]:
    """
    Perform stratified sampling of n integers in a range [a, b] split into m buckets.

    Parameters:
    a (int): Lower bound of the range (inclusive)
    b (int): Upper bound of the range (inclusive)
    m (int): Number of buckets to stratify on
    n (int): Number of samples to draw

    Returns:
    samples (np.ndarray|toch.Tensor): Tensor of n stratified samples
    """
    assert b > a, "Upper bound must be greater than lower bound"
    assert m > 0, "Number of buckets must be greater than 0"
    
    if n < m:
        raise ValueError("Warning: Number of samples is less than number of groups to stratify on")
    
    bucket_size = (b - a) / (m)
    buckets = np.arange(a, b + 1, bucket_size)
    buckets = [(np.round(buckets[i]), np.round(buckets[i+1])) 
               for i in range(len(buckets) - 1)]
    np.random.shuffle(buckets)

    draws = distribute_n_in_m_slots(n, m)
    draws = [i for i in draws if i > 0]

    sampled_t = []
    for draw_n in draws:
        bucket_start, bucket_end = buckets.pop(0)
        sampled_t.append(
            np.random.randint(bucket_start, bucket_end, (draw_n,))
        )
    
    sample = np.concatenate(sampled_t)
    np.random.shuffle(sample)
    
    if return_torch:
        return torch.from_numpy(sample)
    else:
        return sample


def clone_state_dict_to_cpu(state_dict: dict) -> dict:
    """
    Convert state_dict to CPU.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if value.device.type == 'cpu':
            new_state_dict[key] = value.clone()
        else:
            new_state_dict[key] = value.cpu()
    return new_state_dict


def generate_2D_dl_for_vol(*vols: torch.Tensor, batch_size: int, num_workers: int, **dl_kwargs) -> torch.utils.data.DataLoader:
    """
    Generate a 2D dataloader for one or more 3D volume tensors.

    This function takes one or more 5D tensors representing 3D volumes (BCDHW format) and
    creates a DataLoader that yields 2D slices of the volumes.

    Parameters
    ----------
    *vols : torch.Tensor
        Input tensors, each of shape (1, C, D, H, W).
    batch_size : int
        Number of slices per batch.
    num_workers : int
        Number of worker processes for data loading.
    **kwargs
        Additional arguments to pass to the DataLoader constructor.

    Returns
    -------
    torch.utils.data.DataLoader
        A DataLoader that yields 2D slices of the input volumes.
        Each item in the DataLoader will be a tuple of tensors, each with shape (1, C, H, W).

    Examples
    --------
    >>> dl = generate_2D_dl_for_vol(x, bg, batch_size=4, num_workers=2)
    >>> for x_b, bg_b in dl:
    ...     # Process x_b and bg_b
    ...     pass

    """

    processed_vols = []
    for x in vols:
        processed_vols.append(
            from_NCDHW_to_DCHW(x)
        )
    
    # Ensure all tensors have the same number of slices
    assert all(t.size(0) == processed_vols[0].size(0) for t in processed_vols), \
        "All input tensors must have the same number of slices"
    
    return torch.utils.data.DataLoader(
        TensorDataset(*processed_vols),
        batch_size=batch_size,
        num_workers=num_workers,
        **dl_kwargs
    )

def from_DCHW_to_NCDHW(x: torch.Tensor, N: int = 1) -> torch.Tensor:
    assert x.dim() == 4, f"Input tensor must be 4D but is {len(x.shape)}D"
    return x.reshape(N, -1, *x.shape[-3:]).permute(0, 2, 1, 3, 4)

def from_NCDHW_to_DCHW(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 5, f"Input tensor must be 5D but is {len(x.shape)}D"
    x = x.permute(0, 2, 1, 3, 4)
    return x.reshape(-1, *x.shape[-3:])


def parse_bool(value: str) -> bool:
    """
    Parse a string value to a boolean.

    Parameters
    ----------
    value : str
        String value to parse.

    Returns
    -------
    bool
        Boolean value.
    """
    assert value.lower() in ['true', 'false'], "Value must be 'true' or 'false'"
    return value.lower() == 'true'


def torch_to_numpy(*tensors: torch.Tensor) -> np.ndarray | tuple[np.ndarray]:
    """
    Convert one or more PyTorch tensors to NumPy arrays.

    Parameters
    ----------
    *tensors : torch.Tensor
        Input tensors to convert.

    Returns
    -------
    np.ndarray
        NumPy array representation of the input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].detach().cpu().numpy()
    
    return tuple(t.detach().cpu().numpy() for t in tensors)


def exists(x):
    return x is not None


def default(val: Optional[Any], d: Any) -> Any:
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)


# def is_main_process():
#     from accelerate.state import PartialState
#     return PartialState().is_main_process


def is_main_process():
    # Check if the distributed environment is initialized
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # The main process usually has rank 0
        return torch.distributed.get_rank() == 0
    elif "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    else:
        return True
    

def print_if_main_process(string: str):
    if is_main_process():
        print(string)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def garbage_collection():
    gc.collect()
    torch.cuda.empty_cache() 


def nan_sum(*tensors):
    # Step 1: Replace NaNs with zero in each tensor
    tensors_with_nan_as_zero = [torch.nan_to_num(t, nan=0.0) for t in tensors]
    
    # Step 2: Element-wise sum of the tensors with NaNs replaced by zero
    result = sum(tensors_with_nan_as_zero)
    
    # Step 3: Create a mask for positions where all tensors have NaN
    all_nan_mask = torch.ones_like(tensors[0], dtype=bool)
    for t in tensors:
        all_nan_mask &= torch.isnan(t)
    
    # Set result to NaN where all tensors had NaN
    result[all_nan_mask] = float('nan')
    
    return result


def min_max_normalize_channelwise(tensor: torch.Tensor, spatial_dims: tuple[int, ...], eps = 1e-8) -> torch.Tensor:
    min_vals = tensor.amin(dim=spatial_dims, keepdim=True)
    max_vals = tensor.amax(dim=spatial_dims, keepdim=True)
    
    # Perform the min-max normalization
    normalized_tensor = (tensor - min_vals) / (max_vals - min_vals + eps)  # Add epsilon to avoid division by zero
    
    return normalized_tensor