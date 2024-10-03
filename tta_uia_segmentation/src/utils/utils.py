import os
import random
from typing import Optional, Union, Literal, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import nibabel.processing as nibp


def define_device(device: str) -> torch.device:
    if device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
        print('No GPU available, using CPU instead')
    else:
        device = torch.device(device)

    print(f'Using Device {device}')
    
    return device


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
    vol_format: Literal['1CDHW', 'DCHW', 'CDHW'] = '1CDHW',
    output_format: Literal['1CDHW', 'DCHW'] = '1CDHW',
    mode: Literal['trilinear', 'nearest'] = 'trilinear',
    only_inplane_resample: bool = True,
    order_operations: Literal['crop_then_resize', 'resize_then_crop'] = 'resize_then_crop'
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
    vol_format : Literal['1CDHW', 'DCHW', 'CDHW'], optional
        Input volume format, by default '1CDHW'.
    output_format : Literal['1CDHW', 'DCHW'], optional
        Desired output format, by default '1CDHW'.
    mode : Literal['trilinear', 'nearest'], optional
        Interpolation mode, by default 'trilinear'.

    Returns
    -------
    torch.Tensor
        Resized volume tensor in the specified output format.
    """
    
    # Convert tensor 1CDHW format
    # ---------------------------
    assert len(vol_format) == len(x.shape), f"x has {len(x.shape)} dimensions ({x.shape}) " + \
        f"while format has {len(vol_format)} ({vol_format})"
    
    if vol_format == 'DCHW':
        # Convert to 1CDHW
        assert len(x.shape) == 4, f"x does not have 4 dimensions (format specified is {vol_format})"
        x = x.permute(1, 0, 2, 3).unsqueeze(0)
            
    elif vol_format == 'CDHW':
        # Convert to 1CDHW
        assert len(x.shape) == 4, f"x does not have 4 dimensions (format specified is {vol_format})"
        x = x.unsqueeze(0)

    elif vol_format == '1CDHW':
        assert len(x.shape) == 5, f"x does not have 5 dimensions (format specified is {vol_format})"
        pass
    
    else:
        raise ValueError(f"Unsupported volume format: {vol_format}")
    

    # Resample/resize volume
    # ----------------------
    scale_factor = torch.Tensor(current_pix_size) / torch.Tensor(target_pix_size)       # (D, H, W)
    resampled_size = (torch.tensor(x.shape[-3:]) * scale_factor).round().int().tolist() # (D, H, W)

    if only_inplane_resample:
        scale_factor[0] = 1.0
        resampled_size[0] = x.shape[-3]
        if target_img_size is not None:
            assert resampled_size[0] == target_img_size[0], \
                "Target depth must match depth of 'x' for only_inplane_resize=True"
        n_spatial_dims = 2
    else:
        n_spatial_dims = 3
    
    resample_necessary = (scale_factor != 1).any()
    crop_or_pad_necessary = target_img_size is not None and \
        tuple(x.shape[-n_spatial_dims:]) != tuple(target_img_size[-n_spatial_dims:])

    if order_operations == 'crop_then_resize':
        # Crop or pad to target size
        if crop_or_pad_necessary:
            x = crop_or_pad_to_size(x, target_img_size)
    
        # Resample volume 
        if resample_necessary:
            x = F.interpolate(x, size=resampled_size, mode=mode)

    elif order_operations == 'resize_then_crop':
        # Resample volume 
        if resample_necessary:
            x = F.interpolate(x, size=resampled_size, mode=mode)

        # Crop or pad to target size
        if crop_or_pad_necessary:
            x = crop_or_pad_to_size(x, target_img_size)

    else:
        raise ValueError(f"Unsupported order of operations: {order_operations}")


    # Convert tensor to output format
    # -------------------------------
    if output_format == 'DCHW':
        x = x.squeeze(0).permute(1, 0, 2, 3)
    elif output_format == '1CDHW':
        pass
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
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
        print("Warning: Number of samples is less than number of groups to stratify on")
    
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


def state_dict_to_cpu(state_dict: dict) -> dict:
    """
    Convert state_dict to CPU.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key] = value.cpu()
    return new_state_dict


def generate_2D_dl_for_vol(*vols: torch.Tensor, batch_size: int, num_workers: int, **kwargs) -> torch.utils.data.DataLoader:
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
        assert x.dim() == 5, f"Input tensor must be 5D but is {len(x.shape)}D"
        x = x.squeeze(0)  # Remove batch dimension
        x = x.permute(1, 0, 2, 3)  # Permute to DCHW format
        processed_vols.append(x)

    # Ensure all tensors have the same number of slices
    assert all(t.size(0) == processed_vols[0].size(0) for t in processed_vols), \
        "All input tensors must have the same number of slices"

    # Create a custom dataset that iterates over multiple volumes along N dimension
    class MultipleVolumeDataset(torch.utils.data.Dataset):
        def __init__(self, *vols):
            self.vols = vols

        def __getitem__(self, index):
            return tuple(vol[index] for vol in self.vols)

        def __len__(self):
            return self.vols[0].size(0)

    dataset = MultipleVolumeDataset(*processed_vols)
    
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        **kwargs
    )

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


def torch_to_numpy(*tensors: torch.Tensor) -> np.ndarray:
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