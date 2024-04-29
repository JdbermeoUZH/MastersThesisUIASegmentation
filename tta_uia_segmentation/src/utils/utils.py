import os
import random
from typing import Union

import torch
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


def crop_or_pad_slice_to_size(slice, nx, ny):
    """
    Crops or pads a slice to a given size in x, y, and z.
    
    Adapted from https://github.com/neerakara/test-time-adaptable-neural-networks-for-domain-generalization/blob/master/utils.py#L91  
    """
    z, x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[:, x_s: x_s + nx, y_s: y_s + ny]
    else:
        if isinstance(slice, torch.Tensor): 
            slice_cropped = torch.zeros((z, nx, ny,))
        else:
            slice_cropped = np.zeros((z, nx, ny))
        
        if x <= nx and y > ny:
            slice_cropped[:, x_c:x_c + x, :] = slice[:, :, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, :, y_c:y_c + y] = slice[:, x_s:x_s + nx, :]
        else:
            slice_cropped[:, x_c:x_c + x, y_c:y_c + y] = slice[:, :, :]

    return slice_cropped


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
    
    bucket_size = (b - a + 1) // m
    buckets = np.arange(a, b + 1, bucket_size)
    buckets = [(buckets[i], buckets[i+1] + 1) for i in range(len(buckets) - 1)]
    np.random.shuffle(buckets)

    draws = distribute_n_in_m_slots(n, m)
    draws = [i for i in draws if i > 0]

    sampled_t = []
    for draw_n in draws:
        bucket_start, bucket_end = buckets.pop(0)
        sampled_t.append(
            np.random.randint(bucket_start, bucket_end, (draw_n,))
        )
        
    if return_torch:
        return torch.from_numpy(np.concatenate(sampled_t))
    else:
        return np.concatenate(sampled_t)