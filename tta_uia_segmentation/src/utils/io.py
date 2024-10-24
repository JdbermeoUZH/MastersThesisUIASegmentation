import os
import csv
import pickle
import numbers
import argparse
from typing import Optional

import yaml
import torch
import imageio
import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib

from tta_uia_segmentation.src.utils.utils import assert_in

def env_var_constructor(loader, node):
    value = loader.construct_scalar(node)
    return os.path.expandvars(value)


def load_config(path):
    yaml.add_constructor('tag:yaml.org,2002:str', env_var_constructor, Loader=yaml.SafeLoader)

    # Register the constructor for any string value
    with open(path, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)
    return cfg


def rewrite_config_arguments(
    config: dict,
    args: argparse.Namespace, 
    config_name: str,
    prefix_to_remove: Optional[str] = None, 
    ) -> dict:
    for key, value in vars(args).items():
        if value is not None:
            key = key.replace(prefix_to_remove, '') if prefix_to_remove is not None else key
            if key in config:
                config[key] = value

    return config


def dump_config(path, cfg):
    with open(path, 'w') as file:
        yaml.dump(cfg, file, indent=4)
        

def print_config(params, keys=None):
    if keys is None:
        keys = params.keys()
    elif isinstance(keys, str):
        keys = [keys]
    elif not isinstance(keys, list):
        raise NotImplementedError()

    print(yaml.dump(
        {k: params[k] for k in keys}, 
        indent=4, explicit_start=True, explicit_end=True
    ))   
    

# Function taken from https://stackoverflow.com/a/43621819
def deep_get(_dict, *keys, default=None, suppress_warning=True):
    for key in keys:
        if isinstance(_dict, dict) and key in _dict.keys():
            _dict = _dict.get(key, default)
        else:
            if not suppress_warning:
                print(f'Warning: Parameter {"/".join(keys)} not found and set to {default}')
            return default
    return _dict


def add_to_yaml(path, added_data):
    assert isinstance(added_data, dict)

    
    if os.path.exists(path):
        score_dict = load_config(path)
    else:
        score_dict = {}

    for key, value in added_data.items():
        score_dict[key] = value

    dump_config(path, score_dict)

    
def write_to_csv(path, data, header=None, mode='a'):
    assert_in(mode, 'mode', ['w', 'a'])

    existed = os.path.exists(path)

    with open(path, mode) as f:
        writer = csv.writer(f)

        if header is not None and (mode == 'w' or not existed):
            writer.writerow(header)

        for row in data:
            writer.writerow(row)


def read_csv(path, split_header=False, convert_to_num=False, dtype=float, **kwargs):
    data = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, **kwargs)
        for r in reader:
            data.append(r)

    header = []
    if split_header:
        header = data[0]
        data = data[1:]

    data = np.array(data)

    if convert_to_num:
        data = data.astype(dtype)

    return data, header


def save_checkpoint(path, **kwargs):
    torch.save(kwargs, path)


def write_pickle(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def create_directory(*args):
    if len(args) == 0:
        return '.'
    else:
        path = os.path.join(*args)
        path = os.path.expanduser(path)

        if not os.path.exists(path):
            os.makedirs(path)

        return path


def number_or_list_to_array(x):
    if isinstance(x, numbers.Number):
        return np.array([x])
    elif isinstance(x, list):
        return np.array(x)
    

def save_plt_image(dir, filename, image, dpi=300, **kwargs):
    os.makedirs(dir, exist_ok=True)
    plt.imshow(image, interpolation='none', **kwargs)
    plt.axis('off')
    plt.savefig(os.path.join(dir, filename), dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_image(dir, filename, image):
    os.makedirs(dir, exist_ok=True)
    imageio.imwrite(os.path.join(dir, filename), image)
    
    
def save_nii_image(dir, filename, image, affine=None):
    os.makedirs(dir, exist_ok=True)
    image = nib.Nifti1Image(image, affine)
    nib.save(image, os.path.join(dir, filename))
    