import os
import time
from typing import Optional

import wandb

from tta_uia_segmentation.src.utils.io import load_config, dump_config

def prepend_date_time(base_name):
    # Get the current time in the format MM_DD_HHMM
    date_time = time.strftime("%m_%d_%H%M")
    # Prepend the date and time to the base name
    return f"{date_time}_{base_name}"

def update_dict(*keys, value, dict):
    """
    Update the dictionary with the new value at the specified nested keys.
    
    Parameters
    ----------
    *keys : str
        The keys to the value in the dictionary.
        
    value : any
        The new value to set.
    """

    sub_dict = dict
    for key in keys[:-1]:
        if key not in sub_dict:
            sub_dict[key] = {}  # Create nested dictionaries if they don't exist
        sub_dict = sub_dict[key]
    
    # Set the final value
    sub_dict[keys[-1]] = value


def update_wandb_config(new_config: dict):
    """
    Update the wandb configuration with the new values.
    """
    wandb.config.update(new_config, allow_val_change=True)


def setup_wandb(params: dict, logdir: str, wandb_project: str, start_new_exp: bool = False,
                run_name: Optional[str] = None) -> str:
    """
    Setup wandb logging and store its parameters in logdir.
    
    Parameters
    ----------
    params : dict
        Parameters of the experiment to log
        
    logdir : str
        Path to directory where logs and checkpoints are saved.
    
    wandb_project : str
        Name of wandb project to log to.
        
    Returns
    -------
    str
        Path to directory where logs and checkpoints are saved.

    """
    wandb_params_path = os.path.join(logdir, 'wandb.yaml')
    if os.path.exists(wandb_params_path) and not start_new_exp:
        wandb_params = load_config(wandb_params_path)
    else:
        wandb_params = {'id': wandb.util.generate_id()}

    run_name = prepend_date_time(run_name) \
        if run_name is not None else None

    wandb.init(
        project=wandb_project,
        config=params,
        resume='allow',
        id=wandb_params['id'],
        name=run_name,
    )

    wandb_params['name'] = wandb.run.name
    wandb_params['project'] = wandb.run.project
    wandb_params['link'] = wandb.run.get_url()

    dump_config(wandb_params_path, wandb_params)
    
    return wandb.run.dir