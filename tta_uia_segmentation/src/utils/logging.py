import os

import wandb

from tta_uia_segmentation.src.utils.io import load_config, dump_config


def setup_wandb(params: dict, logdir: str, wandb_project: str, start_new_exp: bool = False) -> str:
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

    wandb.init(
        project=wandb_project,
        config=params,
        resume='allow',
        id=wandb_params['id']
    )

    wandb_params['name'] = wandb.run.name
    wandb_params['project'] = wandb.run.project
    wandb_params['link'] = f'https://wandb.ai/{wandb.run.path}'

    dump_config(wandb_params_path, wandb_params)
    
    return wandb.run.dir