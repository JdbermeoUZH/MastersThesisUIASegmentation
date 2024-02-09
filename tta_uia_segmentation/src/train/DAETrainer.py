import os
from tqdm import tqdm
from typing import Union, Optional

import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader

from tta_uia_segmentation.src.models import UNet
from tta_uia_segmentation.src.utils.loss import DiceLoss, dice_score
from tta_uia_segmentation.src.utils.io import save_checkpoint


class DAETrainer:
    """
    Trainer for the DAE.
    
    It executes the training and evaluation loops, 
    it also tracks the state (losses, scores, checkpoints)
    
    Methods
    -------
    train(train_dataloader, epochs, validate_every, checkpoint_best, checkpoint_last, logdir, val_dataloader, device, wandb_log)
        Executes the training loop.
        
    evaluate(val_dataloader, device)
        Executes the evaluation loop. Returns the loss and score over the entire val dataset.
        
    _load_checkpoint(checkpoint_path)
        Loads a checkpoint to resume trainin. Usually called at initialization of the class
        
    _start_from_scratch(checkpoint_path)
        Stores a checkpoint with the initial state.

    """
    def __init__(
        self,
        dae: Union[UNet, torch.nn.Module],
        learning_rate: float,
        device: torch.device,
        loss_func: torch.nn.Module = DiceLoss(),
        is_resumed: bool = False,
        checkpoint_last: str = 'checkpoint_last.pth',
        checkpoint_best: str = 'checkpoint_best.pth',
        logdir: str = 'logs',
        wandb_log: bool = True,
        wandb_dir: str = 'wandb',
        ):

        self.device = device
        self.dae = dae
        self.optimizer = torch.optim.Adam(
            self.dae.parameters(),
            lr=learning_rate
        )    
        if is_resumed:
            self._load_checkpoint(os.path.join(logdir, checkpoint_last))
        else:
            print('Starting training from scratch.')
            self.best_validation_loss = np.inf
            self._save_checkpoint(os.path.join(logdir, checkpoint_last))
        
        self.loss_func = loss_func
        
        self.training_losses = []
        self.validation_losses = []
        self.validation_scores = []

        self.best_validation_loss = np.inf
        self.continue_from_epoch = 0
        
        self.logdir = logdir
        self.checkpoint_last = checkpoint_last
        self.checkpoint_best = checkpoint_best
        
        self.wandb_log = wandb_log
        self.wandb_dir = wandb_dir
        
    def train(
        self,
        train_dataloader: DataLoader,
        epochs: int,
        validate_every: int,
        checkpoint_best: Optional[str] = None,
        checkpoint_last: Optional[str] = None,
        logdir: Optional[str] = None,
        val_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None, 
        wandb_log: Optional[bool] = None,
        wandb_dir: Optional[str] = None,
        ):
        
        wandb_log = wandb_log or self.wandb_log
        device = device or self.device
        logdir = logdir or self.logdir
        checkpoint_best = checkpoint_best or self.checkpoint_best
        checkpoint_last = checkpoint_last or self.checkpoint_last
        wandb_dir = wandb_dir or self.wandb_dir
        
        print('Starting training')
        
        for epoch in tqdm(range(self.continue_from_epoch, epochs)):

            training_loss = 0
            n_samples_train = 0

            self.dae.train()

            print(f'Training for epoch {epoch}')
            for _, y, x, _, _ in train_dataloader:
                x = x.to(device).float()
                y = y.to(device)

                y_pred, _ = self.dae(x)

                loss = self.loss_func(y_pred, y)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                with torch.no_grad():
                    training_loss += loss.detach() * x.shape[0]
                    n_samples_train += x.shape[0]
            
            training_loss /= n_samples_train
            self.training_losses.append(training_loss.item())

            if (epoch + 1) % validate_every != 0 and epoch != epochs - 1:
                continue

            if val_dataloader is not None:
                # Evaluation
                print(f'Validating for epoch {epoch}')
                validation_loss, validation_score = self.evaluate(val_dataloader, device=device)
                self.validation_losses.append(validation_loss.item())
                self.validation_scores.append(validation_score.item())

                # Checkpoint last state
                self._save_checkpoint(os.path.join(logdir, checkpoint_last))
                
                if validation_loss < self.best_validation_loss:
                    self.best_validation_loss = validation_loss

                    # Checkpoint best state
                    self._save_checkpoint(os.path.join(logdir, checkpoint_best))
            
            if wandb_log:
                wandb.log({
                    'training_loss': training_loss,
                    'validation_loss': validation_loss if val_dataloader is not None else None,
                    'validation_score': validation_score if val_dataloader is not None else None,
                }, step=epoch)
                wandb.save(os.path.join(wandb_dir, checkpoint_last), base_path=wandb_dir)
                wandb.save(os.path.join(wandb_dir, checkpoint_best), base_path=wandb_dir)
    
    def evaluate(
        self,
        val_dataloader: DataLoader,
        device: Optional[torch.device] = None,
        ):

        device = device or self.device
        validation_loss = 0
        validation_score = 0
        n_samples_val = 0
        
        self.dae.eval()

        with torch.no_grad():
            for _, y, x, _, _ in val_dataloader:
                x = x.to(device).float()
                y = y.to(device)

                y_pred, _ = self.dae(x)

                loss = self.loss_func(y_pred, y)
                dice, dice_fg = dice_score(y_pred, y, soft=False, reduction='mean')

                validation_loss += loss * x.shape[0]
                validation_score += dice_fg * x.shape[0]
                n_samples_val += x.shape[0]

            validation_loss /= n_samples_val
            validation_score /= n_samples_val

        return validation_loss, validation_score
    
    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        print(f'Resuming training at epoch {checkpoint["epoch"] + 1}.')
        
        self.continue_from_epoch = checkpoint['epoch'] + 1
        self.dae.load_state_dict(checkpoint['dae_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_validation_loss = checkpoint['best_validation_loss']

        del checkpoint
        
    def _save_checkpoint(self, checkpoint_path: str):       
        save_checkpoint(
            path=checkpoint_path,
            epoch=-1,
            dae_state_dict=self.dae.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            best_validation_loss=self.best_validation_loss,
        )
    
    def get_last_checkpoint_name(self):
        return self.checkpoint_last
    
    def get_best_checkpoint_name(self):
        return self.checkpoint_best
    
    def get_last_checkpoint_path(self):
        return os.path.join(self.logdir, self.checkpoint_last)

    def get_best_checkpoint_path(self):
        return os.path.join(self.logdir, self.checkpoint_best)  
    
    def get_loss_function (self):
        return self.loss_func    
    
    def get_training_losses(self):
        return self.training_losses
    
    def get_validation_losses(self):
        return self.validation_losses
    
    def get_validation_scores(self):
        return self.validation_scores


