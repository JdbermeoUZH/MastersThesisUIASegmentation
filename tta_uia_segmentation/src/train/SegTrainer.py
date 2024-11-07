import os
import json
from tqdm import tqdm
from typing import Union, Optional, Literal

import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR

from tta_uia_segmentation.src.models import BaseSeg
from tta_uia_segmentation.src.utils.loss import DiceLoss, dice_score
from tta_uia_segmentation.src.utils.io import save_checkpoint
from tta_uia_segmentation.src.train.utils import LinearWarmupScheduler

class SegTrainer:
    
    def __init__(
        self,
        seg: BaseSeg,
        learning_rate: float,
        device: torch.device,
        bg_suppression_opts: Optional[dict] = None,
        with_bg_supression: bool = False,
        loss_func: torch.nn.Module = DiceLoss(),
        optimizer_type: Literal['adam', 'adamW'] = 'adam',
        is_resumed: bool = False,
        checkpoint_last: str = 'checkpoint_last.pth',
        checkpoint_best: str = 'checkpoint_best.pth',
        logdir: str = 'logs',
        wandb_log: bool = True,
        wandb_dir: str = 'wandb',
        ):
        
        self.device = device
        self.seg = seg
        self.bg_suppression_opts = bg_suppression_opts
        self.with_bg_supression = with_bg_supression if with_bg_supression is not None \
            else bg_suppression_opts['type'] != 'none'
        
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.seg.parameters(),
                lr=learning_rate
            )
        elif optimizer_type == 'adamW':
            self.optimizer = torch.optim.AdamW(
                self.seg.parameters(),
                lr=learning_rate,
                weight_decay=1e-5
            )    
        
        if is_resumed:
            self._load_checkpoint(os.path.join(logdir, checkpoint_last))
        
        else:
            print('Starting training from scratch.')
            self.best_validation_loss = np.inf
        
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
        warmup_steps: Optional[int] = None,
        checkpoint_best: Optional[str] = None,
        checkpoint_last: Optional[str] = None,
        logdir: Optional[str] = None,
        val_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None, 
        wandb_log: Optional[bool] = None,
        wandb_dir: Optional[str] = None,
        ):

        # Define warmup scheduler
        if warmup_steps is not None:
            if 0.0 < warmup_steps < 1.0:
                total_steps = len(train_dataloader) * epochs
                warmup_steps = int(warmup_steps * total_steps)
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps
                )
        else:
            warmup_scheduler = None
        
        wandb_log = wandb_log or self.wandb_log
        device = device or self.device
        logdir = logdir or self.logdir
        checkpoint_best = checkpoint_best or self.checkpoint_best
        checkpoint_last = checkpoint_last or self.checkpoint_last
        wandb_dir = wandb_dir or self.wandb_dir
        
        print('Starting training')

        best_epoch = -1
        
        for epoch in tqdm(range(self.continue_from_epoch, epochs)):

            training_loss = 0
            n_samples_train = 0

            self.seg.train()
            
            print(f'Training for epoch {epoch}')
            for x, y, *_, in tqdm(train_dataloader):
                x = x.to(device).float()
                y = y.to(device).float()

                y_pred, *_ = self.seg(x)

                loss = self.loss_func(y_pred, y)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                # Update learning rate
                if warmup_scheduler is not None: 
                    warmup_scheduler.step()

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
                validation_loss, validation_score_fg, validation_scores = self.evaluate(val_dataloader, device=device)
                self.validation_losses.append(validation_loss.item())
                self.validation_scores.append(validation_score_fg)

                # Checkpoint last state
                self._save_checkpoint(os.path.join(logdir, checkpoint_last), epoch=epoch)
                
                if validation_loss < self.best_validation_loss:
                    self.best_validation_loss = validation_loss

                    # Checkpoint best state
                    self._save_checkpoint(os.path.join(logdir, checkpoint_best), epoch=epoch)

                    best_epoch = epoch
            
            if wandb_log:
                if val_dataloader is not None:
                    validation_metrics_to_log = {
                        'validation_loss': self.validation_losses[-1],
                        'validation_score': self.validation_scores[-1],
                    }
                    
                    if validation_scores.shape[0] > 1:
                        for i, score_per_class in enumerate(validation_scores):
                            label_name = val_dataloader.dataset.get_label_name(i)
                            validation_metrics_to_log[f'validation_score_{label_name}'] = score_per_class.item()
                else :
                    validation_metrics_to_log = {}
                
                wandb.log({
                    'training_loss': training_loss,
                    **validation_metrics_to_log
                }, step=epoch)
                wandb.save(os.path.join(wandb_dir, checkpoint_last), base_path=wandb_dir)
                wandb.save(os.path.join(wandb_dir, checkpoint_best), base_path=wandb_dir)

        print(f'Training finished. Best epoch: {best_epoch}')
                
        # Save the moments and quantiles of the best and last checkpoints
        for checkpoint_path in [self.get_best_checkpoint_path(), self.get_last_checkpoint_path()]:
            self._save_normalized_images_moments_and_quantiles(
                checkpoint_path.replace('.pth', ''), dataloader=train_dataloader, num_epochs=1)
    
    @torch.inference_mode()
    def evaluate(
        self,
        val_dataloader: DataLoader,
        device: Optional[torch.device] = None,
        ):

        device = device or self.device
        validation_loss = 0
        validation_scores = 0
        n_samples_val = 0
        
        self.seg.eval()

        for x, y, *_, in val_dataloader:
            
            x = x.to(device).float()
            y = y.to(device).float()
            
            y_pred, *_ = self.seg(x)
            
            loss = self.loss_func(y_pred, y)
            
            # Get mean dice score per class 
            dices = dice_score(y_pred, y, soft=False, reduction='none', 
                               foreground_only=False, bg_channel=0)
            dices = dices.nanmean(0) # Mean over samples, dice per class

            validation_loss += loss * x.shape[0]
            validation_scores += dices * x.shape[0]
            n_samples_val += x.shape[0]

        validation_loss /= n_samples_val
        validation_scores /= n_samples_val

        validation_score_fg = validation_scores[:, 1:].nanmean()

        return validation_loss, validation_score_fg, validation_scores
        
    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        print(f'Resuming training at epoch {checkpoint["epoch"] + 1}.')
        
        # Load trainer specific attributes
        self.continue_from_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_validation_loss = checkpoint['best_validation_loss']

        # Load model specific attributes
        self.seg.load_checkpoint(checkpoint_path)

        del checkpoint
        
    def _save_checkpoint(self, checkpoint_path: str, epoch: int):
        
        self.seg.save_checkpoint(
            path=checkpoint_path,
            epoch=epoch,
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
    
    def get_validation_losses(self, validate_every: Optional[int] = None):
        return self.validation_losses if validate_every is None else np.repeat(self.validation_losses, validate_every)
    
    def get_validation_scores(self, validate_every: Optional[int] = None):
        return self.validation_scores if validate_every is None else np.repeat(self.validation_losses, validate_every)


       
        