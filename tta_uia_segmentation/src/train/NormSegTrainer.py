import os
import json
from tqdm import tqdm
from typing import Union, Optional

import wandb
import torch
import numpy as np
from tdigest import TDigest
from torch.utils.data import DataLoader

from tta_uia_segmentation.src.models import UNet
from tta_uia_segmentation.src.models.normalization import background_suppression
from tta_uia_segmentation.src.utils.loss import DiceLoss, dice_score
from tta_uia_segmentation.src.utils.io import save_checkpoint


default_quantile_list = list(np.concatenate([
    np.array([0.001, 0.01, 0.025, 0.05]),
    np.arange(.1, 0.9, 0.05),
    np.array([0.95, 0.975, 0.99, 0.999])
]))


class NormSegTrainer:
    """
    Train a segmentation network that is preceded by a shalow image normalization network.
    
    Methods
    -------
    __init__(norm, seg, bg_suppression_opts, learning_rate, device, loss_func,
            is_resumed, checkpoint_last, checkpoint_best, logdir, wandb_log, wandb_dir)
            
        Paramaters
        ----------
        norm : torch.nn.Module
            The normalization network.
            
        seg : Union[UNet, torch.nn.Module]
            The segmentation network.
        
        bg_suppression_opts : dict
            The options or parameters for background suppression.    
        
        
    """
    
    def __init__(
        self,
        norm: torch.nn.Module,
        seg: Union[UNet, torch.nn.Module],
        bg_suppression_opts: dict,
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
        self.norm = norm
        self.seg = seg
        self.bg_suppression_opts = bg_suppression_opts
        self.with_bg_supression = bg_suppression_opts['type'] != 'none'
        self.optimizer = torch.optim.Adam(
            list(self.norm.parameters()) + list(self.seg.parameters()),
            lr=learning_rate
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
        with_bg_supression: Optional[bool] = None,
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
        with_bg_supression = with_bg_supression or self.with_bg_supression
        
        print('Starting training')
        
        for epoch in tqdm(range(self.continue_from_epoch, epochs)):

            training_loss = 0
            n_samples_train = 0

            self.norm.train()
            self.seg.train()
            
            print(f'Training for epoch {epoch}')
            for x, y, _, _, bg_mask in tqdm(train_dataloader):
                x = x.to(device).float()
                y = y.to(device)
                bg_mask = bg_mask.to(device)
                
                x_norm = self.norm(x)
                if with_bg_supression:
                    x_norm = background_suppression(x_norm, bg_mask, self.bg_suppression_opts)
                y_pred, _ = self.seg(x_norm)

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
                self.validation_scores.append(validation_score.nanmean().item())

                # Checkpoint last state
                self._save_checkpoint(os.path.join(logdir, checkpoint_last))
                
                if validation_loss < self.best_validation_loss:
                    self.best_validation_loss = validation_loss

                    # Checkpoint best state
                    self._save_checkpoint(os.path.join(logdir, checkpoint_best))
            
            if wandb_log:
                
                if val_dataloader is not None:
                    validation_metrics_to_log = {
                        'validation_loss': validation_loss.item(),
                        'validation_score': validation_score.nanmean().item(),
                    }
                    
                    if validation_score.shape[0] > 1:
                        for i, score_per_class in enumerate(validation_score):
                            label_idx = i + 1
                            label_name = val_dataloader.dataset.get_label_name(label_idx)
                            validation_metrics_to_log[f'validation_score_{label_name}'] = score_per_class.item()
                else :
                    validation_metrics_to_log = {}
                
                wandb.log({
                    'training_loss': training_loss,
                    **validation_metrics_to_log
                }, step=epoch)
                wandb.save(os.path.join(wandb_dir, checkpoint_last), base_path=wandb_dir)
                wandb.save(os.path.join(wandb_dir, checkpoint_best), base_path=wandb_dir)
                
        # Save the moments and quantiles of the best and last checkpoints
        for checkpoint_path in [self.get_best_checkpoint_path(), self.get_last_checkpoint_path()]:
            self._save_normalized_images_moments_and_quantiles(
                checkpoint_path.replace('.pth', ''), dataloader=train_dataloader, num_epochs=1)
    
    @torch.inference_mode()
    def evaluate(
        self,
        val_dataloader: DataLoader,
        device: Optional[torch.device] = None,
        with_bg_supression: Optional[bool] = None,
        ):

        device = device or self.device
        with_bg_supression = with_bg_supression or self.with_bg_supression
        validation_loss = 0
        validation_score = 0
        n_samples_val = 0
        
        self.norm.eval()
        self.seg.eval()

        for x, y, _, _, bg_mask in val_dataloader:
            x = x.to(device).float()
            y = y.to(device)
            bg_mask = bg_mask.to(device)

            x_norm = self.norm(x)
            if with_bg_supression:
                x_norm = background_suppression(x_norm, bg_mask, self.bg_suppression_opts)
            y_pred, _ = self.seg(x_norm)
            
            loss = self.loss_func(y_pred, y)
            
            _, dice_fg = dice_score(y_pred, y, soft=False, reduction='none', epsilon=1e-5)
            dice_fg = dice_fg.nanmean(0)

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
        self.norm.load_state_dict(checkpoint['norm_state_dict'])
        self.seg.load_state_dict(checkpoint['seg_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_validation_loss = checkpoint['best_validation_loss']

        del checkpoint
        
    def _save_checkpoint(self, checkpoint_path: str):
        
        save_checkpoint(
            path=checkpoint_path,
            epoch=-1,
            norm_state_dict=self.norm.state_dict(),
            seg_state_dict=self.seg.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            best_validation_loss=self.best_validation_loss,
        )
        
    def _save_normalized_images_moments_and_quantiles(self, fp_prefix: str, *args, **kwargs):
        moments_and_quantiles, digest = self._get_normalized_images_moments_and_quantiles(
            *args, **kwargs)
        
        # Save moments and quantiles
        filepath = f'{fp_prefix}_moments_and_quantiles.json'
        json.dump(moments_and_quantiles, open(filepath, 'w'), indent=4)
        
        # Save digest object
        filepath = f'{fp_prefix}_serialized_TDigest_to_calculate_quantiles.json'
        json.dump(digest.to_dict(), open(filepath, 'w'), indent=4)
    
    @torch.inference_mode()
    def _get_normalized_images_moments_and_quantiles(
        self,
        dataloader: DataLoader, 
        num_epochs: int = 1,
        quantiles_to_report: list = default_quantile_list,
        frac_dataset: float = 1.0,
        ) -> tuple[dict, TDigest]:
        
        self.norm.eval()
        
        sum_px = 0
        sum_sq_px = 0
        n_px = 0
        digest = TDigest()

        num_batches = len(dataloader)
        max_batches = int(frac_dataset * num_batches)
        
        print('Estimating moments and quantiles of normalized images.')
        print(f'Using {num_epochs} epochs. Processing {max_batches} batches per epoch.')
        
        for _ in range(num_epochs):
            for i, (x, *_) in tqdm(enumerate(dataloader), total=max_batches):
                if i >= max_batches:
                    break
                x = x.to(self.device).float()
                x_norm = self.norm(x)
                
                n_px += x.numel()
                sum_px += x.sum().item()
                sum_sq_px += (x ** 2).sum().item()
                digest.batch_update(x_norm.flatten().cpu().numpy())
                
        mean = (sum_px / n_px).item()
        std = np.sqrt(sum_sq_px / n_px - mean ** 2).item()
        quantiles = {q: digest.percentile(q * 100) for q in quantiles_to_report}
        
        moments_and_quantiles = {
            'mean': mean,
            'std': std,
            'quantiles': quantiles
        }
        
        return moments_and_quantiles, digest
    
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


       
        