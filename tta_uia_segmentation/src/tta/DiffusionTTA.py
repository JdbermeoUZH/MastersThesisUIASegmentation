import os
from typing import Optional
from tqdm import tqdm

import wandb
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from tta_uia_segmentation.src.models import ConditionalGaussianDiffusion
from tta_uia_segmentation.src.models.normalization import background_suppression
from tta_uia_segmentation.src.utils.io import save_checkpoint, write_to_csv
from tta_uia_segmentation.src.utils.utils import get_seed
from tta_uia_segmentation.src.dataset import DatasetInMemory, utils as du


class DiffusionTTA:
    
    def __init__(
        self,
        norm: torch.nn.Module,
        seg: torch.nn.Module,
        ddpm: ConditionalGaussianDiffusion,
        learning_rate: float,
        learning_rate_norm: Optional[float] = None, 
        learning_rate_seg: Optional[float] = None,
        learning_rate_ddpm: Optional[float] = None,
        min_t_diffusion_tta: int = 0,
        max_t_diffusion_tta: int = 999,
        frac_vol_diffusion_tta: float = 1.0,
        min_max_intensity_imgs: tuple[float, float] = (0, 1), 
        fit_norm_params: bool = True,
        fit_seg_params: bool = True, 
        fit_ddpm_params: bool = True,    
        **kwargs
        ) -> None:
        
        super().__init__(**kwargs)
        self.norm = norm
        self.seg = seg
        self.ddpm = ddpm
        
        self.learning_rate = learning_rate
        self.learning_rate_norm = learning_rate_norm if learning_rate_norm is not None else learning_rate
        self.learning_rate_seg = learning_rate_seg if learning_rate_seg is not None else learning_rate
        self.learning_rate_ddpm = learning_rate_ddpm if learning_rate_ddpm is not None else learning_rate
        
        self.optimizer = torch.optim.Adam(
            [
                {'norm': self.norm.parameters(), 'lr': self.learning_rate_norm},
                {'seg': self.seg.parameters(), 'lr': self.learning_rate_seg},
                {'ddpm': self.ddpm.parameters(), 'lr': self.learning_rate_ddpm}
            ]
        )
        
        # Setting up metrics for model selection.
        self.tta_losses = []
        self.test_scores = []

        # DDPM loss parameters        
        self.min_t_diffusion_tta = min_t_diffusion_tta
        self.max_t_diffusion_tta = max_t_diffusion_tta
        
        self.min_intensity_imgs = min_max_intensity_imgs[0]
        self.max_intensity_imgs = min_max_intensity_imgs[1]
        
        self.frac_vol_diffusion_tta = frac_vol_diffusion_tta
            
        # Set parameters of the networks to trainable or not
        self.fit_norm_params = fit_norm_params
        self.fit_seg_params = fit_seg_params    
        self.fit_ddpm_params = fit_ddpm_params
        
        self.norm.train() if fit_norm_params else self.norm.eval()
        self.norm.requires_grad_(fit_norm_params)
        
        self.seg.train() if fit_seg_params else self.seg.eval()
        self.seg.requires_grad_(fit_seg_params)
        
        self.ddpm.train() if fit_ddpm_params else self.ddpm.eval() 
        self.ddpm.requires_grad_(fit_ddpm_params)
        
        # Setup the custom metrics and steps wandb
        if self.wandb_log:
            self._define_custom_wandb_metrics() 
    
    def tta(
        self,
        volume_dataset: DatasetInMemory,
        dataset_name: str,
        n_classes: int, 
        index: int,
        bg_suppression_opts: dict,
        bg_suppression_opts_tta: dict,
        num_steps: int,
        batch_size: int,
        minibatch_size_ddpm: int,
        num_workers: int,
        calculate_dice_every: int,
        accumulate_over_volume: bool,
        dataset_repetition: int,
        const_aug_per_volume: bool,
        device: str,
        logdir: Optional[str] = None,     
    ):       
        """_summary_

        Arguments:
        ----------
        volume_dataset : DatasetInMemory
            Dataset containing slices of a single volume on which to perform TTA.
        """
        self.tta_losses = []
        self.test_scores = []
        
        # Define the sampler object for the volume dataset
        repeated_volume_dataset = ConcatDataset([volume_dataset] * dataset_repetition)
        sampler = RandomSampler(repeated_volume_dataset, replacement=False, 
                                num_samples=int(self.frac_vol_diffusion_tta * len(repeated_volume_dataset)))
        
        for step in tqdm(range(num_steps)):
            # Load the sample of cuts on which to perform TTA
            volume_dataloader = DataLoader(
                repeated_volume_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
                sampler=sampler
            )
            
            self.norm.eval()
            self.seg.eval()
            volume_dataset.dataset.set_augmentation(False)
                
            # Measure segmentation performance during adaptation
            # :===========================================: 
            if step % calculate_dice_every == 0 and calculate_dice_every != -1:

                _, dices_fg = self.test_volume(
                    volume_dataset=volume_dataset,
                    dataset_name=dataset_name,
                    logdir=logdir,
                    device=device,
                    batch_size=batch_size,
                    n_classes=n_classes,
                    num_workers=num_workers,
                    index=index,
                    iteration=step,
                    bg_suppression_opts=bg_suppression_opts,
                )
                self.test_scores.append(dices_fg.mean().item())

            # Fit the parameters of the networks
            ddpm_loss = torch.tensor(0).float().to(device)
            n_samples = 0

            self.norm.train() if self.fit_norm_params else self.norm.eval()
            self.seg.train() if self.fit_seg_params else self.seg.eval()
            
            volume_dataset.dataset.set_augmentation(True)

            if accumulate_over_volume:
                self.optimizer.zero_grad()

            if const_aug_per_volume:
                volume_dataset.dataset.set_seed(get_seed())
                            
            # Adapting based on image likelihood
            # :===========================================:
            for x, _,_,_, bg_mask in volume_dataloader:

                if not accumulate_over_volume:
                    self.optimizer.zero_grad()

                x = x.to(device).float()                
                bg_mask = bg_mask.to(device)
                
                # Get the predicted segmentation
                x_norm = self.norm(x)
                x_norm_bg_supp = background_suppression(x_norm, bg_mask, bg_suppression_opts_tta)
                x_cond, _  = self.seg(x_norm_bg_supp)
                
                # Calculate gradients of the batch
                ddpm_loss = self.calculate_ddpm_gradients(
                    x,
                    x_cond,
                    minibatch_size=minibatch_size_ddpm
                )
                                                    
                if not accumulate_over_volume:
                    self.optimizer.step()                      

                with torch.no_grad():
                    ddpm_loss += ddpm_loss.detach() * x.shape[0]
                    n_samples += x.shape[0]
                    
            if accumulate_over_volume:
                self.optimizer.step()

            ddpm_loss = (ddpm_loss / n_samples).item()
            
            self.tta_losses.append(ddpm_loss)

            if self.wandb_log:
                wandb.log({
                    f'ddpm_loss/img_{index}': ddpm_loss, 
                    'tta_step': step
                    }
                )  

        os.makedirs(os.path.join(logdir, 'metrics'), exist_ok=True)

        os.makedirs(os.path.join(logdir, 'tta_score'), exist_ok=True)
        
        write_to_csv(
            os.path.join(logdir, 'tta_score', f'{dataset_name}_{index:03d}.csv'),
            np.array([self.test_scores]).T,
            header=['tta_score'],
            mode='w',
        )
        
        dice_scores = {i * calculate_dice_every: score for i, score in enumerate(self.test_scores)}

        return self.norm, self.norm_dict, self.metrics_best, dice_scores
    
    def calculate_ddpm_gradients(
        self,
        img,
        seg,
        minibatch_size: int = 2
        ) -> torch.Tensor:
        
        # Normalize the input image between 0 and 1, (required by the DDPM)
        img = du.normalize_min_max(
            img,
            min=self.min_intensity_imgs, 
            max=self.max_intensity_imgs
            )
        
        if img.max() > 1 or img.min() < 0:
            print(f'WARNING: img.max()={img.max()}, img.min()={img.min()}')
                
        # Map seg to a single channel and normalize between 0 and 1
        n_classes = seg.shape[1]
        seg = du.onehot_to_class(seg)
        seg = seg.float() / (n_classes - 1)
        
        # The DDPM is memory intensive, accumulate gradients over minibatches
        ddpm_loss_value = 0
        for i in range(0, img.shape[0], minibatch_size):
            img_batch = img[i:i+minibatch_size]
            seg_batch = seg[i:i+minibatch_size]
            ddpm_loss = self.ddpm(img_batch, seg_batch)
            
            # Do backward retaining the graph except for the last step
            if i + minibatch_size < img.shape[0]:
                ddpm_loss.backward(retain_graph=True)
            else:
                ddpm_loss.backward()
            
            ddpm_loss_value += ddpm_loss.detach()
                        
        return ddpm_loss_value
    
    def _define_custom_wandb_metrics(self, ):
        wandb.define_metric(f'tta_step')
        wandb.define_metric(f'ddpm_loss/*', step_metric=f'tta_step')
        wandb.define_metric(f'dice_score_fg/*', step_metric=f'tta_step')    
