import os
import copy
from typing import Union, Optional, Any
from tqdm import tqdm

import wandb
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset, ConcatDataset, DataLoader

from tta_uia_segmentation.src.tta.TTADAE import TTADAE
from tta_uia_segmentation.src.models import ConditionalGaussianDiffusion
from tta_uia_segmentation.src.models.normalization import background_suppression
from tta_uia_segmentation.src.utils.io import save_checkpoint, write_to_csv
from tta_uia_segmentation.src.utils.loss import dice_score, DiceLoss
from tta_uia_segmentation.src.utils.visualization import export_images
from tta_uia_segmentation.src.utils.utils import get_seed
from tta_uia_segmentation.src.dataset import DatasetInMemory, utils as du


class TTADAEandDDPM(TTADAE):
    
    def __init__(
        self,
        ddpm: ConditionalGaussianDiffusion,
        num_imgs_diffusion_tta: int = 64,
        dddpm_loss_alpha: float = 0.5,
        min_t_diffusion_tta: int = 250,
        max_t_diffusion_tta: int = 1000,
        sampling_timesteps: Optional[int] = None,
        **kwargs
        ) -> None:
        
        super().__init__(**kwargs)
        self.ddpm = ddpm
        
        assert (0 <= dddpm_loss_alpha <= 1), 'dddpm_loss_alpha must be between 0 and 1'
        self.dddpm_loss_alpha = dddpm_loss_alpha
        self.num_imgs_diffusion_tta = num_imgs_diffusion_tta
        self.min_t_diffusion_tta = min_t_diffusion_tta
        self.max_t_diffusion_tta = max_t_diffusion_tta
        self.sampling_timesteps = sampling_timesteps
        self.ddpm.set_sampling_timesteps(sampling_timesteps)
        
        # Set segmentation, DAE, and DDPM models in eval mode
        self.ddpm.eval()
        self.ddpm.requires_grad_(False)
        
    def tta(
        self,
        volume_dataset: DatasetInMemory,
        dataset_name: str,
        n_classes: int, 
        index: int,
        rescale_factor_dae: tuple[int],
        alpha: float,
        beta: float,
        bg_suppression_opts: dict,
        bg_suppression_opts_tta: dict,
        num_steps: int,
        batch_size: int,
        num_workers: int,
        calculate_dice_every: int,
        update_dae_output_every: int,
        accumulate_over_volume: bool,
        dataset_repetition: int,
        const_aug_per_volume: bool,
        save_checkpoints: bool,
        device: str,
        logdir: Optional[str] = None,        
    ):       
        """_summary_

        Arguments:
        ----------
        volume_dataset : DatasetInMemory
            Dataset containing slices of a single volume on which to perform TTA.
        """
        self.seg.requires_grad_(False)

        if rescale_factor_dae is not None:
            assert (batch_size * rescale_factor_dae[0]) % 1 == 0
            label_batch_size = int(batch_size * rescale_factor_dae[0])
        else:
            label_batch_size = batch_size

        dae_dataloader = DataLoader(
            volume_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        volume_dataloader = DataLoader(
            ConcatDataset([volume_dataset] * dataset_repetition),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
        )
        
        for step in range(num_steps):
            
            self.norm.eval()
            volume_dataset.dataset.set_augmentation(False)
    
            # Test performance during adaptation.
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

            # Update Pseudo label, with DAE or Atlas, depending on which has a better agreement
            if step % update_dae_output_every == 0:
                
                label_dataloader = self.generate_pseudo_labels(
                    dae_dataloader=dae_dataloader,
                    label_batch_size=label_batch_size,
                    bg_suppression_opts_tta=bg_suppression_opts_tta,
                    rescale_factor=rescale_factor_dae,
                    device=device,
                    num_workers=num_workers,
                    dataset_repetition=dataset_repetition,
                    alpha=alpha,
                    beta=beta
                )

            tta_loss = 0
            ddpm_loss = 0
            dae_loss = 0
            n_samples = 0
            n_samples_diffusion = 0

            self.norm.train()
            volume_dataset.dataset.set_augmentation(True)

            if accumulate_over_volume:
                self.optimizer.zero_grad()

            if const_aug_per_volume:
                volume_dataset.dataset.set_seed(get_seed())
                
            # Sample batches of images on which to use the noise estimation task
            num_batches_sample = self.num_imgs_diffusion_tta // batch_size
            b_i_for_diffusion_loss = np.random.choice(
                range(len(volume_dataloader)), num_batches_sample, replace=False)

            # Adapting to the target distribution.
            for b_i, ((x,_,_,_, bg_mask), (y,)) in enumerate(zip(volume_dataloader, label_dataloader)):

                if b_i not in b_i_for_diffusion_loss:
                    continue
                
                if not accumulate_over_volume:
                    self.optimizer.zero_grad()

                x = x.to(device).float()
                y = y.to(device)
                bg_mask = bg_mask.to(device)
                x_norm = self.norm(x)
                
                # Calculate the noise estimation loss
                if b_i in b_i_for_diffusion_loss:
                    ddpm_loss = self.get_ddpm_loss(x_norm, y.detach().clone()) 
                    n_samples_diffusion += x.shape[0]
                    
                # x_norm = background_suppression(x_norm, bg_mask, bg_suppression_opts_tta)

                # mask, _ = self.seg(x_norm)
                
                # if rescale_factor_dae is not None:
                #     mask = mask.permute(1, 0, 2, 3).unsqueeze(0)
                #     mask = F.interpolate(mask, scale_factor=rescale_factor_dae, mode='trilinear')
                #     mask = mask.squeeze(0).permute(1, 0, 2, 3)

                # loss = self.loss_func(mask, y)

                if accumulate_over_volume:
                    # loss = loss / len(volume_dataloader)
                    ddpm_loss = ddpm_loss / len(b_i_for_diffusion_loss)
                    
                #loss = (1 - self.dddpm_loss_alpha) * loss +\
                #    self.dddpm_loss_alpha * noise_estimation_loss

                total_loss = ddpm_loss                
                
                total_loss.backward()

                if not accumulate_over_volume:
                    self.optimizer.step()                      

                with torch.no_grad():
                    tta_loss += total_loss.detach() * x.shape[0]
                    ddpm_loss += ddpm_loss.detach() * x.shape[0]
                    # dae_loss += loss.detach() * x.shape[0]
                    n_samples += x.shape[0]

            if accumulate_over_volume:
                self.optimizer.step()

            self.tta_losses.append((tta_loss / n_samples).item())

            if self.wandb_log:
                wandb.log(
                    {
                        f'noise_estimation_loss_img_{index}': (ddpm_loss / n_samples_diffusion).item(),
                        #f'dae_loss_img_{vol_idx}': (loss / n_samples).item(),
                        f'total_loss_img_{index}': (tta_loss / n_samples).item(),
                    },
                    step=step)  

        if save_checkpoints:
            os.makedirs(os.path.join(logdir, 'checkpoints'), exist_ok=True)
            save_checkpoint(
                path=os.path.join(logdir, 'checkpoints',
                                f'checkpoint_tta_{dataset_name}_{index:02d}.pth'),
                norm_state_dict=self.norm_dict['best_score'],
                seg_state_dict=self.seg.state_dict(),
            )

        os.makedirs(os.path.join(logdir, 'metrics'), exist_ok=True)

        os.makedirs(os.path.join(logdir, 'tta_score'), exist_ok=True)
        
        write_to_csv(
            os.path.join(logdir, 'tta_score', f'{dataset_name}_{index:03d}.csv'),
            np.array([self.test_scores]).T,
            header=['tta_score'],
            mode='w',
        )

        return self.norm, self.norm_dict, self.metrics_best
    
    def get_ddpm_loss(
        self,
        img,
        seg,
        minibatch_size: int = 1,
        ) -> torch.Tensor:
        
        # Normalize the input image between 0 and 1
        img = du.normalize_min_max(img)
        
        # Upsample the segmentation mask to the same size as the input image
        rescale_factor = np.array(img.shape) / np.array(seg.shape)
        rescale_factor = tuple(rescale_factor[[0, 2, 3]])
        
        seg = seg.permute(1, 0, 2, 3).unsqueeze(0)
        seg = F.interpolate(seg, scale_factor=rescale_factor, mode='trilinear')
        seg = seg.squeeze(0).permute(1, 0, 2, 3)
        
        # Map seg to a single channel and normalize between 0 and 1
        n_classes = seg.shape[1]
        seg = du.onehot_to_class(seg)
        seg = seg.float() / (n_classes - 1)
        seg = (seg * 1.) / (n_classes - 1)
        
        # The DDPM is memory intensive, compute the loss in smaller batches
        ddpm_loss = 0
        for i in range(0, img.shape[0], minibatch_size):
            img_batch = img[i:i+minibatch_size]
            seg_batch = seg[i:i+minibatch_size]
            ddpm_loss += self.ddpm(img_batch, seg_batch)
        
        return ddpm_loss  

    