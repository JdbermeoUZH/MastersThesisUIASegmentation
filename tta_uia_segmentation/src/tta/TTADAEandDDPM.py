import os
from typing import Optional
from tqdm import tqdm

import wandb
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from tta_uia_segmentation.src.tta.TTADAE import TTADAE
from tta_uia_segmentation.src.models import ConditionalGaussianDiffusion
from tta_uia_segmentation.src.models.normalization import background_suppression
from tta_uia_segmentation.src.utils.io import save_checkpoint, write_to_csv
from tta_uia_segmentation.src.utils.utils import get_seed
from tta_uia_segmentation.src.dataset import DatasetInMemory, utils as du


class TTADAEandDDPM(TTADAE):
    
    def __init__(
        self,
        ddpm: ConditionalGaussianDiffusion,
        dae_loss_alpha: float = 0.5,
        ddpm_loss_beta: float = 0.5,
        minibatch_size_ddpm: int = 2,
        frac_vol_diffusion_tta: float = 0.25,
        min_t_diffusion_tta: int = 250,
        max_t_diffusion_tta: int = 1000,
        sampling_timesteps: Optional[int] = None,
        min_max_int_norm_imgs: tuple[float, float] = (0, 1),
        use_x_norm_for_ddpm_loss: bool = True,
        use_y_pred_for_ddpm_loss: bool = False,
        use_x_cond_gt: bool = False,    # Of course use only for debugging
        use_ddpm_after_step: Optional[int] = None,
        use_ddpm_after_dice: Optional[float] = None,
        warmup_steps_for_ddpm_loss: Optional[int] = None,
        use_ddpm_sample_guidance: bool = False,
        **kwargs
        ) -> None:
        
        super().__init__(**kwargs)
        self.ddpm = ddpm
        
        self.dae_loss_alpha = dae_loss_alpha
        self.ddpm_loss_beta = ddpm_loss_beta
        
        self.minibatch_size_ddpm = minibatch_size_ddpm
        self.frac_vol_diffusion_tta = frac_vol_diffusion_tta
        
        self.min_t_diffusion_tta = min_t_diffusion_tta
        self.max_t_diffusion_tta = max_t_diffusion_tta
        
        self.min_int_norm_imgs = min_max_int_norm_imgs[0]
        self.max_int_norm_imgs = min_max_int_norm_imgs[1]
        
        self.use_y_pred_for_ddpm_loss = use_y_pred_for_ddpm_loss   
        self.use_x_norm_for_ddpm_loss = use_x_norm_for_ddpm_loss 
        
        self.sampling_timesteps = sampling_timesteps
        self.ddpm.set_sampling_timesteps(sampling_timesteps)
        
        # Set segmentation, DAE, and DDPM models in eval mode
        self.ddpm.eval()
        self.ddpm.requires_grad_(False)
        
        # Attributes used only for debugging
        self.use_x_cond_gt = use_x_cond_gt
        
        # Setup the custom metrics and steps wandb
        if self.wandb_log:
            self._define_custom_wandb_metrics() 
            
        # Set the step at which to use the DDPM
        assert use_ddpm_after_dice is None or use_ddpm_after_step is None, \
            'Only one of use_ddpm_after_dice or use_ddpm_after_step can be set'

        self.use_ddpm_after_step = use_ddpm_after_step
        self.use_ddpm_after_dice = use_ddpm_after_dice
        self.warmup_steps_for_ddpm_loss = warmup_steps_for_ddpm_loss
        
        # If a flag is not set, use the DDPM at every step
        self.use_ddpm_loss = use_ddpm_after_dice is None and use_ddpm_after_step is None
        
        # DDPM sample guidance parameters
        self.use_ddpm_sample_guidance = use_ddpm_sample_guidance
        self.atlas_sampled_vol = self._sample_atlas_vol() \
            if use_ddpm_sample_guidance else None
            
    
    def tta(
        self,
        volume_dataset: DatasetInMemory,
        dataset_name: str,
        n_classes: int, 
        index: int,
        rescale_factor_dae: tuple[int],
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
        running_min_max_momentum: float = 0.8,
    ):       
        """_summary_

        Arguments:
        ----------
        volume_dataset : DatasetInMemory
            Dataset containing slices of a single volume on which to perform TTA.
        """
        
        self.tta_losses = []
        self.test_scores = []
            
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
        
        running_max = self.max_int_norm_imgs
        running_min = self.min_int_norm_imgs
        m = running_min_max_momentum
        
        # Initialize warmup factor for the DDPM loss term
        if self.warmup_steps_for_ddpm_loss is not None:
            warmup_steps_for_ddpm_loss = list(np.linspace(1, 1/self.warmup_steps_for_ddpm_loss,
                                                    self.warmup_steps_for_ddpm_loss))
        else:
            warmup_steps_for_ddpm_loss = []     
               
        for step in tqdm(range(num_steps)):
            
            self.norm.eval()
            volume_dataset.dataset.set_augmentation(False)
            
            step_min, step_max = np.inf, -np.inf
            
            if self.use_ddpm_after_step is not None and not self.use_ddpm_loss:      
                self.use_ddpm_loss = step >= self.use_ddpm_after_step
                if self.use_ddpm_loss:
                    print('---------Start using DDPM loss ---------')
    
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
                # Only update the pseudo label if it has not been calculated yet or
                #  if the beta is less than 1.0

                if step == 0 or self.beta <= 1.0:
                    dice_dae, dice_atlas, label_dataloader = self.generate_pseudo_labels(
                        dae_dataloader=dae_dataloader,
                        label_batch_size=label_batch_size,
                        bg_suppression_opts_tta=bg_suppression_opts_tta,
                        rescale_factor=rescale_factor_dae,
                        device=device,
                        num_workers=num_workers,
                        dataset_repetition=dataset_repetition
                    )
                    
            if self.use_ddpm_after_dice is not None and not self.use_ddpm_loss:
                self.use_ddpm_loss = dice_dae >= self.use_ddpm_after_dice
                if self.use_ddpm_loss:
                    print('---------Start using DDPM loss ---------')

            tta_loss = 0
            step_dae_loss = 0
            step_ddpm_loss = 0
            n_samples = 0
            n_samples_diffusion = 0

            self.norm.train()
            volume_dataset.dataset.set_augmentation(True)

            if accumulate_over_volume:
                self.optimizer.zero_grad()

            if const_aug_per_volume:
                volume_dataset.dataset.set_seed(get_seed())
                
            # Sample batches of images on which to use the noise estimation task
            num_batches_sample = int(self.frac_vol_diffusion_tta * len(volume_dataloader))
            b_i_for_diffusion_loss = np.random.choice(
                range(len(volume_dataloader)), num_batches_sample, replace=False)
            
            # Reweigh factor for the ddpm loss to take into account how many 
            #  times less it is used than the dae loss or if averaged over entire volume
            ddpm_reweigh_factor = (1 / len(b_i_for_diffusion_loss)) * \
                (1 if accumulate_over_volume else len(volume_dataloader))  

            warmup_factor = warmup_steps_for_ddpm_loss.pop() \
                if len(warmup_steps_for_ddpm_loss) > 0 and self.use_ddpm_loss else 1
            
            assert 1 >= warmup_factor >= 0, 'Warmup factor must be between 0 and 1'
                        
            ddpm_reweigh_factor = warmup_factor * ddpm_reweigh_factor
            
            # Adapting to the target distribution
            # :===========================================:
            
            # To avoid memory issues, we compute x_norm twice to separate the gradient
            #  computation for the DAE and DDPM losses. 
            for b_i, ((x, y_gt,_,_, bg_mask), (y_pl,)) in enumerate(zip(volume_dataloader, label_dataloader)):
                dae_loss = torch.tensor(0).float().to(device)
                ddpm_loss = torch.tensor(0).float().to(device)
                            
                if not accumulate_over_volume:
                    self.optimizer.zero_grad()

                x = x.to(device).float()
                y_pl = y_pl.to(device)
                               
                n_samples += x.shape[0]
                
                # Calculate gradients from the noise estimation loss
                if b_i in b_i_for_diffusion_loss and self.ddpm_loss_beta > 0 and self.use_ddpm_loss:
                    n_samples_diffusion += x.shape[0] 
                    
                    x_norm = self.norm(x)
                    
                    img = x_norm if self.use_x_norm_for_ddpm_loss else x
                    
                    # Check if the max and min values of the input images have changed
                    #  especially impotant to do it before the fwd/bwd pass of the DDPM when 
                    #  updating first with dae or atlas for some steps           
                    running_max = max(running_max, img.max().item())
                    running_min = min(running_min, img.min().item())
                    
                    if self.use_x_cond_gt:
                        # Only for debugging
                        y_gt = y_gt.to(device)
                        x_cond = y_gt
                    elif self.use_y_pred_for_ddpm_loss:
                        if self.seg_with_bg_supp:
                            bg_mask = bg_mask.to(device)
                            x_norm_bg_supp = background_suppression(x_norm, bg_mask, bg_suppression_opts_tta)
                            x_cond, _ = self.seg(x_norm_bg_supp)
                        else:
                            x_cond, _ = self.seg(x_norm)
                    else:
                        # Uses the pseudo label for the DDPM loss
                        x_cond = y_pl
                    
                    ddpm_loss = self.calculate_ddpm_gradients(
                        img,
                        x_cond,
                        ddpm_reweight_factor=ddpm_reweigh_factor,
                        max_int_norm_imgs=running_max,
                        min_int_norm_imgs=running_min
                    )
                    
                    # Keep track of the max and min in the current step
                    step_max = max(step_max, img.max().item())
                    step_min = min(step_min, img.min().item())
                    
                # Calculate gradients from the segmentation task on the pseudo label    
                if self.dae_loss_alpha > 0:
                    x_norm = self.norm(x)
                    
                    if self.seg_with_bg_supp:
                        bg_mask = bg_mask.to(device)
                        x_norm = background_suppression(x_norm, bg_mask, bg_suppression_opts_tta)

                    mask, _ = self.seg(x_norm)
                    
                    if rescale_factor_dae is not None:
                        mask = mask.permute(1, 0, 2, 3).unsqueeze(0)
                        mask = F.interpolate(mask, scale_factor=rescale_factor_dae, mode='trilinear')
                        mask = mask.squeeze(0).permute(1, 0, 2, 3)

                    dae_loss = self.dae_loss_alpha * self.loss_func(mask, y_pl)
                    
                    if accumulate_over_volume:
                        dae_loss = dae_loss / len(volume_dataloader)
        
                    dae_loss.backward()
            
                if not accumulate_over_volume:
                    self.optimizer.step()              

                with torch.no_grad():
                    step_ddpm_loss += (ddpm_loss.detach() * x.shape[0]).item()
                    step_dae_loss += (dae_loss.detach() * x.shape[0]).item()
                                              
            # Update the max and min values with those of the current step
            #  Only affects running max and min if it is to decrease their range
            running_max = m * running_max + (1 - m) * step_max  
            running_min = m * running_min + (1 - m) * step_min  

            if accumulate_over_volume:
                self.optimizer.step()

            step_ddpm_loss = (step_ddpm_loss / n_samples_diffusion) if n_samples_diffusion > 0 else 0
            step_dae_loss = (step_dae_loss / n_samples) if n_samples > 0 else 0
            tta_loss = step_ddpm_loss + step_dae_loss
            
            self.tta_losses.append(tta_loss)

            if self.wandb_log:
                wandb.log({
                    f'ddpm_loss/img_{index}': step_ddpm_loss, 
                    f'dae_loss/img_{index}': step_dae_loss, 
                    f'total_loss/img_{index}': tta_loss, 
                    'tta_step': step
                    }
                )  

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
        
        dice_scores = {i * calculate_dice_every: score for i, score in enumerate(self.test_scores)}

        return self.norm, self.norm_dict, self.metrics_best, dice_scores
    
    def calculate_ddpm_gradients(
        self,
        img,
        seg,
        ddpm_reweight_factor: float = 1,
        min_int_norm_imgs: float = 0,
        max_int_norm_imgs: float = 1
        ) -> torch.Tensor:
        
        # Normalize the input image between 0 and 1, (required by the DDPM)
        img = du.normalize_min_max(
            img,
            min=min_int_norm_imgs, 
            max=max_int_norm_imgs
            )
        
        if img.max() > 1 or img.min() < 0:
            print(f'WARNING: img.max()={img.max()}, img.min()={img.min()}')
        
        # Upsample the segmentation mask to the same size as the input image, if neccessary
        rescale_factor = np.array(img.shape) / np.array(seg.shape)
        rescale_factor = tuple(rescale_factor[[0, 2, 3]])
        should_rescale = not all([f == 1. for f in rescale_factor])
        
        if should_rescale:
            seg = seg.permute(1, 0, 2, 3).unsqueeze(0)
            seg = F.interpolate(seg, scale_factor=rescale_factor, mode='trilinear')
            seg = (seg > 0.5).float()                
            seg = seg.squeeze(0).permute(1, 0, 2, 3)
        
        # Map seg to a single channel and normalize between 0 and 1
        n_classes = seg.shape[1]
        seg = du.onehot_to_class(seg)
        seg = seg.float() / (n_classes - 1)
        
        # The DDPM is memory intensive, accumulate gradients over minibatches
        ddpm_loss_value = 0
        minibatch_size = self.minibatch_size_ddpm
        num_minibatches = np.ceil(img.shape[0] / minibatch_size)
        ddpm_reweight_factor = ddpm_reweight_factor * (1 / num_minibatches)  
        
        for i in range(0, img.shape[0], minibatch_size):
            img_batch = img[i: i + minibatch_size]
            seg_batch = seg[i: i + minibatch_size]
            ddpm_loss = ddpm_reweight_factor * self.ddpm_loss_beta * self.ddpm(img_batch, seg_batch)
            
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
        wandb.define_metric(f'dae_loss/*', step_metric=f'tta_step')
        wandb.define_metric(f'total_loss/*', step_metric=f'tta_step')   
        wandb.define_metric(f'dice_score_fg/*', step_metric=f'tta_step')    

    def reset_initial_state(self, state_dict: dict) -> None:
        super().reset_initial_state(state_dict)
        self.use_ddpm_loss = self.use_ddpm_after_dice is None and \
            self.use_ddpm_after_step is None
            
    def _sample_atlas_vol(self, num_vol_samples: int = 1) -> torch.Tensor:
        # Normalize Atlas between 0 and 1
        n_classes = self.atlas_vol.shape[0]
        atlas = du.onehot_to_class(self.atlas_vol)
        atlas = atlas.float() / (n_classes - 1)
        
        norm_atlas_dataloader = DataLoader(
            TensorDataset(atlas),
            batch_size=self.minibatch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        
        atlas_sampled_vols = []
        for _ in range(num_vol_samples):
            vol_i = []
            for x_cond in norm_atlas_dataloader:
                x_cond = x_cond.to(self.device)
                vol_i.append(
                    self.ddpm.ddim_sample(x_cond, return_all_timesteps=False)
                )
            
            atlas_sampled_vols.append(torch.vstack(vol_i))
            
        return torch.concat(atlas_sampled_vols, dim=0)