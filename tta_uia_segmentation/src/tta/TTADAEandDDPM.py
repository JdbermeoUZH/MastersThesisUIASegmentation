import os
from typing import Optional, Union
from tqdm import tqdm

import wandb
import torch
import numpy as np
#from pytorch_msssim import SSIM
from kornia.losses import SSIM3DLoss
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from tta_uia_segmentation.src.tta.TTADAE import TTADAE
from tta_uia_segmentation.src.models import ConditionalGaussianDiffusion
from tta_uia_segmentation.src.models.normalization import background_suppression
from tta_uia_segmentation.src.utils.io import save_checkpoint, write_to_csv
from tta_uia_segmentation.src.utils.utils import get_seed
from tta_uia_segmentation.src.dataset import DatasetInMemory, utils as du


#ssim = SSIM(data_range=1.0, channel=1, spatial_dims=3)
#ssim_loss = lambda x, y: 0.5 * (1 - ssim(x, y))
ssim_loss  = SSIM3DLoss(window_size=11, reduction='mean', max_val=1.0)

class TTADAEandDDPM(TTADAE):
    """
    TODO:
        - DDPM sample guidance
            - Sampled volumes are not in the same intensity range and normalized volumes
            - We should normalize sampled volumes to the same range as the normalized images
        
        - DDPM loss
            - Use importance sampling to make the loss function less noisy
            
        - DAE Loss  
            - Determine if there is a gap between this script and the DAE script
    """
    
    def __init__(
        self,
        ddpm: ConditionalGaussianDiffusion,
        dae_loss_alpha: float = 0.5,
        ddpm_loss_beta: float = 1.0,
        ddpm_sample_guidance_eta: Optional[float] = None,
        guidance_loss: Optional[callable] = ssim_loss,
        minibatch_size_ddpm: int = 2,
        frac_vol_diffusion_tta: float = 1.0,
        min_t_diffusion_tta: int = 0,
        max_t_diffusion_tta: int = 999,
        sampling_timesteps: Optional[int] = None,
        min_max_int_norm_imgs: tuple[float, float] = (0, 1),
        use_x_norm_for_ddpm_loss: bool = True,
        use_y_pred_for_ddpm_loss: bool = False,
        use_x_cond_gt: bool = False,    # Of course use only for debugging
        use_ddpm_after_step: Optional[int] = None,
        use_ddpm_after_dice: Optional[float] = None,
        warmup_steps_for_ddpm_loss: Optional[int] = None,
        **kwargs
        ) -> None:
        
        super().__init__(**kwargs)
        self.ddpm = ddpm
        
        self.dae_loss_alpha = dae_loss_alpha
        self.ddpm_loss_beta = ddpm_loss_beta
        self.ddpm_sample_guidance_eta = ddpm_sample_guidance_eta
        
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
        
        # Set DDPM model in eval mode
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
        self.use_ddpm_sample_guidance = self.ddpm_sample_guidance_eta is not None and \
            self.ddpm_sample_guidance_eta > 0
        self.guidance_loss = guidance_loss
        self.atlas_sampled_vol = self._sample_vol((self.atlas > 0.5).float()) \
            if self.use_ddpm_sample_guidance else None
    
    def tta(
        self,
        volume_dataset: DatasetInMemory,
        dataset_name: str,
        index: int,
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
        running_min_max_momentum: float = 0.95,
    ):       
        """_summary_

        Arguments:
        ----------
        volume_dataset : DatasetInMemory
            Dataset containing slices of a single volume on which to perform TTA.
        """
        
        self.tta_losses = []
        self.test_scores = []
        
        label_dataloader = None
        dae_sampled_vol = None
            
        self.seg.requires_grad_(False)
    
        if self.rescale_factor is not None:
            assert (batch_size * self.rescale_factor[0]) % 1 == 0
            label_batch_size = int(batch_size * self.rescale_factor[0])
        else:
            label_batch_size = batch_size

        dae_dataloader = DataLoader(
            volume_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        )

        volume_dataloader = DataLoader(
            ConcatDataset([volume_dataset] * dataset_repetition),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
        )
        
        if self.use_ddpm_sample_guidance:
            atlas_sampled_vol_dl = DataLoader(
                TensorDataset(self.atlas_sampled_vol.permute(1, 0, 2, 3, 4)), # NDCHW -> DNCHW over depth in 2D 
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
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
            
        # Define the number of batches to use for the DDPM loss for per step (full pass over the volume)
        num_batches_for_ddpm_loss = int(self.frac_vol_diffusion_tta * len(volume_dataloader))
                
        for step in tqdm(range(num_steps)):
            
            self.norm.eval()
            volume_dataset.dataset.set_augmentation(False)
            
            step_min, step_max = np.inf, -np.inf
            
            if self.use_ddpm_after_step is not None and not self.use_ddpm_loss:      
                self.use_ddpm_loss = step >= self.use_ddpm_after_step
                if self.use_ddpm_loss:
                    print('---------Start using DDPM loss ---------')
    
            # Mesaure performance during adaptation
            # :===============================================================:
            if step % calculate_dice_every == 0 and calculate_dice_every != -1:
                
                # Get the pseudo label from the DAE or Atlas to log how it looks like 
                if self.dae_loss_alpha > 0:
                    if not self.using_dae_pl:
                        y_dae_or_atlas = self.atlas 
                    else:
                        y_dae_or_atlas = []
                        for (y_dae_mb,) in label_dataloader:
                            y_dae_or_atlas.append(y_dae_mb)
                        y_dae_or_atlas = torch.vstack(y_dae_or_atlas)
                        y_dae_or_atlas = y_dae_or_atlas.permute(1, 0, 2, 3).unsqueeze(0) # make NCDHW
                    y_dae_or_atlas = y_dae_or_atlas.detach().cpu()    
                else:
                    y_dae_or_atlas = None
                    
                # Get the guidance volume generated with the DDPM to log how it looks like
                if self.use_ddpm_sample_guidance:
                    x_guidance = dae_sampled_vol if self.using_dae_pl else self.atlas_sampled_vol
                    x_guidance = x_guidance[0:1]
                    x_guidance = x_guidance.permute(0, 2, 1, 3, 4) # NDCHW -> NCDHW
                    x_guidance = x_guidance.detach().cpu()  
                else:
                    x_guidance = None

                _, dices_fg = self.test_volume(
                    volume_dataset=volume_dataset,
                    dataset_name=dataset_name,
                    y_dae_or_atlas=y_dae_or_atlas,
                    x_guidance=x_guidance,
                    logdir=logdir,
                    device=device,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    index=index,
                    iteration=step,
                    bg_suppression_opts=self.bg_suppression_opts,
                )
                self.test_scores.append(dices_fg.mean().item())

            # Update Pseudo label, with DAE or Atlas, depending on which has a better agreement
            # :===============================================================:
            if step % update_dae_output_every == 0:
                # Only update the pseudo label if it has not been calculated yet or
                #  if the beta is less than 1.0

                if step == 0 or self.beta <= 1.0:
                    dice_dae, _, label_dataloader = self.generate_pseudo_labels(
                        dae_dataloader=dae_dataloader,
                        label_batch_size=label_batch_size,
                        device=device,
                        num_workers=num_workers,
                        dataset_repetition=dataset_repetition
                    )
                    
                # Check whether to use the DDPM loss based on the dice score flag
                if self.use_ddpm_after_dice is not None and not self.use_ddpm_loss:
                    self.use_ddpm_loss = dice_dae >= self.use_ddpm_after_dice
                    if self.use_ddpm_loss:
                        print('---------Start using DDPM loss ---------')
                
                # Sample volumes from the DDPM using the DAE predicted labels 
                if self.use_ddpm_sample_guidance and self.using_dae_pl:
                    dae_sampled_vol = self._sample_vol(label_dataloader)
                    dae_sampled_vol_dl = DataLoader(
                        TensorDataset(dae_sampled_vol.permute(1, 0, 2, 3, 4)), # NDCHW -> DNCHW over depth in 2D
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        drop_last=False,
                    )
                    
            # Define parameters related to the DDPM loss for the current step
            # :============================================================:
            # Sample on which specific batches of images to use the DDPM loss for this step
            b_i_for_diffusion_loss = np.random.choice(
                range(len(volume_dataloader)), num_batches_for_ddpm_loss, replace=False)
            
            # Reweigh factor for the ddpm loss to take into account how many 
            #  times less it is used than the dae loss or if averaged over entire volume
            ddpm_reweigh_factor = (1 / len(b_i_for_diffusion_loss)) * \
                (1 if accumulate_over_volume else len(volume_dataloader))  

            warmup_factor = warmup_steps_for_ddpm_loss.pop() \
                if len(warmup_steps_for_ddpm_loss) > 0 and self.use_ddpm_loss else 1
            
            assert 1 >= warmup_factor >= 0, 'Warmup factor must be between 0 and 1'
                        
            ddpm_reweigh_factor = warmup_factor * ddpm_reweigh_factor
            
            # Define which volumes to use for the DDPM sample guidance (if any at all)
            # :============================================================:
            if self.use_ddpm_sample_guidance:
                sampled_vol_dl = dae_sampled_vol_dl if self.using_dae_pl else atlas_sampled_vol_dl
            else:
                sampled_vol_dl = [[None]] * len(volume_dataloader)
                
                
            # :===========================================:
            # Adapting to the target distribution
            # :===========================================:
            
            step_tta_loss = 0
            step_dae_loss = 0
            step_ddpm_loss = 0
            step_ddpm_guidance_loss = 0
            
            n_samples = 0
            n_samples_diffusion = 0

            self.norm.train()
            volume_dataset.dataset.set_augmentation(True)

            if accumulate_over_volume:
                self.optimizer.zero_grad()

            if const_aug_per_volume:
                volume_dataset.dataset.set_seed(get_seed())
            
            # To avoid memory issues, we compute x_norm twice to separate the gradient
            #  computation for the DAE and DDPM losses. 
            for b_i, ((x, y_gt,_,_, bg_mask), (y_pl,), (x_sampled,)) in enumerate(zip(volume_dataloader, label_dataloader, sampled_vol_dl)):
                            
                if not accumulate_over_volume:
                    self.optimizer.zero_grad()

                x = x.to(device).float()
                y_pl = y_pl.to(device)
                               
                n_samples += x.shape[0]
                
                # DAE loss: Calculate gradients from the segmentation task on the pseudo label  
                # :===============================================================:  
                if self.dae_loss_alpha > 0:
                    x_norm = self.norm(x)
                    
                    _, mask, _ = self.forward_pass_seg(
                        x, bg_mask, self.bg_suppression_opts_tta, device)
                    
                    if self.rescale_factor is not None:
                        mask = self.rescale_volume(mask)

                    dae_loss = self.dae_loss_alpha * self.loss_func(mask, y_pl)
                    
                    if accumulate_over_volume:
                        dae_loss = dae_loss / len(volume_dataloader)
        
                    dae_loss.backward()
                    
                 # DDPM loss: Calculate gradients from the noise estimation loss
                # :============================================================:
                calculate_ddpm_loss_gradients = b_i in b_i_for_diffusion_loss and \
                    self.ddpm_loss_beta > 0 and self.use_ddpm_loss
                    
                if calculate_ddpm_loss_gradients:
                    n_samples_diffusion += x.shape[0] 
                    
                    x_norm = self.norm(x)
                    
                    img = x_norm if self.use_x_norm_for_ddpm_loss else x
                    
                    # Keep track of the max and min in the current step
                    step_max = max(step_max, img.max().item())
                    step_min = min(step_min, img.min().item())
                    
                    if self.use_x_cond_gt:
                        # Only for debugging
                        y_gt = y_gt.to(device)
                        x_cond = y_gt
                        
                    elif self.use_y_pred_for_ddpm_loss:
                        # Use predicted segmentation mask for the DDPM loss
                        _, x_cond, _ = self.forward_pass_seg(
                            x_norm=x_norm, bg_mask=bg_mask, 
                            bg_suppression_opts=self.bg_suppression_opts_tta, 
                            device=device
                        )
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
                    
                # DDPM sample guidance
                # :===============================================================: 
                if self.use_ddpm_sample_guidance:
                    print('DEBG, use_ddpm_sample_guidance')
                    x_norm = self.norm(x)
                    x_sampled = x_sampled.to(device)    
                    
                    x_norm = x_norm.unsqueeze(1).repeat(1, x_sampled.shape[1], 1, 1, 1) # BCHW -> BNCHW to match x_sampled 
                    
                    x_norm = x_norm.permute(1, 2, 0, 3, 4)       # BNCHW -> NCBHW to compare as volumes, B = Depth
                    x_sampled = x_sampled.permute(1, 2, 0, 3, 4) # BNCHW -> NCBHW to compare as volumes, B = Depth
                    
                    # Calculate guidance loss wrt to Atlas or DAE sampled volumes                                
                    ddpm_guidance_loss = self.ddpm_sample_guidance_eta * \
                            self.guidance_loss(x_norm, x_sampled)
                            
                    if accumulate_over_volume:
                        ddpm_guidance_loss = ddpm_guidance_loss / len(volume_dataloader)    
                        
                    ddpm_guidance_loss.backward()
                                    
                if not accumulate_over_volume:
                    self.optimizer.step()              

                with torch.no_grad():
                    step_dae_loss += (dae_loss.detach() * x.shape[0]).item() \
                        if self.dae_loss_alpha > 0 else 0               
                    step_ddpm_loss += (ddpm_loss.detach() * x.shape[0]).item() \
                        if calculate_ddpm_loss_gradients else 0 
                    step_ddpm_guidance_loss += (ddpm_guidance_loss.detach() * x.shape[0]).item() \
                        if self.use_ddpm_sample_guidance else 0
                                              
            # Update the max and min values with those of the current step
            #  Only affects running max and min if it is to decrease their range
            running_max = m * running_max + (1 - m) * step_max  
            running_min = m * running_min + (1 - m) * step_min  

            if accumulate_over_volume:
                self.optimizer.step()

            step_dae_loss = (step_dae_loss / n_samples) if n_samples > 0 else 0
            step_ddpm_loss = (step_ddpm_loss / n_samples_diffusion) if n_samples_diffusion > 0 else 0
            step_ddpm_guidance_loss = (step_ddpm_guidance_loss / n_samples) if n_samples > 0 else 0
            step_tta_loss = step_dae_loss + step_ddpm_loss + step_ddpm_guidance_loss
            
            self.tta_losses.append(step_tta_loss)

            if self.wandb_log:
                wandb.log({
                    f'dae_loss/img_{index}': step_dae_loss, 
                    f'ddpm_loss/img_{index}': step_ddpm_loss,
                    f'ddpm_guidance_loss/img_{index}': step_ddpm_guidance_loss,
                    f'total_loss/img_{index}': step_tta_loss, 
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

        return self.norm_dict, self.metrics_best, dice_scores
    
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
        wandb.define_metric(f'dae_loss/*', step_metric=f'tta_step')
        wandb.define_metric(f'ddpm_loss/*', step_metric=f'tta_step')
        wandb.define_metric(f'ddpm_guidance_loss/*', step_metric=f'tta_step')
        wandb.define_metric(f'total_loss/*', step_metric=f'tta_step')   
        wandb.define_metric(f'dice_score_fg/*', step_metric=f'tta_step')    

    def reset_initial_state(self, state_dict: dict) -> None:
        super().reset_initial_state(state_dict)
        
        self.use_ddpm_loss = self.use_ddpm_after_dice is None and \
            self.use_ddpm_after_step is None
                
        self.use_ddpm_sample_guidance = self.ddpm_sample_guidance_eta is not None and \
            self.ddpm_sample_guidance_eta > 0
        
    def _sample_vol(self, x_cond_vol: Union[torch.Tensor, DataLoader], num_sampled_vols: int = 1, 
                    convert_onehot_to_cat: bool = True) -> torch.Tensor:
               
        if isinstance(x_cond_vol, DataLoader):
            vol_dataloader = x_cond_vol
            
        else:
            vol_dataloader = DataLoader(
                TensorDataset(x_cond_vol.squeeze().permute(1, 0, 2, 3)),  # NDCHW -> DCHW over depth in 2D
                batch_size=self.minibatch_size_ddpm,
                shuffle=False,
                drop_last=False,
            )
        
        sampled_vols = []
        for _ in range(num_sampled_vols):
            vol_i = []
            for (x_cond, )in vol_dataloader:
                # print('DEBUG: x_cond.shape', x_cond.shape)
                # print('DEBUG: x_cond.min()', x_cond.min())
                # print('DEBUG: x_cond.max()', x_cond.max())
                if convert_onehot_to_cat:
                    x_cond = du.onehot_to_class(x_cond)
                    x_cond = x_cond.float() / (self.n_classes - 1)
                    # print('DEBUG, after onehot_to_class and normalizing')
                    # print('DEBUG: x_cond.shape', x_cond.shape)
                    # print('DEBUG: x_cond.min()', x_cond.min())
                    # print('DEBUG: x_cond.max()', x_cond.max())

                x_cond = x_cond.to(self.device)

                vol_i.append(
                    self.ddpm.ddim_sample(x_cond, return_all_timesteps=False)
                )
                
            # Concatenate and upsample the generated volumes, add batch and channel dimensions
            vol_i = torch.vstack(vol_i)
            
            # Upsample the volume to the same size as the input image
            vol_i = self.rescale_volume(vol_i, how='up', return_dchw=True)
            vol_i = vol_i.unsqueeze(0)
            
            # Add upsampled volume to the list
            sampled_vols.append(vol_i) # add batch dimensions
            
        return torch.concat(sampled_vols, dim=0).cpu() # NDCHW