import os
from tqdm import tqdm
from collections import defaultdict
from typing import Optional, Union, Literal


import wandb
import torch
import numpy as np
#from pytorch_msssim import SSIM
from kornia.losses import SSIM3DLoss
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from tta_uia_segmentation.src.tta.TTADAE import TTADAE
from tta_uia_segmentation.src.utils.loss import DescriptorRegularizationLoss, DiceLoss
from tta_uia_segmentation.src.models import ConditionalGaussianDiffusion
from tta_uia_segmentation.src.utils.io import save_checkpoint, write_to_csv
from tta_uia_segmentation.src.utils.utils import get_seed
from tta_uia_segmentation.src.dataset import DatasetInMemory, utils as du


#ssim = SSIM(data_range=1.0, channel=1, spatial_dims=3)
#ssim_loss = lambda x, y: 0.5 * (1 - ssim(x, y))
ssim_loss  = SSIM3DLoss(window_size=11, reduction='mean', max_val=1.0)


def subtract_gradients_dicts(gradients_old: defaultdict, gradients_new: dict) -> dict:
                
        gradients_diff = {}
        for param_name in gradients_new.keys():
            gradients_diff[param_name] = gradients_new[param_name] - gradients_old[param_name]
        
        assert set(gradients_new.keys()) == set(gradients_old.keys()), \
            'gradients_dict_2 must have the same keys as gradients_dict_1'
            
        return gradients_diff  
    
    
def add_gradients_dicts(gradients_dict_1: defaultdict, gradients_dict_2: dict) -> dict:
            
        gradients_sum = {}
        for param_name in gradients_dict_2.keys():
            gradients_sum[param_name] = gradients_dict_1[param_name] + gradients_dict_2[param_name]
        
        assert set(gradients_dict_2.keys()) == set(gradients_dict_1.keys()), \
            'gradients_dict_2 must have the same keys as gradients_dict_1'
            
        return gradients_sum      


class TTADAEandDDPM(TTADAE):
    """
    TODO:
        - DDPM sample guidance
            - Sampled volumes are not in the same intensity range and normalized volumes
            - We should normalize sampled volumes to the same range as the normalized images
            
        - DAE Loss  
            - Determine if there is a gap between this script and the DAE script
    """
    
    def __init__(
        self,
        ddpm: ConditionalGaussianDiffusion,
        dae_loss_alpha: float = 0.5,
        ddpm_loss_beta: float = 1.0,
        ddpm_loss_adaptive_beta_init: float = 1.0,
        ddpm_sample_guidance_eta: float = 0.0,
        x_norm_regularization_loss_gamma: float = 1.0,
        dae_loss: Optional[callable] = DiceLoss(),
        guidance_loss: Optional[callable] = ssim_loss,
        x_norm_regularization_loss: Optional[Literal['sift', 'zncc', 'mi']] = 'sift',
        x_norm_kwargs: dict = {},
        minibatch_size_ddpm: int = 2,
        frac_vol_diffusion_tta: float = 1.0,
        min_t_diffusion_tta: int = 0,
        max_t_diffusion_tta: int = 999,
        sampling_timesteps: Optional[int] = None,
        min_max_int_norm_imgs: tuple[float, float] = (0, 1),
        detach_x_norm_from_ddpm_loss: bool = False,
        use_y_pred_for_ddpm_loss: bool = False,
        use_y_gt_for_ddpm_loss: bool = False,    # Of course use only for debugging
        use_ddpm_after_step: Optional[int] = None,
        use_ddpm_after_dice: Optional[float] = None,
        warmup_steps_for_ddpm_loss: Optional[int] = None,
        **kwargs
        ) -> None:
        
        super().__init__(loss_func=dae_loss, **kwargs)
        self.ddpm = ddpm
        
        self.dae_loss_alpha = dae_loss_alpha
        self.ddpm_loss_beta = ddpm_loss_beta
        self.ddpm_loss_adaptive_beta_init = ddpm_loss_adaptive_beta_init
        self.ddpm_loss_adaptive_beta = ddpm_loss_adaptive_beta_init
        self.ddpm_sample_guidance_eta = ddpm_sample_guidance_eta
        
        self.x_norm_grads_dae_loss = defaultdict(int)          # Gradient of the last layer of the normalizer wrt the DAE loss
        self.x_norm_grads_ddpm_loss = defaultdict(int)         # Gradient of the last layer of the normalizer wrt the DDPM loss
        self.x_norm_grads_x_norm_reg_loss = defaultdict(int)   # Gradient of the last layer of the normalizer wrt the x_norm regularization loss
        
        self.minibatch_size_ddpm = minibatch_size_ddpm
        self.frac_vol_diffusion_tta = frac_vol_diffusion_tta
        
        self.min_t_diffusion_tta = min_t_diffusion_tta
        self.max_t_diffusion_tta = max_t_diffusion_tta
        
        self.min_int_norm_imgs = min_max_int_norm_imgs[0]
        self.max_int_norm_imgs = min_max_int_norm_imgs[1]
        
        self.use_y_pred_for_ddpm_loss = use_y_pred_for_ddpm_loss   
        self.detach_x_norm_from_ddpm_loss = detach_x_norm_from_ddpm_loss 
        
        self.sampling_timesteps = sampling_timesteps
        self.ddpm.set_sampling_timesteps(sampling_timesteps)
        
        # Set DDPM model in eval mode
        self.ddpm.eval()
        self.ddpm.requires_grad_(False)
        
        # Attributes used only for debugging
        self.use_y_gt_for_ddpm_loss = use_y_gt_for_ddpm_loss
        
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
        
        # x_norm regularization parameters
        self.x_norm_regularization_loss_gamma = x_norm_regularization_loss_gamma
        self.x_norm_regularization_loss = DescriptorRegularizationLoss(
            type=x_norm_regularization_loss, **x_norm_kwargs
        ) if self.x_norm_regularization_loss_gamma > 0 else None
        
        # DDPM sample guidance parameters
        self.guidance_loss = guidance_loss
        self.atlas_sampled_vol = self._sample_vol((self.atlas > 0.5).float()) \
            if self.ddpm_sample_guidance_eta > 0 else None
    
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
        running_min_max_momentum: float = 0.96,
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
        
        if self.ddpm_sample_guidance_eta > 0:
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
            
            if self.ddpm_loss_beta > 0 and self.use_ddpm_after_step is not None and not self.use_ddpm_loss:      
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
                        y_dae_or_atlas = y_dae_or_atlas[:, :, 0:self.atlas.shape[2]] # To handle dataset repetitions of the atlas
                    y_dae_or_atlas = y_dae_or_atlas.detach().cpu()    
                else:
                    y_dae_or_atlas = None
                    
                # Get the guidance volume generated with the DDPM to log how it looks like
                if self.ddpm_sample_guidance_eta > 0:
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
                if self.ddpm_loss_beta > 0 and self.use_ddpm_after_dice is not None and not self.use_ddpm_loss:
                    self.use_ddpm_loss = dice_dae >= self.use_ddpm_after_dice
                    if self.use_ddpm_loss:
                        print('---------Start using DDPM loss ---------')
                
                # Sample volumes from the DDPM using the DAE predicted labels 
                if self.ddpm_sample_guidance_eta > 0 and self.using_dae_pl:
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
            b_i_for_ddpm_loss = np.random.choice(
                range(len(volume_dataloader)), num_batches_for_ddpm_loss, replace=False)
            
            # Reweigh factor for the ddpm loss to take into account how many 
            #  times less it is used than the dae loss or if averaged over entire volume
            ddpm_reweigh_factor = (1 / len(b_i_for_ddpm_loss)) * \
                (1 if accumulate_over_volume else len(volume_dataloader))  

            warmup_factor = warmup_steps_for_ddpm_loss.pop() \
                if len(warmup_steps_for_ddpm_loss) > 0 and self.use_ddpm_loss else 1
            
            assert 1 >= warmup_factor >= 0, 'Warmup factor must be between 0 and 1'
                        
            ddpm_reweigh_factor *= warmup_factor 
            
            # Define which volumes to use for the DDPM sample guidance (if any at all)
            # :============================================================:
            if self.ddpm_sample_guidance_eta > 0:
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
            step_x_norm_reg_loss = 0
            
            n_samples = 0
            n_samples_diffusion = 0
            
            self.norm.train()
            volume_dataset.dataset.set_augmentation(True)

            if accumulate_over_volume:
                self.optimizer.zero_grad()
                self.x_norm_grads_dae_loss = defaultdict(int)
                self.x_norm_grads_ddpm_loss = defaultdict(int)
                self.x_norm_grads_x_norm_reg_loss = defaultdict(int)

            if const_aug_per_volume:
                volume_dataset.dataset.set_seed(get_seed())
            
            for b_i, ((x, y_gt,_,_, bg_mask), (y_pl,), (x_sampled,)) in enumerate(zip(volume_dataloader, label_dataloader, sampled_vol_dl)):
                x_norm = None        
                if not accumulate_over_volume:
                    self.optimizer.zero_grad()
                    self.x_norm_grads_dae_loss = defaultdict(int)
                    self.x_norm_grads_ddpm_loss = defaultdict(int)
                    self.x_norm_grads_x_norm_reg_loss = defaultdict(int)

                x = x.to(device).float()                               
                n_samples += x.shape[0]
                
                # DAE loss: Calculate gradients from the segmentation task on the pseudo label  
                # :===============================================================:  
                if self.dae_loss_alpha > 0:
                    x_norm_grads_old = self._get_gradients_x_norm()
                    
                    y_pl = y_pl.to(device)
                    x_norm = self.norm(x)
                    
                    _, mask, _ = self.forward_pass_seg(
                        x, bg_mask, self.bg_suppression_opts_tta, device)
                    
                    if self.rescale_factor is not None:
                        mask = self.rescale_volume(mask)

                    dae_loss = self.dae_loss_alpha * self.loss_func(mask, y_pl)
                    
                    if accumulate_over_volume:
                        dae_loss = dae_loss / len(volume_dataloader)
        
                    dae_loss.backward()
                    
                    # Get the gradient for the last layer of the normalizer
                    x_norm_grads_new = self._get_gradients_x_norm()
                    self.x_norm_grads_dae_loss = add_gradients_dicts(
                        self.x_norm_grads_dae_loss,
                        subtract_gradients_dicts(x_norm_grads_old, x_norm_grads_new),
                    )                 
                    
                # DDPM loss: Calculate gradients from the noise estimation loss
                # :============================================================:
                
                calculate_ddpm_loss_gradients = b_i in b_i_for_ddpm_loss and \
                    self.ddpm_loss_beta > 0 and self.use_ddpm_loss
                    
                if calculate_ddpm_loss_gradients:
                    
                    n_samples_diffusion += x.shape[0] 
                    
                    # Keep track of the max and min in the current step
                    if x_norm is None:
                        x_norm = self.norm(x) 
                    step_max = max(step_max, x_norm.max().item())
                    step_min = min(step_min, x_norm.min().item()) 
                    del x_norm
                    
                    # Fix x_cond depending on the configuration
                    if self.use_y_gt_for_ddpm_loss:
                        # Only for debugging
                        x_cond = y_gt.type(torch.int8).to(device)
                        
                    elif not self.use_y_pred_for_ddpm_loss:
                        x_cond = y_pl.type(torch.int8).to(device)
                        
                        # Upsample the segmentation mask to the same size as the input image, if neccessary
                        rescale_factor = np.array(x.shape) / np.array(x_cond.shape)
                        rescale_factor = tuple(rescale_factor[[0, 2, 3]])       # Volume is DCHW
                        should_rescale = not all([f == 1. for f in rescale_factor])
                        
                        if should_rescale:
                            x_cond = x_cond.permute(1, 0, 2, 3).unsqueeze(0)
                            x_cond = F.interpolate(x_cond, scale_factor=rescale_factor, mode='trilinear')
                            x_cond = (x_cond > 0.5).float()                
                            x_cond = x_cond.squeeze(0).permute(1, 0, 2, 3)

                    # Or leave it undefined and the predicted segmentation mask for each minibatch of x
                    #  will be calculated and used within _calculate_ddpm_gradients()
                    else:
                        x_cond = None
                        
                    x_norm_grads_old = self._get_gradients_x_norm()
                                        
                    ddpm_loss = self._calculate_ddpm_gradients(
                        x,
                        x_cond,
                        ddpm_reweigh_factor=ddpm_reweigh_factor,
                        max_int_norm_imgs=running_max,
                        min_int_norm_imgs=running_min
                    )
                    
                    x_norm_grads_new = self._get_gradients_x_norm()
                    self.x_norm_grads_ddpm_loss = add_gradients_dicts(
                        self.x_norm_grads_ddpm_loss,
                        subtract_gradients_dicts(x_norm_grads_old, x_norm_grads_new),
                    )
                    
                # DDPM sample guidance
                # :===============================================================: 
                if self.ddpm_sample_guidance_eta > 0:
                    print('DEBUG, use_ddpm_sample_guidance')
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
                    
                # Regularize the shift between x and x_norm, to ensure information of the image is preserved
                # :===============================================================:
                if self.x_norm_regularization_loss_gamma > 0:
                    x_norm_grads_old = self._get_gradients_x_norm()
                    
                    x_norm = self.norm(x)
                    x_norm_regularization_loss = self.x_norm_regularization_loss_gamma * \
                        self.x_norm_regularization_loss(x_norm, x)
                    x_norm_regularization_loss.backward()
                    
                    x_norm_grads_new = self._get_gradients_x_norm()
                    self.x_norm_grads_x_norm_reg_loss = add_gradients_dicts(
                        self.x_norm_grads_x_norm_reg_loss,
                        subtract_gradients_dicts(x_norm_grads_old, x_norm_grads_new),
                    )
                                    
                if not accumulate_over_volume:
                    self._take_optimize_step(index, step)              

                with torch.no_grad():
                    step_dae_loss += (dae_loss.detach() * x.shape[0]).item() \
                        if self.dae_loss_alpha > 0 else 0             
                          
                    mini_batch_ddpm_loss = ddpm_loss.detach() * x.shape[0] \
                        if calculate_ddpm_loss_gradients else 0
                    mini_batch_ddpm_loss *= self.ddpm_loss_adaptive_beta if not accumulate_over_volume else 1
                    
                    step_ddpm_loss += mini_batch_ddpm_loss 
                    
                    step_ddpm_guidance_loss += (ddpm_guidance_loss.detach() * x.shape[0]).item() \
                        if self.ddpm_sample_guidance_eta > 0 else 0
                        
                    step_x_norm_reg_loss += (x_norm_regularization_loss.detach() * x.shape[0]).item() \
                        if self.x_norm_regularization_loss_gamma > 0 else 0
                                                                        
            # Update the max and min values with those of the current step
            #  Only affects running max and min if it is to decrease their range
            running_max = m * running_max + (1 - m) * step_max  
            running_min = m * running_min + (1 - m) * step_min  

            if accumulate_over_volume:
                self._take_optimize_step(index, step)

            step_dae_loss = (step_dae_loss / n_samples) if n_samples > 0 else 0
            
            step_ddpm_loss = (step_ddpm_loss / n_samples_diffusion) if n_samples_diffusion > 0 else 0
            step_ddpm_loss *= self.ddpm_loss_adaptive_beta if accumulate_over_volume else 1
            
            step_ddpm_guidance_loss = (step_ddpm_guidance_loss / n_samples) if n_samples > 0 else 0
            
            step_tta_loss = step_dae_loss + step_ddpm_loss + step_ddpm_guidance_loss + step_x_norm_reg_loss
            
            self.tta_losses.append(step_tta_loss)

            if self.wandb_log:
                wandb.log({
                    f'dae_loss/img_{index:03d}': step_dae_loss, 
                    f'ddpm_loss/img_{index:03d}': step_ddpm_loss,
                    f'ddpm_guidance_loss/img_{index:03d}': step_ddpm_guidance_loss,
                    f'x_norm_reg_loss/img_{index:03d}': step_x_norm_reg_loss,
                    f'total_loss/img_{index:03d}': step_tta_loss, 
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
    
    def _calculate_ddpm_gradients(
        self,
        x: torch.Tensor,
        device: str,
        x_cond: Optional[torch.Tensor] = None,
        bg_mask: Optional[torch.Tensor] = None,
        ddpm_reweigh_factor: float = 1,
        min_int_norm_imgs: float = 0,
        max_int_norm_imgs: float = 1,
        min_max_tolerance: float = 2e-1
        ) -> torch.Tensor:
                
        # The DDPM is memory intensive, accumulate gradients over minibatches
        ddpm_loss_value = 0
        minibatch_size = self.minibatch_size_ddpm
        num_minibatches = np.ceil(x.shape[0] / minibatch_size)
        ddpm_reweigh_factor = ddpm_reweigh_factor * (1 / num_minibatches)  
        
        # Multiply the reweight factor by the loss' beta 
        ddpm_reweigh_factor *= self.ddpm_loss_beta  
        
        for i in range(0, x.shape[0], minibatch_size):
            x_minibatch = x[i: i + minibatch_size]
            
            x_norm_mb = self.norm(x_minibatch)
            
            if x_cond is None:
                # Use predicted segmentation mask for the DDPM loss
                _, x_cond_mb, _ = self.forward_pass_seg(
                    x_norm=x_norm_mb, bg_mask=bg_mask, 
                    bg_suppression_opts=self.bg_suppression_opts_tta, 
                    device=device
                )
            else:
                x_cond_mb = x_cond[i: i + minibatch_size]
                    
            # Normalize the input image between 0 and 1, (required by the DDPM)
            x_norm_mb_ddpm = x_norm_mb if not self.detach_x_norm_from_ddpm_loss else x_norm_mb.detach().clone()
            
            x_norm_mb_ddpm = du.normalize_min_max(
                x_norm_mb_ddpm,
                min=min_int_norm_imgs, 
                max=max_int_norm_imgs
                )
            
            if x_norm_mb_ddpm.max() > 1 + min_max_tolerance or x_norm_mb_ddpm.min() < 0 - min_max_tolerance:
                print(f'WARNING: x_norm_mb.max()={x_norm_mb_ddpm.max()}, x_norm_mb.min()={x_norm_mb.min()}')
            
            # Calculate the DDPM loss and backpropagate
            ddpm_loss = ddpm_reweigh_factor * self.ddpm(
                x_norm_mb_ddpm, 
                x_cond_mb
            )
            ddpm_loss.backward()
            
            ddpm_loss_value += ddpm_loss.detach()
                                    
        return ddpm_loss_value
    
    def _take_optimize_step(self, index, step):
        if self.use_ddpm_loss and self.dae_loss_alpha > 0: 
                    
            # Calculate adaptive beta for the DDPM loss
            self._update_ddpm_loss_adaptive_beta(index, step)
                    
            # Replace all gradients in x_norm for grad_dae + beta * grad_ddpm
            assert self.x_norm_grads_dae_loss.keys() == self.x_norm_grads_ddpm_loss.keys(), \
                'Gradients for the DAE and DDPM losses must have the same keys'
                
            for name, param in self.norm.named_parameters():
                if param.grad is not None:
                    param.grad = self.x_norm_grads_dae_loss[name] + \
                        self.ddpm_loss_adaptive_beta * self.x_norm_grads_ddpm_loss[name] + \
                        self.ddpm_loss_adaptive_beta * self.x_norm_grads_x_norm_reg_loss[name]
        
        # Log the magnitude of gradients from the different losses
        if self.wandb_log: self._log_x_norm_out_gradient_magnitudes(index, step)
                    
        # Take optimizer step
        self.optimizer.step()

    
    def _define_custom_wandb_metrics(self, ):
        wandb.define_metric(f'tta_step')
        wandb.define_metric(f'dae_loss/*', step_metric=f'tta_step')
        wandb.define_metric(f'ddpm_loss/*', step_metric=f'tta_step')
        wandb.define_metric(f'ddpm_guidance_loss/*', step_metric=f'tta_step')
        wandb.define_metric(f'total_loss/*', step_metric=f'tta_step')   
        wandb.define_metric(f'dice_score_fg/*', step_metric=f'tta_step')    

    def reset_initial_state(self, state_dict: dict) -> None:
        super().reset_initial_state(state_dict)
        
        self.ddpm_loss_adaptive_beta = self.ddpm_loss_adaptive_beta_init
        self.x_norm_grads_dae_loss = defaultdict(int)          # Gradient of the last layer of the normalizer wrt the DAE loss
        self.x_norm_grads_ddpm_loss = defaultdict(int)         # Gradient of the last layer of the normalizer wrt the DDPM loss
        self.x_norm_grads_x_norm_reg_loss = defaultdict(int)   # Gradient of the last layer of the normalizer wrt the x_norm regularization loss
        
        self.use_ddpm_loss = self.use_ddpm_after_dice is None and \
            self.use_ddpm_after_step is None
                    
    def _get_gradients_x_norm(self) -> dict:
        gradients_dict = defaultdict(int)
        for name, param in self.norm.named_parameters():
            if param.grad is not None:
                gradients_dict[name] = param.grad.clone().detach()  # Store gradients and detach to avoid memory leaks
                
        return gradients_dict
    
    def _log_x_norm_out_gradient_magnitudes(self, index, step):
        # Get the name of the last layer
        last_layer_name = [name for name, _ in self.norm.named_parameters() if 'weight' in name][-1]
        
        log_dict = {'tta_step': step}
        
        if self.dae_loss_alpha > 0:
            log_dict[f'norm_x_norm_out_grad_dae_loss/img_{index:03d}'] = \
                torch.norm(self.x_norm_grads_dae_loss[last_layer_name])
        
        if self.ddpm_loss_beta > 0:
            log_dict[f'norm_x_norm_out_grad_ddpm_loss/img_{index:03d}'] = \
                torch.norm(self.x_norm_grads_ddpm_loss[last_layer_name])
                
        if self.x_norm_regularization_loss_gamma > 0:
            log_dict[f'norm_x_norm_out_grad_xnorm_reg_loss/img_{index:03d}'] = \
                torch.norm(self.x_norm_grads_x_norm_reg_loss[last_layer_name])        
        
        wandb.log(log_dict)
    
    def _update_ddpm_loss_adaptive_beta(self, index, step):
        # Get the name of the last layer
        last_layer_name = [name for name, _ in self.norm.named_parameters() if 'weight' in name][-1]
        norm_x_norm_out_grad_dae_loss = torch.norm(self.x_norm_grads_dae_loss[last_layer_name])
        norm_x_norm_out_grad_ddpm_loss = torch.norm(self.x_norm_grads_ddpm_loss[last_layer_name])
        λ = norm_x_norm_out_grad_dae_loss / (norm_x_norm_out_grad_ddpm_loss + 1e-7)
        λ = torch.clamp(λ, 0, 1e4).detach()
        self.ddpm_loss_adaptive_beta = 0.8 * λ 
        
        if self.wandb_log:
            wandb.log({
                f'ddpm_loss_adaptive_beta/img_{index:03d}': self.ddpm_loss_adaptive_beta,
                'tta_step': step
            })
    
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