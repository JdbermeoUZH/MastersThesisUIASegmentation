import os
import copy
from tqdm import tqdm
from collections import defaultdict
from typing import Optional, Union, Literal   

import wandb
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset


from tta_uia_segmentation.src.tta import TTADAE
from tta_uia_segmentation.src.models import ConditionalGaussianDiffusion
from tta_uia_segmentation.src.utils.io import save_checkpoint, write_to_csv
from tta_uia_segmentation.src.utils.loss import dice_score
from tta_uia_segmentation.src.utils.visualization import export_images
from tta_uia_segmentation.src.utils.utils import get_seed, stratified_sampling
from tta_uia_segmentation.src.dataset import DatasetInMemory, utils as du


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
    

class DiffusionTTA_and_TTADAE_indp(TTADAE):
    
    def __init__(
        self,
        norm: torch.nn.Module,
        seg: torch.nn.Module,
        ddpm: ConditionalGaussianDiffusion,
        dae: torch.nn.Module,
        n_classes: int,
        learning_rate_dae: float,
        learning_rate_ddpm: float,
        dae_loss_alpha: float = 1.0,
        ddpm_loss_beta: float = 1.0,
        classes_of_interest: Optional[list[int]] = None,
        fit_norm_params: bool = True,
        fit_seg_params: bool = True, 
        fit_ddpm_params: bool = True, 
        ddpm_loss: Literal['jacobian', 'sds', 'dds', 'pds'] = 'jacobian',
        t_ddpm_range: tuple[float, float] = [0, 1.0],
        min_max_intensity_imgs: tuple[float, float] = (0, 1), 
        pair_sampling_type: Literal['one_per_volume', 'one_per_image'] = 'one_per_volume',
        t_sampling_strategy: Literal['uniform', 'stratified'] = 'uniform',
        w_cfg: float = 0.0,
        minibatch_size_ddpm: int = 2,
        wandb_log: bool = False,
        **kwargs
        ) -> None:
        
        self.norm = norm
        self.seg = seg
        self.ddpm = ddpm
        self.dae = dae  
        
        self.dae_loss_alpha = dae_loss_alpha
        self.ddpm_loss_beta = ddpm_loss_beta
        
        # Initialize TTADAE class
        super().__init__(norm=norm, seg=seg, dae=dae,
                         learning_rate=learning_rate_dae,
                         n_classes=n_classes,
                         norm_sd_statistics=None,**kwargs)
        
        # Save the initial state of the networks
        self.norm_dict_sd = copy.deepcopy(self.norm.state_dict())
        self.seg_dict_sd = copy.deepcopy(self.seg.state_dict())
        self.ddpm_dict_sd = copy.deepcopy(self.ddpm.state_dict())
        self.dae_dict_sd = copy.deepcopy(self.dae.state_dict())
        
        # Define dictionary to store the gradients of norm
        self.x_norm_grads_dae_loss = defaultdict(int)
        self.x_norm_grads_ddpm_loss = defaultdict(int)
        
        # Define learning rates and optimizer  
        self.learning_rate_dae = learning_rate_dae
        self.learning_rate_ddpm = learning_rate_ddpm
        
        self.optimizer_dae = torch.optim.AdamW(
            [
                {'params': self.norm.parameters(), 'lr': self.learning_rate_dae},
            ]
        )
        
        self.optimizer_ddpm = torch.optim.AdamW(
            [
                {'params': self.norm.parameters(), 'lr': self.learning_rate_ddpm},
                {'params': self.seg.parameters(), 'lr': self.learning_rate_ddpm},
                {'params': self.ddpm.parameters(), 'lr': self.learning_rate_ddpm}
            ]
        )
        
        # DDPM loss parameters  
        self.ddpm_loss = ddpm_loss
        self.w_cfg = w_cfg
        
        self.pair_sampling_type = pair_sampling_type 
        self.t_sampling_strategy = t_sampling_strategy
        self.min_t_diffusion_tta = int(np.ceil(t_ddpm_range[0] * (self.ddpm.num_timesteps - 1)))
        self.max_t_diffusion_tta = int(np.floor(t_ddpm_range[1] * (self.ddpm.num_timesteps - 1)))
        
        self.min_intensity_imgs = min_max_intensity_imgs[0]
        self.max_intensity_imgs = min_max_intensity_imgs[1]
        
        self.minibatch_size_ddpm = minibatch_size_ddpm
        
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
        
        # Define parameters for evaluation
        self.n_classes = n_classes
        self.classes_of_interest = classes_of_interest
        
        # Setup the custom metrics and steps wandb
        self.wandb_log = wandb_log
        
        if self.wandb_log:
            self._define_custom_wandb_metrics() 
    
    def tta(
        self,
        volume_dataset: DatasetInMemory,
        dataset_name: str,
        index: int,
        num_steps: int,
        accumulate_over_volume: bool,
        num_t_noise_pairs_per_img: int,
        batch_size: int,
        num_workers: int,
        calculate_dice_every: int,
        update_dae_output_every: int,
        dataset_repetition: int,
        const_aug_per_volume: bool,
        device: str,
        logdir: Optional[str] = None,     
        save_checkpoints: bool = False,
    ):       
        """_summary_

        Arguments:
        ----------
        volume_dataset : DatasetInMemory
            Dataset containing slices of a single volume on which to perform TTA.
        """
        self.tta_losses = []
        self.tta_score = []
        
        pseudo_label_dataloader = None
        
        if self.rescale_factor is not None:
            assert (batch_size * self.rescale_factor[0]) % 1 == 0
            pseudo_label_batch_size = int(batch_size * self.rescale_factor[0])
        else:
            pseudo_label_batch_size = batch_size
            
        # Define the sampler object for the volume dataset
        volume_dataloader = DataLoader(
            ConcatDataset([volume_dataset] * dataset_repetition),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        
        for step in tqdm(range(num_steps)):
            # Measure segmentation performance during adaptation
            # :===========================================: 
            if step % calculate_dice_every == 0 and calculate_dice_every != -1:
                self.norm.eval()
                #self.seg.eval()
                volume_dataset.dataset.set_augmentation(False)
                y_dae_or_atlas = self._get_current_pseudo_label(pseudo_label_dataloader)
                _, dices_fg = self.test_volume(
                    volume_dataset=volume_dataset,
                    dataset_name=dataset_name,
                    y_dae_or_atlas=y_dae_or_atlas,
                    logdir=logdir,
                    device=device,
                    num_workers=num_workers,
                    batch_size=batch_size,
                    index=index,
                    iteration=step,
                    classes_of_interest=self.classes_of_interest,
                )
                self.tta_score.append(dices_fg.mean().item())

                # Reset the state of the networks and dataloader
                self.norm.train() if self.fit_norm_params else self.norm.eval()
                self.seg.train() if self.fit_seg_params else self.seg.eval()
                volume_dataset.dataset.set_augmentation(True)

            # DAE: Get pseudo label for current step 
            # :===========================================:
            if step % update_dae_output_every == 0 and self.dae_loss_alpha > 0:
                if step == 0 or self.beta <= 1.0:
                    dice_dae, _, pseudo_label_dataloader = self.generate_pseudo_labels(
                        dae_dataloader=volume_dataloader,
                        label_batch_size=pseudo_label_batch_size,
                        device=device,
                        num_workers=num_workers,
                        dataset_repetition=dataset_repetition
                    )
                else:
                    pseudo_label_dataloader = [[None]] * len(volume_dataloader)
             
            # DDPM: Get (t, epsilon) tuples for the current step if using a given pair for the entire volume
            # :===========================================:
            # Sample t and noise for the DDPM
            if self.pair_sampling_type == 'one_per_volume':
                t_noise_dl = self._sample_t_noise_pairs(
                    num_samples=num_t_noise_pairs_per_img,
                    dl_batch_size=batch_size,
                    num_workers=num_workers,
                    pair_sampling_type=self.pair_sampling_type  
                )
            
            # 1) DAE loss: Calculate gradients from the segmentation task on the pseudo label  
            # :===============================================================:  
            step_dae_loss = 0
            n_samples = 0
            
            if accumulate_over_volume:
                self.optimizer_dae.zero_grad()
                self.optimizer_ddpm.zero_grad()
                self.x_norm_grads_dae_loss = defaultdict(int)

            if const_aug_per_volume:
                volume_dataset.dataset.set_seed(get_seed())
        
            for (x, _,_,_, bg_mask), (y_pl,) in zip(volume_dataloader, pseudo_label_dataloader):
                if not accumulate_over_volume:
                    self.optimizer_dae.zero_grad()
                    self.optimizer_ddpm.zero_grad()
                    self.x_norm_grads_dae_loss = defaultdict(int)
                    
                x = x.to(device).float()    
                y_pl = y_pl.to(device)
                bg_mask = bg_mask.to(device).float()                 
                n_samples += x.shape[0]

                if self.dae_loss_alpha > 0:
                    x_norm_grads_old = self._get_gradients_x_norm()
                    
                    _, mask, _ = self.forward_pass_seg(
                        x, bg_mask, self.bg_supp_x_norm_dae, self.bg_suppression_opts_tta, device,
                        manually_norm_img_before_seg=self.manually_norm_img_before_seg_tta
                    )
                    
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
                
                if not accumulate_over_volume:
                    self.optimizer_dae.step()
                    if self.wandb_log: self._log_x_norm_out_gradient_magnitudes(index, step, log_ddpm_grad_norm=False)
                                                    
                with torch.no_grad():
                    step_dae_loss += (dae_loss.detach() * x.shape[0]).item() \
                        if self.dae_loss_alpha > 0 else 0           
            
            step_dae_loss = (step_dae_loss / n_samples) if n_samples > 0 else 0
                            
            if accumulate_over_volume:
                self.optimizer_dae.step()
                if self.wandb_log: self._log_x_norm_out_gradient_magnitudes(index, step, log_ddpm_grad_norm=False)
                
            if self.wandb_log:
                wandb.log({
                    f'dae_loss/img_{index:03d}': step_dae_loss, 
                    'tta_step': step
                    }
                )  
            
            step_ddpm_loss = 0
            n_samples = 0
            
            if accumulate_over_volume:
                self.optimizer_dae.zero_grad()
                self.optimizer_ddpm.zero_grad()
                self.x_norm_grads_ddpm_loss = defaultdict(int)

            if const_aug_per_volume:
                volume_dataset.dataset.set_seed(get_seed())
        
            for (x, _,_,_, _), (y_pl,) in zip(volume_dataloader, pseudo_label_dataloader):
                if not accumulate_over_volume:
                    self.optimizer_dae.zero_grad()
                    self.optimizer_dae.zero_grad()
                    self.x_norm_grads_ddpm_loss = defaultdict(int)
        
                # 2) DDPM gradient calculation 
                if self.pair_sampling_type == 'one_per_image':
                    t_noise_dl = self._sample_t_noise_pairs(
                        num_samples=num_t_noise_pairs_per_img,
                        dl_batch_size=batch_size, 
                        num_workers=num_workers,
                        pair_sampling_type=self.pair_sampling_type
                    )
                
                x_norm_grads_old = self._get_gradients_x_norm()
                for (t,), (noise,) in t_noise_dl:
                    ddpm_reweigh_factor = 1 / (num_t_noise_pairs_per_img)
                    
                    if accumulate_over_volume:
                        ddpm_reweigh_factor *= 1 / len(volume_dataloader) 
                    
                    assert x.shape[0] == t.shape[0] == noise.shape[0], 'Number of samples must match'      
                                        
                    # Calculate gradients of the batch
                    ddpm_loss = self._calculate_ddpm_gradients(
                        x,
                        t=t,
                        noise=noise,
                        ddpm_reweigh_factor=ddpm_reweigh_factor,
                        min_int_imgs=self.min_intensity_imgs,
                        max_int_imgs=self.max_intensity_imgs,
                    )
                    
                x_norm_grads_new = self._get_gradients_x_norm()
                self.x_norm_grads_ddpm_loss = add_gradients_dicts(
                    self.x_norm_grads_ddpm_loss,
                    subtract_gradients_dicts(x_norm_grads_old, x_norm_grads_new),
                )
                    
                if not accumulate_over_volume:
                    self.optimizer.step()
                    if self.wandb_log: self._log_x_norm_out_gradient_magnitudes(index, step, log_dae_grad_norm=False)
                                                    
                with torch.no_grad():
                    step_ddpm_loss += (ddpm_loss.detach() * x.shape[0]).item() \
                        if self.ddpm_loss_beta > 0 else 0      
        
            step_ddpm_loss = (step_ddpm_loss / n_samples) if n_samples > 0 else 0  
            
            if accumulate_over_volume:
                self.optimizer_ddpm.step()
                if self.wandb_log: self._log_x_norm_out_gradient_magnitudes(index, step, log_dae_grad_norm=False)
                
            step_tta_loss = step_dae_loss + step_ddpm_loss                                                                    
                        
            self.tta_losses.append(step_tta_loss)  
                
            if self.wandb_log:
                wandb.log({
                    f'ddpm_loss/img_{index:03d}': step_ddpm_loss,
                    f'total_loss/img_{index:03d}': step_tta_loss, 
                    'tta_step': step
                    }
                )  

        if save_checkpoints:
            self._save_checkpoint(logdir, dataset_name, index)
            
        os.makedirs(os.path.join(logdir, 'metrics'), exist_ok=True)

        os.makedirs(os.path.join(logdir, 'tta_score'), exist_ok=True)
        
        write_to_csv(
            os.path.join(logdir, 'tta_score', f'{dataset_name}_{index:03d}.csv'),
            np.array([self.tta_score]).T,
            header=['tta_score'],
            mode='w',
        )
        
        dice_scores = {i * calculate_dice_every: score for i, score in enumerate(self.tta_score)}

        return dice_scores
    
    def _calculate_ddpm_gradients(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        ddpm_reweigh_factor: float = 1,
        min_int_imgs: float = 0,
        max_int_imgs: float = 1,
        min_max_tolerance: float = 2e-1,
        ) -> torch.Tensor:
        """
        
        Assumes x_cond is one hot encoded
        
        Args:
            x (torch.Tensor): _description_
            device (str): _description_
            x_cond (Optional[torch.Tensor], optional): _description_. Defaults to None.
            t (Optional[torch.Tensor], optional): _description_. Defaults to None.
            ddpm_reweigh_factor (float, optional): _description_. Defaults to 1.
            min_int_norm_imgs (float, optional): _description_. Defaults to 0.
            max_int_norm_imgs (float, optional): _description_. Defaults to 1.
            use_unconditional_ddpm (bool, optional): _description_. Defaults to False.
            min_max_tolerance (float, optional): _description_. Defaults to 2e-1.

        Returns:
            torch.Tensor: _description_
        """
                
        # The DDPM is memory intensive, accumulate gradients over minibatches
        ddpm_loss_value = 0
        num_minibatches = np.ceil(x.shape[0] / self.minibatch_size_ddpm)
        ddpm_reweigh_factor = ddpm_reweigh_factor * (1 / num_minibatches)  
        
        for i in range(0, x.shape[0], self.minibatch_size_ddpm):
            x_mb = x[i: i + self.minibatch_size_ddpm].to(self.ddpm.device)
            t_mb = t[i: i + self.minibatch_size_ddpm].to(self.ddpm.device) if t is not None else None
            noise_mb = noise[i: i + self.minibatch_size_ddpm].to(self.ddpm.device) if noise is not None else None            
            
            # Get the predicted segmentation
            x_cond_mb, _  = self.seg(self.norm(x_mb))

            # Normalize the input image between 0 and 1
            x_mb = du.normalize_min_max(
                x_mb,
                min=min_int_imgs, 
                max=max_int_imgs
            )
            
            if x_mb.max() > 1 + min_max_tolerance or x_mb.min() < 0 - min_max_tolerance:
                print(f'WARNING: x_norm_mb.max()={x_mb.max()}, x_norm_mb.min()={x_mb.min()}')
                
            assert 1 >= x_cond_mb.max() >= 0 and 1 >= x_cond_mb.min() >= 0 , \
                f'Image must be one hot encoded: currently x_norm_mb.max()={x_mb.max()}, x_norm_mb.min()={x_mb.min()}'
            
            # Rescale the input image to the same size as the DDPM, if necessary
            if x_mb.shape[-1] / self.ddpm.image_size != 1:
                rescale_factor = self.ddpm.image_size / x_mb.shape[-1]
                rescale_factor = (1, rescale_factor, rescale_factor)
                
                x_mb = F.interpolate(
                    x_mb.permute(1, 0, 2, 3).unsqueeze(0), 
                    scale_factor=rescale_factor, mode='trilinear')
                x_mb = x_mb.squeeze(0).permute(1, 0, 2, 3)
            
            if x_cond_mb.shape[-1] / self.ddpm.image_size != 1:    
                x_cond_mb = F.interpolate(
                    x_cond_mb.permute(1, 0, 2, 3).unsqueeze(0),
                    scale_factor=rescale_factor, mode='trilinear')
                x_cond_mb = x_cond_mb.squeeze(0).permute(1, 0, 2, 3)
                                            
            # Calculate the DDPM loss and backpropagate
            if self.ddpm_loss == 'jacobian':
                ddpm_loss = ddpm_reweigh_factor * self.ddpm(
                    x_mb, 
                    x_cond_mb,
                    t_mb,
                    noise=noise_mb,
                    min_t=self.min_t_diffusion_tta,
                    max_t=self.max_t_diffusion_tta,
                    w_clf_free=self.w_cfg,
                )
            
            elif self.ddpm_loss in ['sds', 'dds', 'pds']:
                ddpm_loss = ddpm_reweigh_factor * self.ddpm.distillation_loss(
                    x_mb, 
                    x_cond_mb,
                    t_mb,
                    noise=noise_mb,
                    type=self.ddpm_loss,
                    min_t=self.min_t_diffusion_tta,
                    max_t=self.max_t_diffusion_tta,
                    w_clf_free=self.w_cfg,
                )
    
            else:
                raise ValueError(f'Invalid DDPM loss: {self.ddpm_loss}, options are: jacobian, sds, dds, pds')

            ddpm_loss.backward()
            ddpm_loss_value += ddpm_loss.detach()
                                    
        return ddpm_loss_value
    
    def _sample_t_noise_pairs(
        self, 
        num_samples: int,
        dl_batch_size: int,
        num_workers: int, 
        pair_sampling_type: Literal['one_per_volume', 'one_per_image'] = 'one_per_volume',
        num_groups_stratified_sampling: int = 32,
        ) -> torch.utils.data.DataLoader:
        
        num_samples_orig = num_samples
        
        if pair_sampling_type == 'one_per_image':
            num_samples = num_samples * dl_batch_size
        
        # Sample t values
        if self.t_sampling_strategy == 'uniform':
            t_values = torch.randint(self.min_t_diffusion_tta, self.max_t_diffusion_tta, (num_samples, ))
        
        elif self.t_sampling_strategy == 'stratified':
            t_values = stratified_sampling(self.min_t_diffusion_tta, self.max_t_diffusion_tta, 
                                           num_groups_stratified_sampling, num_samples)

        else:
            raise ValueError('Invalid t_sampling_strategy')
        
        assert len(t_values) == num_samples, 'Number of samples must match the number of t values'
        assert t_values.min() >= self.min_t_diffusion_tta and t_values.max() <= self.max_t_diffusion_tta, \
            't values must be within the range of the DDPM timesteps'
        assert t_values.shape == (num_samples,), 't values must be a 1D tensor'

        # Sample noise
        noise = torch.randn((num_samples, self.ddpm.channels,
                             int(self.ddpm.image_size), int(self.ddpm.image_size)))
        
        if self.pair_sampling_type == 'one_per_volume':
            t_values = t_values.repeat_interleave(dl_batch_size)
            noise = noise.repeat_interleave(dl_batch_size, dim=0)
            num_samples = num_samples * dl_batch_size
        
        assert len(t_values) == len(noise), 'Number of samples must match the number of noise samples'
        assert len(t_values) == num_samples, 'Number of samples must match the number of noise samples'
                    
        t_dl = DataLoader(
            TensorDataset(t_values),
            batch_size=dl_batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        
        noise_dl = DataLoader(
            TensorDataset(noise),
            batch_size=dl_batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        
        assert len(t_dl) == len(noise_dl) == num_samples_orig, 'Number of samples must match the number of noise samples'
        
        return zip(t_dl, noise_dl)         
       
    def _save_checkpoint(self, logdir: str, dataset_name: str, index: int, save_ddpm: bool = False) -> None:
        os.makedirs(os.path.join(logdir, 'checkpoints'), exist_ok=True)
        
        # Save the normalizer weights in the last step
        save_checkpoint(
            path=os.path.join(logdir, 'checkpoints',
                            f'checkpoint_tta_{dataset_name}_{index:02d}_last_step.pth'),
            norm_state_dict=self.norm.state_dict(),
            seg_state_dict=self.seg.state_dict(),
            ddpm_state_dict=None if save_ddpm else self.ddpm.state_dict(), 
        )
        
    def _log_x_norm_out_gradient_magnitudes(self, index, step, log_dae_grad_norm = True, log_ddpm_grad_norm = True):
        # Get the name of the last layer
        last_layer_name = [name for name, _ in self.norm.named_parameters() if 'weight' in name][-1]
        
        log_dict = {'tta_step': step}
        
        if self.dae_loss_alpha > 0 and log_dae_grad_norm:
            log_dict[f'norm_x_norm_out_grad_dae_loss/img_{index:03d}'] = \
                torch.norm(self.x_norm_grads_dae_loss[last_layer_name])
        
        if self.ddpm_loss_beta > 0 and log_ddpm_grad_norm:
            log_dict[f'norm_x_norm_out_grad_ddpm_loss/img_{index:03d}'] = \
                torch.norm(self.x_norm_grads_ddpm_loss[last_layer_name])  
        
        wandb.log(log_dict)

        
    def _get_gradients_x_norm(self) -> dict:
        gradients_dict = defaultdict(int)
        for name, param in self.norm.named_parameters():
            if param.grad is not None:
                gradients_dict[name] = param.grad.detach().clone()  # Store gradients and detach to avoid memory leaks
                
        return gradients_dict
    
    @torch.inference_mode()
    def test_volume(
        self,
        volume_dataset: DatasetInMemory,
        dataset_name: str,
        index: int,
        num_workers: int,
        batch_size: int, 
        appendix='',
        y_dae_or_atlas: Optional[torch.Tensor] = None,  
        iteration=-1,
        device: Optional[Union[str, torch.device]] = None,
        logdir: Optional[str] = None,
        classes_of_interest=None,
    ):
        classes_of_interest = classes_of_interest or self.classes_of_interest
        
        # Get original images
        x_original, y_original, _ = volume_dataset.dataset.get_original_images(index)
        _, C, D, H, W = y_original.shape  # xyz = HWD

        x_ = x_original.permute(0, 2, 3, 1).unsqueeze(0)  # NCHWD (= NCxyz)
        y_ = y_original.permute(0, 1, 3, 4, 2)  # NCHWD

        # Rescale x and y to the target resolution of the dataset
        original_pix_size = volume_dataset.dataset.pix_size_original[:, index]
        target_pix_size = volume_dataset.dataset.resolution_proc  # xyz
        scale_factor = original_pix_size / target_pix_size
        scale_factor[-1] = 1

        y_ = y_.float()

        output_size = (y_.shape[2:] * scale_factor).round().astype(int).tolist()
        x_ = F.interpolate(x_, size=output_size, mode='trilinear')
        y_ = F.interpolate(y_, size=output_size, mode='trilinear')

        y_ = y_.round().byte()

        x_ = x_.squeeze(0).permute(3, 0, 1, 2)  # DCHW
        y_ = y_.squeeze(0).permute(3, 0, 1, 2)  # DCHW

        # Get segmentation
        volume_dataloader = DataLoader(
            TensorDataset(x_, y_),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
  
        x_norm = []
        y_pred = []
        
        for x, *_  in volume_dataloader:
            x_norm_part = self.norm(x.to(device).float()) 
            y_pred_part, _ = self.seg(x_norm_part)
            
            x_norm.append(x_norm_part.cpu())
            y_pred.append(y_pred_part.cpu())

        x_norm = torch.vstack(x_norm)
        y_pred = torch.vstack(y_pred)

        # Rescale x and y to the original resolution
        x_norm = x_norm.permute(1, 0, 2, 3).unsqueeze(0)  # convert to NCDHW (with N=1)
        y_pred = y_pred.permute(1, 0, 2, 3).unsqueeze(0)  # convert to NCDHW (with N=1)

        x_norm = F.interpolate(x_norm, size=(D, H, W), mode='trilinear')
        y_pred = F.interpolate(y_pred, size=(D, H, W), mode='trilinear')
            
        if y_dae_or_atlas is not None:
            y_dae_or_atlas = F.interpolate(y_dae_or_atlas, size=(D, H, W), mode='trilinear')
        
        export_images(
            x_original,
            x_norm,
            y_original,
            y_pred,
            y_dae=y_dae_or_atlas,
            n_classes=self.n_classes,
            output_dir=os.path.join(logdir, 'segmentations'),
            image_name=f'{dataset_name}_test_{index:03}_{iteration:03}{appendix}.png'
        )

        dices, dices_fg = dice_score(y_pred, y_original, soft=False, reduction='none', smooth=1e-5)
        print(f'Iteration {iteration} - dice score' + f': {dices_fg.mean().item()}')
            
        if self.wandb_log:
            wandb.log(
                {
                    f'dice_score_fg/img_{index:03d}': dices_fg.mean().item(),
                    'tta_step': iteration
                }
            )
            
        if classes_of_interest is not None:
            dices_classes_of_interest = dices[:, classes_of_interest, ...].nanmean().item()    
            print(f'Iteration {iteration} - dice score classes of interest {classes_of_interest}' + 
                    f' dices_classes_of_interest: {dices_classes_of_interest}')

            if self.wandb_log:
                wandb.log(
                    {
                        f'dice_score_classes_of_interest/img_{index:03d}': dices_classes_of_interest,
                        'tta_step': iteration
                    }
                )

            export_images(
                x_original,
                x_norm,
                y_original[:, [0, classes_of_interest], ...],
                y_pred[:, [0, classes_of_interest], ...],
                y_dae=y_dae_or_atlas[:, [0, classes_of_interest], ...] if y_dae_or_atlas is not None else None,
                n_classes=self.n_classes,                output_dir=os.path.join(logdir, 'segmentations_classes_of_interest'),
                image_name=f'{dataset_name}_test_{index:03}_{iteration:03}{appendix}.png'
            )
            
        return dices.cpu(), dices_fg.cpu()
    
    def reset_initial_state(self):
        self.x_norm_grads_dae_loss = defaultdict(int)          
        self.x_norm_grads_ddpm_loss = defaultdict(int)         
        
        self.norm.load_state_dict(self.norm_dict_sd)
        self.seg.load_state_dict(self.seg_dict_sd)
        self.ddpm.load_state_dict(self.ddpm_dict_sd)
        
        # Reset Optimizer
        self.optimizer_dae = torch.optim.AdamW(
            [
                {'params': self.norm.parameters(), 'lr': self.learning_rate_dae},
            ]
        )
        
        self.optimizer_ddpm = torch.optim.AdamW(
            [
                {'params': self.norm.parameters(), 'lr': self.learning_rate_ddpm},
                {'params': self.seg.parameters(), 'lr': self.learning_rate_ddpm},
                {'params': self.ddpm.parameters(), 'lr': self.learning_rate_ddpm}
            ]
        )
        
        
        # Reset flags and state info related to TTA-DAE
        self.norm_seg_dict = {
            'best_score': {
                'norm_state_dict': copy.deepcopy(self.norm.state_dict()),
                'seg_state_dict': copy.deepcopy(self.seg.state_dict())
                }
        }
        self.metrics_best['best_score'] = 0
        self.tta_losses = []
        self.test_scores = []
        
        # DAE PL states
        self.use_only_dae_pl = self.alpha == 0 and self.beta == 0
        self.using_dae_pl = False
        self.using_atlas_pl = False

    
    def _define_custom_wandb_metrics(self, ):
        wandb.define_metric(f'tta_step')
        wandb.define_metric(f'ddpm_loss/*', step_metric=f'tta_step')
        wandb.define_metric(f'dice_score_fg/*', step_metric=f'tta_step')    
        
        if self.classes_of_interest is not None:
            wandb.define_metric(f'dice_score_classes_of_interest/*', step_metric=f'tta_step')
