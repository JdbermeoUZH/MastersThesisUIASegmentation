import os
import copy
from tqdm import tqdm
from typing import Optional, Union, Literal   

import wandb
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from tta_uia_segmentation.src.models import BaseConditionalGaussianDiffusion
from tta_uia_segmentation.src.utils.io import save_checkpoint, write_to_csv
from tta_uia_segmentation.src.utils.loss import dice_score
from tta_uia_segmentation.src.utils.visualization import export_images
from tta_uia_segmentation.src.utils.utils import get_seed, stratified_sampling
from tta_uia_segmentation.src.dataset import DatasetInMemory, utils as du


class DiffusionTTA:
    
    def __init__(
        self,
        norm: torch.nn.Module,
        seg: torch.nn.Module,
        ddpm: BaseConditionalGaussianDiffusion,
        learning_rate: float,
        n_classes: int,
        classes_of_interest: Optional[list[int]] = None,
        learning_rate_norm: Optional[float] = None, 
        learning_rate_seg: Optional[float] = None,
        learning_rate_ddpm: Optional[float] = None,
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
        ) -> None:
        
        self.norm = norm
        self.seg = seg
        self.ddpm = ddpm
        
        # Save the initial state of the networks
        self.norm_dict_sd = copy.deepcopy(self.norm.state_dict())
        self.seg_dict_sd = copy.deepcopy(self.seg.state_dict())
        self.ddpm_dict_sd = copy.deepcopy(self.ddpm.state_dict())
        
        # Define learning rates and optimizer     
        self.learning_rate = learning_rate
        self.learning_rate_norm = learning_rate_norm if learning_rate_norm is not None else learning_rate
        self.learning_rate_seg = learning_rate_seg if learning_rate_seg is not None else learning_rate
        self.learning_rate_ddpm = learning_rate_ddpm if learning_rate_ddpm is not None else learning_rate
        
        self.optimizer = torch.optim.AdamW(
            [
                {'params': self.norm.parameters(), 'lr': self.learning_rate_norm},
                {'params': self.seg.parameters(), 'lr': self.learning_rate_seg},
                {'params': self.ddpm.parameters(), 'lr': self.learning_rate_ddpm}
            ]
        )
        
        # Setting up metrics for model selection.
        self.tta_losses = []
        self.tta_score = []

        # DDPM loss parameters  
        self.ddpm_loss = ddpm_loss
        self.w_cfg = w_cfg
        
        self.pair_sampling_type = pair_sampling_type 
        self.t_sampling_strategy = t_sampling_strategy
        self.min_t_diffusion_tta = int(np.ceil(t_ddpm_range[0] * 
                                               (self.ddpm.num_train_timesteps - 1)))
        self.max_t_diffusion_tta = int(np.floor(t_ddpm_range[1] * 
                                                (self.ddpm.num_train_timesteps - 1)))
        
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
        num_t_noise_pairs_per_img: int,
        batch_size: int,
        num_workers: int,
        calculate_dice_every: int,
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
        
        # Define the sampler object for the volume dataset
        volume_dataloader = DataLoader(
            ConcatDataset([volume_dataset] * dataset_repetition),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        
        for step in range(num_steps):
            print(f'Step: {step}')
            # Measure segmentation performance during adaptation
            # :===========================================: 
            self.norm.eval()
            #self.seg.eval()
            volume_dataset.dataset.set_augmentation(False)
            
            if step % calculate_dice_every == 0 and calculate_dice_every != -1:

                _, dices_fg = self.test_volume(
                    volume_dataset=volume_dataset,
                    dataset_name=dataset_name,
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


            # Adapting based on image likelihood
            # :===========================================:
            
            # Sample t and noise for the DDPM
            t_noise_dl = self._sample_t_noise_pairs(
                num_samples=num_t_noise_pairs_per_img,
                dl_batch_size=batch_size,
                num_workers=num_workers,
                num_imgs_per_volume=len(volume_dataset) * dataset_repetition
            )
            
            # Fit the parameters of the networks
            ddpm_loss = 0
            n_samples = 0
            
            self.optimizer.zero_grad()

            if const_aug_per_volume:
                volume_dataset.dataset.set_seed(get_seed())
                            
            for ((t,), (noise,)) in tqdm(t_noise_dl, total=num_t_noise_pairs_per_img):
                for x, *_ in volume_dataloader:
                    ddpm_reweigh_factor = 1 / ( num_t_noise_pairs_per_img * len(volume_dataloader) )
                    
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
                                                    
                    with torch.no_grad():
                        ddpm_loss += ddpm_loss.detach() * x.shape[0]
                        n_samples += x.shape[0]
                        
            self.optimizer.step()

            ddpm_loss = (ddpm_loss / n_samples).item()
            
            self.tta_losses.append(ddpm_loss)

            if self.wandb_log:
                wandb.log({
                    f'ddpm_loss/img_{index}': ddpm_loss, 
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
    
    '''
    def tta_original_algorithm(
        self,
        volume_dataset: DatasetInMemory,
        dataset_name: str,
        n_classes: int, 
        index: int,
        bg_suppression_opts: dict,
        bg_suppression_opts_tta: dict,
        num_steps: int = 5,
        batch_size: int = 8,
        batch_size_ddpm: int = 180,
        minibatch_size_ddpm: int = 2,
        num_workers: int = cpu_count(),
        device: str = 'cuda',
        logdir: Optional[str] = None,     
    ):  
        """
        TTA algorithm as described in the original paper.
        
        See https://openreview.net/pdf?id=gUTVpByfVX
        
        The algorithm is applied on each image individually.
        
        Each step of the TTA algorithm consists of the following steps:
        1. Compute current segmentation label estimate
        2. Sample `batch_size` (noise, timestep) tuples and calculate the gradients of the DDPM
        3. Calculate and accumulate the reverse process gradients for the sampled tuples
        4. Update the parameters of the Segmentation network and the DDPM   
         
        """ 
        self.tta_score = []
        
        self.norm.train() if self.fit_norm_params else self.norm.eval()
        self.seg.train() if self.fit_seg_params else self.seg.eval()
        self.ddpm.train() if self.fit_ddpm_params else self.ddpm.eval()
        
        # Load the sample of cuts on which to perform TTA
        volume_dataloader = DataLoader(
            volume_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        
        vol_estimates = {k: {'x_norm':[], 'y_pred': []} for k in range(num_steps)}
        
        for x, _,_,_, bg_mask in tqdm(volume_dataloader):
            x = x.to(device).float()      
            bg_mask = bg_mask.to(device)
            
            # Reset the state of the networks
            self.norm.load_state_dict(self.norm_dict)
            self.seg.load_state_dict(self.seg_dict)
            self.ddpm.load_state_dict(self.ddpm_dict)
            
            for step in range(num_steps):
                print('Step:', step)
                self.optimizer.zero_grad()
    
                # Get the predicted segmentation
                x_norm = self.norm(x)
                y_pred, _  = self.seg(x_norm)
                
                # Accumulate gradients over the batch_size
                for i in tqdm(range(batch_size_ddpm)):
                    not_last_batch = i < batch_size_ddpm - 1
                    self.calculate_ddpm_gradients(
                        x.clone(), 
                        y_pred,
                        minibatch_size=minibatch_size_ddpm,
                        retain_graph=not_last_batch
                    )
                    
                self.optimizer.step()
                
                # Get the current estimate for the image
                with torch.no_grad():
                    x_norm = self.norm(x)
                    x_norm = background_suppression(x_norm, bg_mask, bg_suppression_opts)
                    y_pred, _  = self.seg(x_norm)
                    vol_estimates[step]['x_norm'].append(x_norm.cpu())
                    vol_estimates[step]['y_pred'].append(y_pred.cpu())
        
        # Measure segmentation performance during adaptation
        x_original, y_original, _ = volume_dataset.dataset.get_original_images(index)
        _, _, D, H, W = y_original.shape  # xyz = HWD   
        
        for step, pred_list in vol_estimates.items():
            x_norm = torch.vstack(pred_list['x_norm'])
            y_pred = torch.vstack(pred_list['y_pred'])
            
            # Rescale x and y to the original resolution
            x_norm = x_norm.permute(1, 0, 2, 3).unsqueeze(0)  # convert to NCDHW (with N=1)
            y_pred = y_pred.permute(1, 0, 2, 3).unsqueeze(0)  # convert to NCDHW (with N=1)

            x_norm = F.interpolate(x_norm, size=(D, H, W), mode='trilinear')
            y_pred = F.interpolate(y_pred, size=(D, H, W), mode='trilinear')
            
            export_images(
                x_original,
                x_norm,
                y_original,
                y_pred,
                n_classes=n_classes,
                output_dir=os.path.join(logdir, 'segmentations'),
                image_name=f'{dataset_name}_test_{index:03}_{step:03}.png'
            )
            
            _, dices_fg = dice_score(y_pred, y_original, soft=False, reduction='none', smooth=1e-5)
            print(f'Step {step} - dice score {dices_fg.mean().item()}')
            self.tta_score.append(dices_fg.mean().item())

        write_to_csv(
            os.path.join(logdir, 'tta_score', f'{dataset_name}_{index:03d}.csv'),
            np.array([self.tta_score]).T,
            header=['tta_score'],
            mode='w',
        )
    '''    
    
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
            ddpm_loss = ddpm_reweigh_factor * self.ddpm.tta_loss(
                x_mb, 
                x_cond_mb,
                t_mb,
                noise=noise_mb,
                min_t=self.min_t_diffusion_tta,
                max_t=self.max_t_diffusion_tta,
                w_clf_free=self.w_cfg,
            )
            
            ddpm_loss.backward()
            ddpm_loss_value += ddpm_loss.detach()
                                    
        return ddpm_loss_value
    
    def _sample_t_noise_pairs(
        self, num_samples: int,
        dl_batch_size: int, num_workers: int, 
        num_imgs_per_volume: int,
        num_groups_stratified_sampling: int = 32,
         
        ) -> torch.utils.data.DataLoader:
        
        num_samples_orig = num_samples
        
        if self.pair_sampling_type == 'one_per_image':
            num_samples = num_samples * num_imgs_per_volume
        
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
            t_values = t_values.repeat_interleave(num_imgs_per_volume)
            noise = noise.repeat_interleave(num_imgs_per_volume, dim=0)
        
        assert len(t_values) == len(noise), 'Number of samples must match the number of noise samples'
        assert len(t_values) == num_samples_orig * num_imgs_per_volume, 'Number of samples must match the number of noise samples'
                    
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
    
    @torch.inference_mode()
    def test_volume(
        self,
        volume_dataset: DatasetInMemory,
        dataset_name: str,
        index: int,
        num_workers: int,
        batch_size: int, 
        appendix='',
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
            
        export_images(
            x_original,
            x_norm,
            y_original,
            y_pred,
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
                n_classes=self.n_classes,
                output_dir=os.path.join(logdir, 'segmentations_classes_of_interest'),
                image_name=f'{dataset_name}_test_{index:03}_{iteration:03}{appendix}.png'
            )
            
        return dices.cpu(), dices_fg.cpu()
    
    def reset_initial_state(self):
        self.norm.load_state_dict(self.norm_dict_sd)
        self.seg.load_state_dict(self.seg_dict_sd)
        self.ddpm.load_state_dict(self.ddpm_dict_sd)
    
    def _define_custom_wandb_metrics(self, ):
        wandb.define_metric(f'tta_step')
        wandb.define_metric(f'ddpm_loss/*', step_metric=f'tta_step')
        wandb.define_metric(f'dice_score_fg/*', step_metric=f'tta_step')    
        wandb.define_metric(f'dice_score_classes_of_interest/*', step_metric=f'tta_step')
