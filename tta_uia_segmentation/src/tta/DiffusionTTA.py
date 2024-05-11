import os
from tqdm import tqdm
from typing import Optional, Union
from multiprocessing import cpu_count   

import wandb
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, TensorDataset

from tta_uia_segmentation.src.models import ConditionalGaussianDiffusion
from tta_uia_segmentation.src.models.normalization import background_suppression
from tta_uia_segmentation.src.utils.io import save_checkpoint, write_to_csv
from tta_uia_segmentation.src.utils.loss import dice_score
from tta_uia_segmentation.src.utils.visualization import export_images
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
        wandb_log: bool = False,
        ) -> None:
        
        self.norm = norm
        self.seg = seg
        self.ddpm = ddpm
        
        # Store the state_dict of each network
        self.norm_dict = self.norm.state_dict()
        self.seg_dict = self.seg.state_dict()
        self.ddpm_dict = self.ddpm.state_dict()
        
        self.learning_rate = learning_rate
        self.learning_rate_norm = learning_rate_norm if learning_rate_norm is not None else learning_rate
        self.learning_rate_seg = learning_rate_seg if learning_rate_seg is not None else learning_rate
        self.learning_rate_ddpm = learning_rate_ddpm if learning_rate_ddpm is not None else learning_rate
        
        self.optimizer = torch.optim.Adam(
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
        self.wandb_log = wandb_log
        
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
        self.tta_score = []
        
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
                self.tta_score.append(dices_fg.mean().item())

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
            np.array([self.tta_score]).T,
            header=['tta_score'],
            mode='w',
        )
        
        dice_scores = {i * calculate_dice_every: score for i, score in enumerate(self.tta_score)}

        return self.norm, self.norm_dict, self.metrics_best, dice_scores
    
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
                x_norm = background_suppression(x_norm, bg_mask, bg_suppression_opts_tta)
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
                
    def calculate_ddpm_gradients(
        self,
        img,
        seg,
        minibatch_size: int = 2,
        retain_graph: bool = False
        ) -> torch.Tensor:
        
        # Normalize the input image between 0 and 1, (required by the DDPM)
        img, seg = self._prepare_img_and_seg_for_ddpm(img, seg)
                
        # The DDPM is memory intensive, accumulate gradients over minibatches
        ddpm_loss_value = 0
        for i in range(0, img.shape[0], minibatch_size):
            img_batch = img[i:i+minibatch_size]
            seg_batch = seg[i:i+minibatch_size]
            ddpm_loss = self.ddpm(
                img_batch, seg_batch, 
                min_t=self.min_t_diffusion_tta, 
                max_t=self.max_t_diffusion_tta
                )
            
            retain_graph = i + minibatch_size < img.shape[0] or retain_graph
            ddpm_loss.backward(retain_graph=retain_graph)
            
            ddpm_loss_value += ddpm_loss.detach()
                        
        return ddpm_loss_value
    
    def _prepare_img_and_seg_for_ddpm(self, img, seg) -> tuple[torch.Tensor, torch.Tensor]:
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
        
        return img, seg    
    
    @torch.inference_mode()
    def test_volume(
        self,
        volume_dataset: DataLoader,
        dataset_name: str,
        index: int,
        n_classes: int,
        batch_size: int,
        num_workers: int,
        appendix='',
        bg_suppression_opts=None,
        iteration=-1,
        device: Optional[Union[str, torch.device]] = None,
        logdir: Optional[str] = None,
    ):

        # Get original images
        x_original, y_original, bg = volume_dataset.dataset.get_original_images(index)
        _, C, D, H, W = y_original.shape  # xyz = HWD

        x_ = x_original.permute(0, 2, 3, 1).unsqueeze(0)  # NCHWD (= NCxyz)
        y_ = y_original.permute(0, 1, 3, 4, 2)  # NCHWD
        bg_ = torch.from_numpy(bg).permute(1, 2, 0).unsqueeze(0).unsqueeze(0)  # NCHWD

        # Rescale x and y to the target resolution of the dataset
        original_pix_size = volume_dataset.dataset.pix_size_original[:, index]
        target_pix_size = volume_dataset.dataset.resolution_proc  # xyz
        scale_factor = original_pix_size / target_pix_size
        scale_factor[-1] = 1

        y_ = y_.float()
        bg_ = bg_.float()

        output_size = (y_.shape[2:] * scale_factor).round().astype(int).tolist()
        x_ = F.interpolate(x_, size=output_size, mode='trilinear')
        y_ = F.interpolate(y_, size=output_size, mode='trilinear')
        bg_ = F.interpolate(bg_, size=output_size, mode='trilinear')

        y_ = y_.round().byte()
        bg_ = bg_.round().bool()

        x_ = x_.squeeze(0).permute(3, 0, 1, 2)  # DCHW
        y_ = y_.squeeze(0).permute(3, 0, 1, 2)  # DCHW
        bg_ = bg_.squeeze(0).permute(3, 0, 1, 2)  # DCHW

        # Get segmentation
        volume_dataloader = DataLoader(
            TensorDataset(x_, y_, bg_),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        x_norm = []
        y_pred = []
        
        for x, _, bg_mask in volume_dataloader:
            x_norm_part = self.norm(x.to(device))
            bg_mask = bg_mask.to(device)

            x_norm_part = background_suppression(x_norm_part, bg_mask, bg_suppression_opts)

            x_norm.append(x_norm_part.cpu())

            y_pred_part, _ = self.seg(x_norm_part)
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
            n_classes=n_classes,
            output_dir=os.path.join(logdir, 'segmentations'),
            image_name=f'{dataset_name}_test_{index:03}_{iteration:03}{appendix}.png'
        )

        dices, dices_fg = dice_score(y_pred, y_original, soft=False, reduction='none', smooth=1e-5)
        print(f'Iteration {iteration} - dice score {dices_fg.mean().item()}')
        
        if self.wandb_log:
            wandb.log(
                {
                    f'dice_score_fg/img_{index}': dices_fg.mean().item(),
                    'tta_step': iteration
                }
            )

        return dices.cpu(), dices_fg.cpu()
    
    def _define_custom_wandb_metrics(self, ):
        wandb.define_metric(f'tta_step')
        wandb.define_metric(f'ddpm_loss/*', step_metric=f'tta_step')
        wandb.define_metric(f'dice_score_fg/*', step_metric=f'tta_step')    
