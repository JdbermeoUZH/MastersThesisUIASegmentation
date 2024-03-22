import os
import copy
from typing import Union, Optional, Any, Literal

import wandb
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader


from tta_uia_segmentation.src.models.normalization import background_suppression
from tta_uia_segmentation.src.utils.io import save_checkpoint, write_to_csv
from tta_uia_segmentation.src.utils.loss import dice_score, DiceLoss
from tta_uia_segmentation.src.utils.visualization import export_images
from tta_uia_segmentation.src.utils.utils import get_seed
from tta_uia_segmentation.src.dataset import DatasetInMemory



class TTADAE:
    """
    Class to perform Test-Time Adaptation (TTA) using a DAE model to generate pseudo labels.
    
    Arguments:
    norm: torch.nn.Module
        Normalization model.
    seg: torch.nn.Module
        Segmentation model.
    dae: torch.nn.Module
        DAE model.
    atlas: Any
        Atlas of the source domain segmentation labels
    loss_func: torch.nn.Module
        Loss function to be used during adaptation. Default: DiceLoss()
    learning_rate: float
        Learning rate for the optimizer. Default: 1e-3
    alpha: float
        Threshold for the proportion between the dice score of the DAE output and the atlas. Default: 1.0
        Both alpha and beta need to be satisfied to use the DAE output as pseudo label.
    beta: float
        Threshold for the dice score of the atlas. Default: 0.25. 
        Both alpha and beta need to be satisfied to use the DAE output as pseudo label.
    use_atlas_only_for_intit: bool
        Whether to use the atlas as pseudo label only until the first time the DAE output is used. Default: False
        Meaning, it is used as a switch and will only change from Atlas to DAE PL once.
    wandb_log: bool
        Whether to log the results to wandb. Default: False
    """
    
    def __init__(
        self,
        norm: torch.nn.Module,
        seg: torch.nn.Module,
        dae: torch.nn.Module, 
        atlas: Any,
        n_classes: int, 
        rescale_factor: tuple[int],
        bg_suppression_opts: dict,
        bg_suppression_opts_tta: dict,
        loss_func: torch.nn.Module = DiceLoss(),
        learning_rate: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 0.25,
        use_atlas_only_for_init: bool = False,
        seg_with_bg_supp: bool = True,
        wandb_log: bool = False,
        device: str = 'cuda',
        ) -> None:
    
        self.norm = norm
        self.seg = seg
        self.dae = dae
        self.atlas = atlas
        
        self.n_classes = n_classes
        
        # Whether the segmentation model uses background suppression of input images
        self.seg_with_bg_supp = seg_with_bg_supp
        self.bg_suppression_opts = bg_suppression_opts
        self.bg_suppression_opts_tta = bg_suppression_opts_tta
        
        # Rescale factor for pseudo labels
        self.rescale_factor = rescale_factor
        
        # Thresholds for pseudo label selection
        self.alpha = alpha
        self.beta = beta
        self.use_atlas_only_for_intit = use_atlas_only_for_init
        
        self.use_only_dae_pl = self.alpha == 0 and self.beta == 0
        
        # Loss function and optimizer
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        
        self.optimizer = torch.optim.Adam(
            self.norm.parameters(),
            lr=learning_rate
        )
        
        # Setting up metrics for model selection.
        self.tta_losses = []
        self.test_scores = []

        self.norm_dict = {'best_score': copy.deepcopy(self.norm.state_dict())}
        self.metrics_best = {'best_score': 0}
        
        # Set segmentation and DAE models in eval mode
        self.seg.eval()
        self.seg.requires_grad_(False)
        
        self.dae.eval()
        self.dae.requires_grad_(False)
        
        # DAE PL states
        self.using_dae_pl = False
        self.using_atlas_pl = False
        
        self.device = device
        
        # wandb logging
        self.wandb_log = wandb_log
        
        if self.wandb_log:
            self._define_custom_wandb_metrics()

        
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
    ):
        # Set loss and dice scores to empty lists for each new TTA
        self.tta_losses = []
        self.test_scores = []
            
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
                    num_workers=num_workers,
                    index=index,
                    iteration=step,
                    bg_suppression_opts=self.bg_suppression_opts,
                )
                self.test_scores.append(dices_fg.mean().item())

            # Update Pseudo label, with DAE or Atlas, depending on which has a better agreement
            if step % update_dae_output_every == 0:
                
                _, _, label_dataloader = self.generate_pseudo_labels(
                    dae_dataloader=dae_dataloader,
                    label_batch_size=label_batch_size,
                    device=device,
                    num_workers=num_workers,
                    dataset_repetition=dataset_repetition,
                )

            tta_loss = 0
            n_samples = 0

            self.norm.train()
            volume_dataset.dataset.set_augmentation(True)  
            
            if accumulate_over_volume:
                self.optimizer.zero_grad()

            if const_aug_per_volume:
                volume_dataset.dataset.set_seed(get_seed())
                
            # TODO: Delete these next lines, DEBUGGING
            print('DEBUG DELETE ME: len(volume_dataloader)', len(volume_dataloader))
            print('DEBUG DELETE ME: len(label_dataloader)', len(label_dataloader))
                                                        
            # Adapting to the target distribution.
            for (x,_,_,_, bg_mask), (y_pl,) in zip(volume_dataloader, label_dataloader):

                if not accumulate_over_volume:
                    self.optimizer.zero_grad()

                x = x.to(device).float()
                y_pl = y_pl.to(device)
                
                _, mask, _ = self.forward_pass_seg(
                    x, bg_mask, self.bg_suppression_opts_tta, device)

                if self.rescale_factor is not None:
                    mask = self.rescale_volume(mask)
                    
                loss = self.loss_func(mask, y_pl)

                if accumulate_over_volume:
                    loss /= len(volume_dataloader)

                loss.backward()

                if not accumulate_over_volume:
                    self.optimizer.step()

                with torch.no_grad():
                    tta_loss += loss.detach() * x.shape[0]
                    n_samples += x.shape[0]

            if accumulate_over_volume:
                self.optimizer.step()

            self.tta_losses.append((tta_loss / n_samples).item())
            
            if self.wandb_log:
                wandb.log({
                    f'tta_loss/img_{index}': self.tta_losses[-1],
                    'tta_step': step
                    })

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

    def forward_pass_seg(
        self, 
        x: Optional[torch.Tensor] = None,
        bg_mask: Optional[torch.Tensor] = None,
        bg_suppression_opts: Optional[dict] = None,
        device: Optional[Union[str, torch.device]] = None,
        x_norm: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x_norm = x_norm or self.norm(x)
        
        if self.seg_with_bg_supp:
            print('DEBUG, delete me: Using background suppression')
            bg_mask = bg_mask.to(device)
            x_norm_bg_supp = background_suppression(x_norm, bg_mask, bg_suppression_opts)
            mask, logits = self.seg(x_norm_bg_supp)
        else:
            mask, logits = self.seg(x_norm)
        
        return x_norm, mask, logits
    
    def rescale_volume(
        self,
        x: torch.Tensor,
        how: Literal['up', 'down'] = 'down',
        return_dchw: bool = True
    ) -> torch.Tensor:
        # By define the recale factor as the proportion between
        #  the Atlas or DAE volumes and the processed volumes 
        rescale_factor = list((1 / np.array(self.rescale_factor))) \
            if how == 'up' else self.rescale_factor
            
        x = x.permute(1, 0, 2, 3).unsqueeze(0)
        x = F.interpolate(x, scale_factor=rescale_factor, mode='trilinear')
        
        if return_dchw:
            x = x.squeeze(0).permute(1, 0, 2, 3)
        
        return x
    
    @torch.inference_mode()
    def test_volume(
        self,
        volume_dataset: DatasetInMemory,
        dataset_name: str,
        index: int,
        batch_size: int,
        num_workers: int,
        appendix='',
        y_dae_or_atlas: Optional[torch.Tensor] = None,
        x_guidance: Optional[torch.Tensor] = None,
        bg_suppression_opts: Optional[dict] = None,
        iteration=-1,
        device: Optional[Union[str, torch.device]] = None,
        logdir: Optional[str] = None,
    ):
        bg_suppression_opts = bg_suppression_opts or self.bg_suppression_opts

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
            x_norm_part, y_pred_part, _ = self.forward_pass_seg(
                x.to(device), bg_mask.to(device), 
                self.bg_suppression_opts, device
                )
            
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
            
        if x_guidance is not None:
            x_guidance = F.interpolate(x_guidance, size=(D, H, W), mode='trilinear')
            
        export_images(
            x_original,
            x_norm,
            y_original,
            y_pred,
            x_guidance=x_guidance,
            y_dae=y_dae_or_atlas,
            n_classes=self.n_classes,
            output_dir=os.path.join(logdir, 'segmentations'),
            image_name=f'{dataset_name}_test_{index:03}_{iteration:03}{appendix}.png'
        )

        dices, dices_fg = dice_score(y_pred, y_original, soft=False, reduction='none', epsilon=1e-5)
        print(f'Iteration {iteration} - dice score {dices_fg.mean().item()}')
        
        if self.wandb_log:
            wandb.log(
                {
                    f'dice_score_fg/img_{index}': dices_fg.mean().item(),
                    'tta_step': iteration
                }
            )

        return dices.cpu(), dices_fg.cpu()
    
    @torch.inference_mode()
    def generate_pseudo_labels(
        self,
        dae_dataloader: DataLoader,
        label_batch_size: int,
        device: Union[str, torch.device],
        num_workers: int,
        dataset_repetition: int
    ) -> tuple[float, float, DataLoader]:  
        
        masks = []
        for x, _, _, _, bg_mask in dae_dataloader:
            x = x.to(device).float()
            _, mask, _ = self.forward_pass_seg(
                x, bg_mask, self.bg_suppression_opts_tta, device)
            masks.append(mask)

        masks = torch.cat(masks)
        masks = masks.permute(1,0,2,3).unsqueeze(0) # CDHW -> DCHW

        if self.rescale_factor is not None:
            masks = F.interpolate(masks, scale_factor=self.rescale_factor, mode='trilinear')

        dae_output, _ = self.dae(masks)

        dice_denoised, _ = dice_score(masks, dae_output, soft=True, reduction='mean')
        dice_atlas, _ = dice_score(masks, self.atlas, soft=True, reduction='mean')

        print(f'DEBUG: dice_denoised: {dice_denoised}, dice_atlas: {dice_atlas}')  # TODO: Delete me
        
        if (dice_denoised / dice_atlas >= self.alpha and dice_atlas >= self.beta) \
            or self.use_only_dae_pl:
            print('Using DAE output as pseudo label')
            target_labels = dae_output
            dice = dice_denoised.item()
            self.using_dae_pl = True
            self.using_atlas_pl = False
            
            if self.use_atlas_only_for_intit and not self.use_only_dae_pl:    
                # Set alpha and beta to 0 to always use DAE output as pseudo label from now on
                print('----------------Only DAE PL from now on----------------')
                self.set_use_only_dae_pl(True)

        else:
            print('Using Atlas as pseudo label')
            target_labels = self.atlas
            dice = dice_atlas.item()
            self.using_atlas_pl = True
            self.using_dae_pl = False
            
        target_labels = target_labels.squeeze(0)
        target_labels = target_labels.permute(1,0,2,3)

        if self.metrics_best['best_score'] < dice:
            self.norm_dict['best_score'] = copy.deepcopy(self.norm.state_dict())
            self.metrics_best['best_score'] = dice
            
        pl_dataloader =  DataLoader(
            ConcatDataset([TensorDataset(target_labels.cpu())] * dataset_repetition), 
            batch_size=label_batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True, 
            )
        
        return dice_denoised.item(), dice_atlas.item(), pl_dataloader
        
    def _define_custom_wandb_metrics(self):
        wandb.define_metric("tta_step")
        wandb.define_metric('dice_score_fg/*', step_metric='tta_step')
        wandb.define_metric('tta_loss/*', step_metric='tta_step')

    def set_use_only_dae_pl(self, use_only_dae_pl: bool) -> None:
        self.use_only_dae_pl = use_only_dae_pl
        
    def reset_initial_state(self, state_dict: dict) -> None:
        self.norm.load_state_dict(state_dict)
        self.norm_dict['best_score'] = copy.deepcopy(self.norm.state_dict())
        self.metrics_best['best_score'] = 0
        self.tta_losses = []
        self.test_scores = []
                
        self.seg.eval()
        self.seg.requires_grad_(False)
        
        self.dae.eval()
        self.dae.requires_grad_(False)
        
        print('DEBUG DELETE ME: resetting optimizer state')
        self.optimizer = torch.optim.Adam(
            self.norm.parameters(),
            lr=self.learning_rate
        )
        
        # DAE PL states
        self.use_only_dae_pl = self.alpha == 0 and self.beta == 0
        self.using_dae_pl = False
        self.using_atlas_pl = False
        
    def load_state_dict_norm(self, state_dict: dict) -> None:
        self.norm.load_state_dict(state_dict)