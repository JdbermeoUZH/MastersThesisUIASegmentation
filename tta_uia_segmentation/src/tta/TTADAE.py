import os
import copy
from typing import Union, Optional, Any

import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset


from models.normalization import background_suppression
from utils.io import save_checkpoint, write_to_csv
from utils.loss import dice_score, DiceLoss
from utils.visualization import export_images
from utils.utils import get_seed


class TTADAE:
    
    def __init__(
        self,
        norm: torch.nn.Module,
        seg: torch.nn.Module,
        dae: torch.nn.Module, 
        atlas: Any,
        loss_func: torch.nn.Module = DiceLoss(),
        learning_rate: float = 1e-3
        ) -> None:
        
        self.norm = norm
        self.seg = seg
        self.dae = dae
        self.atlas = atlas
        
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        
        self.optimizer = torch.optim.Adam(
            norm.parameters(),
            lr=learning_rate
        )
        
        # Setting up metrics for model selection.
        self.tta_losses = []
        self.test_scores = []

        self.norm_dict = {'best_score': copy.deepcopy(self.norm.state_dict())}
        self.metrics_best = {'best_score': 0}
        
    def tta(
        self,
        volume_dataset: DataLoader,
        dataset_name: str,
        n_classes: int, 
        index: int,
        rescale_factor: tuple[int],
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
        logdir: Optional[str] = None,
    ):
        device = device or self.device
        logdir = logdir or self.logdir
        
        self.seg.requires_grad_(False)

        if rescale_factor is not None:
            assert (batch_size * rescale_factor[0]) % 1 == 0
            label_batch_size = int(batch_size * rescale_factor[0])
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
                    rescale_factor=rescale_factor,
                    device=device,
                    num_workers=num_workers,
                    dataset_repetition=dataset_repetition,
                    alpha=alpha,
                    beta=beta
                )

            tta_loss = 0
            n_samples = 0

            volume_dataset.dataset.set_augmentation(True)

            if accumulate_over_volume:
                self.optimizer.zero_grad()

            if const_aug_per_volume:
                volume_dataset.dataset.set_seed(get_seed())

            # Adapting to the target distribution.
            for (x,_,_,_, bg_mask), (y,) in zip(volume_dataloader, label_dataloader):

                if not accumulate_over_volume:
                    self.optimizer.zero_grad()

                x = x.to(device).float()
                y = y.to(device)
                bg_mask = bg_mask.to(device)
                x_norm = self.norm(x)
                
                x_norm = background_suppression(x_norm, bg_mask, bg_suppression_opts_tta)

                mask, logits = self.seg(x_norm)

                if rescale_factor is not None:
                    mask = mask.permute(1, 0, 2, 3).unsqueeze(0)
                    mask = F.interpolate(mask, scale_factor=rescale_factor, mode='trilinear')
                    mask = mask.squeeze(0).permute(1, 0, 2, 3)

                loss = self.loss_func(mask, y)

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


        if save_checkpoints:
            os.makedirs(os.path.join(logdir, 'checkpoints'), exist_ok=True)
            save_checkpoint(
                path=os.path.join(logdir, 'checkpoints',
                                f'checkpoint_tta_{dataset_name}_{index:02d}.pth'),
                norm_state_dict=norm_dict['best_score'],
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

        return norm, norm_dict, metrics_best

    
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

        self.norm.eval()
        volume_dataset.dataset.set_augmentation(False)

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
        with torch.no_grad():
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

        dices, dices_fg = dice_score(y_pred, y_original, soft=False, reduction='none', epsilon=1e-5)
        print(f'Iteration {iteration} - dice score {dices_fg.mean().item()}')

        self.norm.train()
        volume_dataset.dataset.set_augmentation(True)

        return dices.cpu(), dices_fg.cpu()
    
    def generate_pseudo_labels(
        self,
        dae_dataloader: DataLoader,
        label_batch_size: int,
        bg_suppression_opts_tta: dict,
        rescale_factor: tuple[int],
        device: Union[str, torch.device],
        num_workers: int,
        dataset_repetition: int,
        alpha: float = 1.0,
        beta: float = 0.25
    ) -> DataLoader:  
    
        with torch.no_grad():
            masks = []
            for x, _, _, _, bg_mask in dae_dataloader:
                x = x.to(device).float()

                bg_mask = bg_mask.to(device)
                x_norm = self.norm(x)

                x_norm = background_suppression(
                    x_norm, bg_mask, bg_suppression_opts_tta)

                mask, _ = self.seg(x_norm)
                masks.append(mask)

            masks = torch.cat(masks)
            masks = masks.permute(1,0,2,3).unsqueeze(0)

            if rescale_factor is not None:
                masks = F.interpolate(masks, scale_factor=rescale_factor, mode='trilinear')

            dae_output, _ = self.dae(masks)

            dice_denoised, _ = dice_score(masks, dae_output, soft=True, reduction='mean')
            dice_atlas, _ = dice_score(masks, self.atlas, soft=True, reduction='mean')

            if dice_denoised / dice_atlas >= alpha and dice_atlas >= beta:
                target_labels = dae_output
                dice = dice_denoised
            else:
                target_labels = self.atlas
                dice = dice_atlas

            target_labels = target_labels.squeeze(0)
            target_labels = target_labels.permute(1,0,2,3)

        if self.metrics_best['best_score'] < dice:
            self.norm_dict['best_score'] = copy.deepcopy(self.norm.state_dict())
            self.metrics_best['best_score'] = dice

        return DataLoader(
                    ConcatDataset([TensorDataset(target_labels.cpu())] * dataset_repetition),
                    batch_size=label_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    drop_last=True,
                )