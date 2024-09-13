import os
from typing import Optional, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tta_uia_segmentation.src.models import DomainStatistics
from tta_uia_segmentation.src.dataset import DatasetInMemory
from tta_uia_segmentation.src.tta.BaseTTA import BaseTTA, BaseTTAState
from tta_uia_segmentation.src.utils.loss import dice_score
from tta_uia_segmentation.src.utils.utils import generate_2D_dl_for_vol
from tta_uia_segmentation.src.models.normalization import background_suppression

DICE_SMOOTHING = 1e-10

EVAL_METRICS = {
    'dice_score_all_classes': lambda y_pred, y_gt: dice_score(y_pred, y_gt, soft=False, reduction='none', smooth=DICE_SMOOTHING),
    'dice_score_fg_classes': lambda y_pred, y_gt: dice_score(y_pred, y_gt, soft=False, reduction='none', foreground_only=True, 
                                                             smooth=DICE_SMOOTHING)
}

class NoTTA(BaseTTA):
    """
    A class that implements no test-time adaptation, simply evaluating a segmentation model.

    This class extends the BaseTTA abstract class and overrides its methods to provide
    a no-adaptation implementation.

    """

    def __init__(
        self,
        norm: torch.nn.Module,
        seg: torch.nn.Module,
        n_classes: int,
        classes_of_interest: Optional[list[int]] = None,
        bg_supp_x_norm_eval: bool = False,
        bg_suppression_opts_eval: Optional[dict] = None,
        eval_metrics: Optional[dict[str, callable]] = EVAL_METRICS,
        wandb_log: bool = False,
        debug_mode: bool = False,
        device: str = 'cuda'
    ):
        super().__init__()
        
        # Models
        self._norm = norm
        self._seg = seg

        # Information about the problem
        self._n_classes = n_classes
        self._classes_of_interest = classes_of_interest

        # Background suppression settings
        self._bg_supp_x_norm_eval = bg_supp_x_norm_eval
        self._bg_suppression_opts_eval = bg_suppression_opts_eval

        # Logging and debug settings
        self._wandb_log = wandb_log
        self._debug_mode = debug_mode

        # Evaluation metrics
        self._eval_metrics = eval_metrics

        # Device
        self._device = device

        # Initialize base state of the model to keep track of test-time metrics
        self._state = BaseTTAState()

    def tta(
        self,
        dataset: DatasetInMemory,
        vol_idx: int,
        batch_size: int,
        num_workers: int,
        logdir: Optional[str] = None,
        slice_vols_for_viz: Optional[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = None
    ) -> None:
    
        # Evaluate the model on the volume
        eval_metrics = self.evaluate(
                dataset=dataset,
                vol_idx=vol_idx,
                iteration=-1,
                output_dir=logdir,
                store_visualization=True,
                save_predicted_vol_as_nifti=True,
                batch_size=batch_size,
                num_workers=num_workers,
                slice_vols_for_viz=slice_vols_for_viz
            )

        for eval_metric_name, eval_metric_values in eval_metrics.items():
            self.state.add_test_score(
                iteration=-1, metric_name=eval_metric_name, score=eval_metric_values)

        # Print mean dice score of the foreground classes
        dices_fg_mean = np.mean(eval_metrics['dice_score_fg_classes']).mean().item()
        print(f'dice score_fg_classes: {dices_fg_mean}') 

    def evaluate(
        self,
        dataset: DatasetInMemory,
        vol_idx: int,
        iteration: int,
        output_dir: str,
        store_visualization: bool = True,
        save_predicted_vol_as_nifti: bool = False,
        file_name: Optional[str] = None,
        **kwargs,
    ) -> dict:
        self._evaluation_mode()
        dataset.set_augmentation(False)

        # Get the preprocessed vol for that has the same position as
        # the original vol (preprocessed vol may have a translation in xy)
        vol_preproc, _, bg_preproc = dataset.get_preprocessed_images(
            vol_idx, same_position_as_original=True)

        vol_orig, y_gt, _ = dataset.get_original_images(vol_idx) 

        file_name = f'{dataset.dataset_name}_vol_{vol_idx:03d}_step_{iteration:03d}' \
            if file_name is None else file_name
        
        prediction_kwargs = {
            'bg_mask': bg_preproc,
            'bg_supp_x_norm': self._bg_supp_x_norm_eval,
            'bg_suppression_opts': self._bg_suppression_opts_eval,
        }
        
        eval_metrics = super().evaluate(
            x_preprocessed=vol_preproc,
            x_original=vol_orig,
            y_original_gt=y_gt.float(),
            n_classes=self._n_classes,
            preprocessed_pix_size=dataset.resolution_proc,
            gt_pix_size=dataset.get_original_pixel_size(vol_idx),
            metrics=self._eval_metrics,
            output_dir=output_dir,
            file_name=file_name,
            store_visualization=store_visualization,
            save_predicted_vol_as_nifti=save_predicted_vol_as_nifti,
            **prediction_kwargs,
            **kwargs
        )

        return eval_metrics
        
    @torch.inference_mode()
    def predict(
        self, x: torch.Tensor | DataLoader,
        bg_mask: Optional[torch.Tensor] = None,
        bg_supp_x_norm: Optional[bool] = None,
        bg_suppression_opts: Optional[dict] = None,
        output_vol_format: Literal['DCHW', '1CDHW'] = '1CDHW',
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        **kwargs
        ) -> torch.Tensor:
        """Predict the segmentation mask for a given volume.

        Parameters
        ----------
        x : torch.Tensor or DataLoader
            Volume to be segmented (in 1CDHW format) or a DataLoader providing batches.
            If a DataLoader is passed, it is assumed to provide the bg_mask as well.
        bg_mask : torch.Tensor, optional
            Background mask of the volume. Required if x is a Tensor.
        bg_supp_x_norm : bool, optional
            Whether to apply background suppression on normalized input.
        bg_suppression_opts : dict, optional
            Options for background suppression.
        manually_norm_img_before_seg : bool, optional
            Whether to manually normalize the image before segmentation.
        device : str, optional
            Device to be used for computation.
        **kwargs : dict
            Additional arguments to be passed to generate_2D_dl_for_vol if x is a Tensor.

        Returns
        -------
        torch.Tensor
            Predicted segmentation mask.

        Notes
        -----
        If x is a Tensor, it will be converted to a DataLoader of 2D slices internally.
        """
        self._evaluation_mode()
        
        bg_supp_x_norm = bg_supp_x_norm if bg_supp_x_norm is not None \
            else self._bg_supp_x_norm_eval
        bg_suppression_opts = bg_suppression_opts if bg_suppression_opts is not None \
            else self._bg_suppression_opts_eval

        x_norms, masks, logits_list = [], [], []
        
        if isinstance(x, DataLoader):
            for x_b, _, _, _, bg_mask_b in x:
                x_b = x_b.to(self._device).float()
                x_norm, mask, logits = self.forward_pass_seg(
                    x_b, bg_mask_b, bg_supp_x_norm, bg_suppression_opts,
                    **kwargs)
                x_norms.append(x_norm)
                masks.append(mask)
                logits_list.append(logits)

        elif isinstance(x, torch.Tensor):
            n_dims = 3 if x.dim() == 5 else 2
            if bg_mask is not None:
                assert x.shape[-n_dims:] == bg_mask.shape[-n_dims:], 'x and bg_mask must have the same spatial dims'

            vols = (x, bg_mask) if bg_mask is not None else (x,)

            dl = generate_2D_dl_for_vol(*vols, batch_size=batch_size, num_workers=num_workers)
            
            for x_b, bg_mask_b in dl:
                x_b = x_b.to(self._device).float()
                x_norm, mask, logits = self.forward_pass_seg(
                    x_b, bg_mask_b, bg_supp_x_norm, 
                    bg_suppression_opts, **kwargs)
                x_norms.append(x_norm)
                masks.append(mask)
                logits_list.append(logits)

        # Concatenate and permute dimensions for all tensors
        x_norms, masks, logits = [torch.cat(t) for t in [x_norms, masks, logits_list]]
        
        if output_vol_format == 'DCHW':
            return x_norms, masks, logits
        elif output_vol_format == '1CDHW':
            return [t.permute(1, 0, 2, 3).unsqueeze(0) for t in [x_norms, masks, logits]]
        else:
            raise ValueError(f"Unsupported return format: {output_vol_format}")


    def forward_pass_seg(
        self, 
        x: Optional[torch.Tensor] = None,
        bg_mask: Optional[torch.Tensor] = None,
        bg_supp_x_norm: bool = False,
        bg_suppression_opts: Optional[dict] = None,
        x_norm: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x_norm = x_norm if x_norm is not None else self._norm(x)
        
        if bg_supp_x_norm:
            bg_mask = bg_mask.to(self._device)
            x_norm_bg_supp = background_suppression(x_norm, bg_mask, bg_suppression_opts)
            mask, logits = self._seg(x_norm_bg_supp)
        else:
            mask, logits = self._seg(x_norm)
        
        return x_norm, mask, logits

    def load_state(self, path: str) -> None:
        """
        Load the state of the model from a file.

        Parameters
        ----------
        path : str
            Path to the file containing the model state.
        """
        state_dict = torch.load(path)
        self._norm.load_state_dict(state_dict['norm'])
        self._seg.load_state_dict(state_dict['seg'])

    def save_state(self, path: str) -> None:
        """
        Save the state of the model to a file.

        Parameters
        ----------
        path : str
            Path to the file where the model state will be saved.
        """
        state_dict = {
            'norm': self._norm.state_dict(),
            'seg': self._seg.state_dict()
        }
        torch.save(state_dict, path)

    def reset_state(self) -> None:
        raise NotImplementedError("NoTTA is stateless and does not require resetting state.")

    def load_best_state(self) -> None:
        raise NotImplementedError("NoTTA is stateless and does not require loading a best state.")

    def _evaluation_mode(self) -> None:
        """
        Set the model to evaluation mode.
        """
        self._norm.eval()
        self._seg.eval()

    @property
    def state(self) -> BaseTTAState:
        return self._state
