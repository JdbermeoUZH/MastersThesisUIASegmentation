import os
import copy
from tqdm import tqdm
from dataclasses import asdict, dataclass, field
from typing import OrderedDict, Optional, Any, Literal

import wandb
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tta_uia_segmentation.src.utils.io import save_checkpoint
from tta_uia_segmentation.src.utils.loss import dice_score, DiceLoss, onehot_to_class
from tta_uia_segmentation.src.utils.utils import (
    get_seed, clone_state_dict_to_cpu, generate_2D_dl_for_vol, resize_volume)
from tta_uia_segmentation.src.dataset import DatasetInMemory
from tta_uia_segmentation.src.dataset.utils import normalize
from tta_uia_segmentation.src.models import DomainStatistics
from tta_uia_segmentation.src.tta import BaseTTAState, BaseTTA, NoTTA


@dataclass
class TTADAEState(BaseTTAState):
    """
    Dataclass to store the state of TTADAE.

    Attributes
    ----------
    using_dae_pl : bool
        Flag indicating whether DAE pseudo-labels are being used.
    using_atlas_pl : bool
        Flag indicating whether atlas pseudo-labels are being used.
    use_only_dae_pl : bool
        Flag indicating whether to use only DAE pseudo-labels.
    use_only_atlas : bool
        Flag indicating whether to use only atlas pseudo-labels.
    use_atlas_only_for_init : bool
        Flag indicating whether to use atlas only for initialization.
    y_pl: torch.Tensor
        Pseudo-labels/Prior, it may be the atlas or a denoised prior with the DAE.
    norm_state_dict : dict[str, Any]
        Dictionary storing the state of the normalization model.
    norm_td_statistics : Optional[DomainStatistics]
        Statistics of the normalized target domain.
    """
    using_dae_pl: bool = False
    using_atlas_pl: bool = False
    use_only_dae_pl: bool = False
    use_only_atlas: bool = False
    use_atlas_only_for_init: bool = False
    y_pl: torch.Tensor = field(default=None)
    
    norm_state_dict: dict[str, Any] = field(default_factory=dict)
    norm_td_statistics: Optional[DomainStatistics | dict] = None
        
    _create_initial_state: bool = field(default=True)
    _initial_state: Optional['TTADAEState'] = field(default=None)
    _best_state: Optional['TTADAEState'] = field(default=None)

    def __post_init__(self):
        """
        Ensure consistency between flags and initialize the state.
        """
        self._ensure_flag_consistency()
        self._initialize_atlas_flag()
        super().__post_init__()

        if isinstance(self.norm_td_statistics, dict):
            self.norm_td_statistics = DomainStatistics(**self.norm_td_statistics)

    def _ensure_flag_consistency(self):
        """
        Ensure consistency between flags for pseudo-label/prior.
        """
        assert not (self.use_only_dae_pl and self.use_only_atlas), \
            "Cannot use only DAE and only Atlas simultaneously"
        assert not (self.using_dae_pl and self.using_atlas_pl), \
            "Cannot use DAE and Atlas simultaneously"
        assert not self.use_atlas_only_for_init or (not self.use_only_atlas and not self.use_only_dae_pl), \
            "Cannot use only Atlas or only DAE and use Atlas only for initialization"
        
        if self.use_only_dae_pl:
            self.using_dae_pl = True
            self.using_atlas_pl = False
            
        elif self.use_only_atlas or self.use_atlas_only_for_init:
            self.using_atlas_pl = True
            self.using_dae_pl = False

    def _initialize_atlas_flag(self):
        """
        Initialize _atlas_init_done flag.
        """
        self.atlas_init_done = False if self.use_atlas_only_for_init else None

    def use_dae_pl(self, y_dae: torch.Tensor):
        """
        Switch to using DAE pseudo-labels.

        This method sets the flags to use DAE pseudo-labels and not use Atlas pseudo-labels.
        """
        
        assert not self.use_only_atlas, ("Cannot use only DAE pseudo-labels when only" + 
                                         " Atlas pseudo-labels are enabled")
        
        self.using_dae_pl = True
        self.using_atlas_pl = False

        if self.use_atlas_only_for_init and not self.atlas_init_done:
            self.atlas_init_done = True

        self.y_pl = y_dae
        
    def use_atlas_pl(self, y_atlas: torch.Tensor):
        """
        Switch to using Atlas pseudo-labels.

        This method sets the flags to use Atlas pseudo-labels and not use DAE pseudo-labels.
        """
        assert not self.use_only_dae_pl, "Cannot use only Atlas pseudo-labels with only DAE pseudo-labels"
        
        self.using_atlas_pl = True
        self.using_dae_pl = False

        if self.use_atlas_only_for_init and self.atlas_init_done:
            raise RuntimeError("Cannot switch to Atlas pseudo-labels after initialization" +
                               " when use_atlas_only_for_intit is enabled")
    
        self.y_pl = y_atlas

    @property
    def current_state(self) -> 'TTADAEState':
        """
        Get a deep copy of the current state of the TTADAEState.

        state_dict's that correspond to model states are moved to the CPU.

        Returns:
            TTADAEState: The current state.
        """

        # Remove the initial and best states from the state_dict
        current_state_dict = asdict(self)
        current_state_dict['_create_initial_state'] = False
        
        # Move the state_dict of the model to CPU
        model_state_dicts = ['norm_state_dict']
        current_state_dict['norm_state_dict'] = clone_state_dict_to_cpu(
            current_state_dict['norm_state_dict'])  

        # Create a deep copy of all other attributes
        current_state_dict = {
            key: copy.deepcopy(value) if key not in model_state_dicts else value
            for key, value in current_state_dict.items()
            }

        # Create the new TTADAEState instance
        current_state_dict = TTADAEState(**current_state_dict)

        return current_state_dict

    @property
    def y_pl_categorical(self) -> torch.Tensor:
        """
        Get the pseudo-labels in categorical format.

        Returns:
            torch.Tensor: Pseudo-labels in categorical format. N1DHW format.
        """
        return onehot_to_class(self.y_pl) if self.y_pl is not None else None


class TTADAE(NoTTA):
    """
    Class to perform Test-Time Adaptation (TTA) using a DAE model to generate pseudo labels.

    Parameters
    ----------
    norm : torch.nn.Module
        Normalization model.
    seg : torch.nn.Module
        Segmentation model.
    dae : torch.nn.Module
        DAE model.
    atlas : Any
        Atlas of the source domain segmentation labels.
    n_classes : int
        Number of classes in the segmentation task.
    rescale_factor : tuple[int]
        Factor to rescale the pseudo labels.
    loss_func : torch.nn.Module, optional
        Loss function to be used during adaptation, by default DiceLoss().
    learning_rate : float, optional
        Learning rate for the optimizer, by default 1e-3.
    alpha : float, optional
        Threshold for the proportion between the dice score of the DAE output and the atlas, by default 1.0.
        Both alpha and beta need to be satisfied to use the DAE output as pseudo label.
    beta : float, optional
        Threshold for the dice score of the atlas, by default 0.25.
        Both alpha and beta need to be satisfied to use the DAE output as pseudo label.
    eval_metrics : OrderedDict[str, callable], optional
        Dictionary of evaluation metrics, by default EVAL_METRICS.
    classes_of_interest : Optional[list[int]], optional
        List of classes to focus on during evaluation, by default None.
    use_only_dae_pl : bool, optional
        Whether to use only DAE pseudo labels, by default False.
    use_only_atlas : bool, optional
        Whether to use only atlas labels, by default False.
    use_atlas_only_for_intit : bool, optional
        Whether to use the atlas as pseudo label only until the first time the DAE output is used, by default False.
        It acts as a switch and will only change from Atlas to DAE PL once.
    norm_sd_statistics : Optional[DomainStatistics], optional
        Statistics of the source domain for normalization, by default None.
    update_norm_td_statistics : bool, optional
        Whether to update target domain statistics during normalization, by default False.
    manually_norm_img_before_seg_tta : bool, optional
        Whether to manually normalize images before segmentation during TTA, by default False.
    manually_norm_img_before_seg_eval : bool, optional
        Whether to manually normalize images before segmentation during evaluation, by default False.
    normalization_strategy : Literal['standardize', 'min_max', 'histogram_eq'], optional
        Strategy for normalization, by default 'standardize'.
    bg_supp_x_norm_eval : bool, optional
        Whether to suppress background during normalization in evaluation, by default False.
    bg_suppression_opts_eval : Optional[dict], optional
        Options for background suppression during evaluation, by default None.
    bg_supp_x_norm_tta_dae : bool, optional
        Whether to suppress background during normalization in TTA DAE, by default False.
    bg_suppression_opts_tta_dae : Optional[dict], optional
        Options for background suppression during TTA DAE, by default None.
    wandb_log : bool, optional
        Whether to log the results to wandb, by default False.
    debug_mode : bool, optional
        Whether to run in debug mode, by default False.
    device : str, optional
        Device to run the models on, by default 'cuda'.
    optimizer : Optional[torch.optim.Optimizer], optional
        Optimizer for the adaptation process, by default None.
    """
    
    def __init__(
        self,
        norm: torch.nn.Module,
        seg: torch.nn.Module,
        dae: torch.nn.Module, 
        atlas: Any,
        n_classes: int, 
        rescale_factor: tuple[int],
        loss_func: torch.nn.Module = DiceLoss(),
        learning_rate: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 0.25,
        eval_metrics: OrderedDict[str, callable] = NoTTA.EVAL_METRICS,
        classes_of_interest: Optional[list[int]] = None,
        use_only_dae_pl: bool = False,
        use_only_atlas: bool = False,
        use_atlas_only_for_intit: bool = False,
        max_grad_norm: Optional[float] = None,
        norm_sd_statistics: Optional[DomainStatistics] = None,
        update_norm_td_statistics: bool = False,
        manually_norm_img_before_seg_tta: bool = False,
        manually_norm_img_before_seg_eval: bool = False,
        normalization_strategy: Literal['standardize', 'min_max', 'histogram_eq'] = 'standardize',
        bg_supp_x_norm_eval: bool = False,
        bg_suppression_opts_eval: Optional[dict] = None,
        bg_supp_x_norm_tta_dae: bool = False,
        bg_suppression_opts_tta_dae: Optional[dict] = None,
        wandb_log: bool = False,
        debug_mode: bool = False,
        device: str = 'cuda',
        optimizer: Optional[torch.optim.Optimizer] = None,
        ) -> None:
        
        super().__init__(
            norm=norm,
            seg=seg,
            n_classes=n_classes,
            classes_of_interest=classes_of_interest,
            bg_supp_x_norm_eval=bg_supp_x_norm_eval,
            bg_suppression_opts_eval=bg_suppression_opts_eval,
            eval_metrics=eval_metrics,
            wandb_log=wandb_log,
            debug_mode=debug_mode,
            device=device,
        )

        # Models and objects used in TTA
        self._dae = dae
        self._atlas = atlas
        
        # Rescale factor for pseudo labels
        self._rescale_factor = rescale_factor
        
        # Thresholds for pseudo label selection
        self._alpha = alpha
        self._beta = beta      
          
        # Loss function and optimizer
        self._loss_func = loss_func
        self._learning_rate = learning_rate
        
        if optimizer is None:
            self._optimizer = torch.optim.Adam(
                self._norm.parameters(),
                lr=learning_rate
            )

        self._max_grad_norm = max_grad_norm
        
        # Whether the segmentation model uses background suppression of input images
        self._bg_supp_x_norm_tta_dae = bg_supp_x_norm_tta_dae
        self._bg_suppression_opts_tta = bg_suppression_opts_tta_dae

        # Set segmentation and DAE models in eval mode
        self._tta_fit_mode()
                
        # Wandb logging
        if self._wandb_log:
            TTADAE._define_custom_wandb_metrics(self)

        # Handling of target domain statistics
        self._update_norm_td_statistics = update_norm_td_statistics        
        self._norm_sd_statistics = norm_sd_statistics
    
        self._manually_norm_img_before_seg_tta = manually_norm_img_before_seg_tta
        self._manually_norm_img_before_seg_eval = manually_norm_img_before_seg_eval
        self._normalization_strategy = normalization_strategy

        if norm_sd_statistics is not None:
            norm_td_statistics = DomainStatistics(**asdict(norm_sd_statistics))
            norm_td_statistics.frozen = not update_norm_td_statistics
            norm_td_statistics.quantile_cal = None
            norm_td_statistics.precalculated_quantiles = None
        else:
            norm_td_statistics = None
            self._update_norm_td_statistics = False

        # Initialize the state of the class
        self._state = TTADAEState(
            using_dae_pl=False,
            using_atlas_pl=False,
            use_only_dae_pl=(self._alpha == 0 and self._beta == 0) or use_only_dae_pl,
            use_only_atlas=use_only_atlas,
            use_atlas_only_for_init=use_atlas_only_for_intit,
            norm_state_dict=norm.state_dict(),
            norm_td_statistics=norm_td_statistics
        )

        # Set the object in tta_fit_mode 
        self._tta_fit_mode()
    
    def _evaluation_mode(self) -> None:
        """
        Set the models to evaluation mode.

        """
        self._norm.eval()
        self._seg.eval()
        self._dae.eval()

    def _tta_fit_mode(self) -> None:
        """
        Set the objects used for TTA into training or frozen and evaluation mode.
        """
        self._norm.train()

        self._seg.eval()
        self._seg.requires_grad_(False)

        self._dae.eval()
        self._dae.requires_grad_(False)

    def tta(
        self,
        dataset: DatasetInMemory,
        vol_idx: int,
        num_steps: int,
        batch_size: int,
        num_workers: int,
        calculate_dice_every: int,
        update_dae_output_every: int,
        accumulate_over_volume: bool,
        const_aug_per_volume: bool,
        save_checkpoints: bool,
        logdir: Optional[str] = None,
        slice_vols_for_viz: Optional[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = None,
        repeat_dataset: int = 1,
        drop_last_batch_dl: bool = True
    ) -> None:

        # Change batch size for the pseudo label (in case it has a smaller size)
        if self._rescale_factor is not None:
            assert (batch_size * self._rescale_factor[0]) % 1 == 0
            label_batch_size = int(batch_size * self._rescale_factor[0])
        else:
            label_batch_size = batch_size

        # Create dataloader for the volume on which we wish to adapt
        #  We might want to repeat the dataset in 
        vol_dataset = Subset(dataset, dataset.get_idxs_for_volume(vol_idx))
        vol_dataset = ConcatDataset([vol_dataset] * repeat_dataset)
        volume_dataloader = DataLoader(
            vol_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last_batch_dl,
        ) 
        
        # Start TTA iterations 
        for step in tqdm(range(num_steps)):
            self._state.iteration = step

            # Update Pseudo label, with DAE or Atlas, depending on which has a better agreement
            if step % update_dae_output_every == 0:
                
                y_pl = self.generate_pseudo_labels(
                    dae_dataloader=volume_dataloader
                )

                pl_dataloader = generate_2D_dl_for_vol(
                    y_pl,
                    repeat_dataset=repeat_dataset,
                    batch_size=label_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    drop_last=drop_last_batch_dl, 
                )

            # Test performance during adaptation.
            if (step % calculate_dice_every == 0 or step == num_steps - 1) and calculate_dice_every != -1:
                _ = self.evaluate(
                    dataset=dataset,
                    vol_idx=vol_idx,
                    iteration=step,
                    output_dir=logdir,
                    store_visualization=True,
                    save_predicted_vol_as_nifti=False,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    slice_vols_for_viz=slice_vols_for_viz
                )

            # Take optimization TTA step
            self._tta_fit_mode()
            dataset.set_augmentation(True)  
            breakpoint()
            print('Make sure this parameter changes for the dataset in vol_dataloader')
            
            tta_loss = 0
            n_samples = 0
            
            if accumulate_over_volume:
                self._optimizer.zero_grad()

            if const_aug_per_volume:
                dataset.set_seed(get_seed())
                                                            
            for (x,_,_,_, bg_mask), (y_pl,) in zip(volume_dataloader, pl_dataloader):

                if not accumulate_over_volume:
                    self._optimizer.zero_grad()

                x = x.to(self._device).float()
                y_pl = y_pl.to(self._device)
                
                # Update the statistics of the normalized target domain in the current step
                if self._update_norm_td_statistics:
                    with torch.no_grad():
                        self._state.norm_td_statistics.update_step_statistics(self._norm(x))
                
                _, mask, _ = self.forward_pass_seg(
                    x, bg_mask, self._bg_supp_x_norm_tta_dae, self._bg_suppression_opts_tta,
                    manually_norm_img_before_seg=self._manually_norm_img_before_seg_tta)

                if self._rescale_factor is not None:
                    # Downsample the mask to the size of the pseudo label
                    mask = resize_volume(
                        mask,
                        target_pix_size=[1., 1., 1.],
                        current_pix_size=self._rescale_factor,
                        only_inplane_resample=False,
                        vol_format='DCHW',
                        output_format='DCHW')
                    
                loss = self._loss_func(mask, y_pl)

                if accumulate_over_volume:
                    loss /= len(volume_dataloader)
                
                loss.backward()

                if not accumulate_over_volume:
                    if self._max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self._norm.parameters(), self._max_grad_norm)
                    self._optimizer.step()

                with torch.no_grad():
                    tta_loss += loss.detach() * x.shape[0]
                    n_samples += x.shape[0]

            if accumulate_over_volume:
                if self._max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self._norm.parameters(), self._max_grad_norm)
                self._optimizer.step()
            
            if self._update_norm_td_statistics:
                self._state.norm_td_statistics.update_statistics()

            tta_loss = (tta_loss / n_samples).item()

            self._state.add_test_loss(
                iteration=step, loss_name='tta_loss', loss_value=tta_loss)
            
            if self._wandb_log:
                wandb.log({f'tta_loss/img_{vol_idx:02d}': tta_loss, 'tta_step': step})

        if save_checkpoints:
            self._save_checkpoint(logdir, dataset.dataset_name, vol_idx)

        self._state.is_adapted = True

        test_scores_dir = os.path.join(logdir, 'tta_score')
        
        self._state.store_test_scores_in_dir(
            output_dir=test_scores_dir,
            file_name_prefix=f'{dataset.dataset_name}_{vol_idx:03d}',
            reduce='mean_accross_iterations'
        )

    def evaluate(
        self,
        dataset: DatasetInMemory,
        vol_idx: int,
        output_dir: str,
        batch_size: int,
        num_workers: int,
        store_visualization: bool = True,
        save_predicted_vol_as_nifti: bool = False,
        iteration: Optional[int] = None,
        file_name: Optional[str] = None,
        slice_vols_for_viz: Optional[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = None
    ):
        self._evaluation_mode()
        dataset.set_augmentation(False)

        # Create dictionary of the other volumes to visualize, namely the current prior
        current_pixels_size = dataset.get_processed_pixel_size()
        current_pixels_size /= np.array(self._rescale_factor) if self._rescale_factor is not None else 1 
        target_pixels_size = np.array(dataset.get_original_pixel_size(vol_idx)) 
        
        other_volumes_to_visualize = {
            'y_dae_or_atlas': {
                'vol': self._state.y_pl.cpu(),
                'current_pix_size': current_pixels_size,
                'target_pix_size': target_pixels_size,
                'target_img_size': dataset.get_original_image_size(vol_idx), # xyz
                'slice_z_axis': False
                } 
        } if self._state.y_pl is not None else None

        # Dictionary with the different normalization mode parameters
        normalization_modes = {
            'no_manual_normalization': False
        }

        if self._manually_norm_img_before_seg_eval:
            norm_type = f'with_manual_normalization_{self._normalization_strategy}'
            normalization_modes[norm_type] = True

        # Get the preprocessed vol for that has the same position as
        # the original vol (preprocessed vol may have a translation in xy)
        vol_preproc, _, bg_preproc = dataset.get_preprocessed_images(
            vol_idx, same_position_as_original=True)

        vol_orig, y_gt, _ = dataset.get_original_images(vol_idx) 

        file_name = f'{dataset.dataset_name}_vol_{vol_idx:03d}' \
            if file_name is None else file_name

        if iteration is not None:
            file_name = f'{file_name}_{iteration:03d}'
        
        predict_kwargs = {
            'bg_mask': bg_preproc,
            'bg_supp_x_norm': self._bg_supp_x_norm_eval,
            'bg_suppression_opts': self._bg_suppression_opts_eval,
            'batch_size': batch_size,
            'num_workers': num_workers
        }

        results = {}
        for norm_mode, norm_value in normalization_modes.items():
            output_dir_norm_mode = os.path.join(output_dir, norm_mode)

            predict_kwargs['other_fwd_pass_seg_kwargs'] = {
                'manually_norm_img_before_seg': norm_value
            }

            eval_metrics = BaseTTA.evaluate(
                self,
                x_preprocessed=vol_preproc,
                x_original=vol_orig,
                y_original_gt=y_gt.float(),
                preprocessed_pix_size=dataset.get_processed_pixel_size(),
                gt_pix_size=dataset.get_original_pixel_size(vol_idx),
                metrics=self._eval_metrics,
                classes_of_interest=self._classes_of_interest,
                output_dir=output_dir_norm_mode,
                file_name=file_name,
                store_visualization=store_visualization,
                save_predicted_vol_as_nifti=save_predicted_vol_as_nifti,
                slice_vols_for_viz=slice_vols_for_viz,
                predict_kwargs=predict_kwargs,
                other_volumes_to_visualize=other_volumes_to_visualize
            )

            results[norm_mode] = eval_metrics   

            for eval_metric, eval_metric_values in eval_metrics.items():
                metric_name = f'{norm_mode}/{eval_metric}'
                self._state.add_test_score(
                    iteration=iteration, metric_name=metric_name, score=eval_metric_values)

            # Print mean dice score of the foreground classes
            dices_fg_mean = np.mean(eval_metrics['dice_score_fg_classes']).mean().item()
            dices_fg_sklearn_mean = np.mean(eval_metrics['dice_score_fg_classes_sklearn']).mean().item()
            
            print(f'Iteration {iteration} - dice score_fg_classes ({norm_mode}): {dices_fg_mean}')
            print(f'Iteration {iteration} - dice score_fg_sklearn_mean ({norm_mode}): {dices_fg_sklearn_mean}') 

            if self._wandb_log and iteration is not None:
                wandb.log({f'dice_score_fg_sklearn_mean/{norm_mode}/img_{vol_idx:03d}': dices_fg_sklearn_mean,
                           'tta_step': iteration})
                
            if self._classes_of_interest is not None and iteration is not None:
                classes_of_interest = [cls - 1 for cls in self._classes_of_interest]
                dices_fg_sklearn_mean = np.mean(eval_metrics['dice_score_fg_classes_sklearn'][classes_of_interest])   
                
                print(f'Iteration {iteration} - dice score_classes_of_interest ({norm_mode}): {dices_fg_sklearn_mean}')

                if self._wandb_log:
                    wandb.log({f'dice_score_classes_of_interest/{norm_mode}/img_{vol_idx:03d}': dices_fg_sklearn_mean,
                                'tta_step': iteration})
            print()

        self._tta_fit_mode()
                
        return results

    
    def _normalize_image_intensities_to_sd(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Normalize image intensities to match the source domain statistics.

        Args:
            x_norm (torch.Tensor): Normalized image resulting from the normalization model.

        Returns:
            torch.Tensor: Preprocess normalized image intensities to match statistics of the source domain.
        """
        
        if self._normalization_strategy == 'standardize':
            x_norm_standardized = normalize(
                type='standardize', data=x_norm,
                mean=self._state.norm_td_statistics.mean, 
                std=self._state.norm_td_statistics.std
                )
            x_norm_norm_to_sd = self._norm_sd_statistics.std * x_norm_standardized + self._norm_sd_statistics.mean
        
        elif self._normalization_strategy == 'min_max':
            x_norm_btw_0_1 = normalize(
                type='min_max', data=x_norm,
                min=self._state.norm_td_statistics.min,
                max=self._state.norm_td_statistics.max
                )
            x_norm_norm_to_sd = (self._norm_sd_statistics.max - self._norm_sd_statistics.min) * x_norm_btw_0_1  +\
                self._norm_sd_statistics.min
                
        elif self._normalization_strategy == 'histogram_eq':
            raise NotImplementedError('Histogram equalization is not implemented yet')

        else:
            raise ValueError(f'Normalization strategy {self._normalization_strategy} is not valid')
        
        return x_norm_norm_to_sd


    def forward_pass_seg(
        self, 
        x: Optional[torch.Tensor] = None,
        bg_mask: Optional[torch.Tensor] = None,
        bg_supp_x_norm: bool = False,
        bg_suppression_opts: Optional[dict] = None,
        manually_norm_img_before_seg: bool = False,
        x_norm: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x_norm = x_norm if x_norm is not None else self._norm(x)
        
        # Normalize image intensities to match the source domain statistics
        if manually_norm_img_before_seg:
            x_norm = self._normalize_image_intensities_to_sd(x_norm)
        
        x_norm, mask, logits = super().forward_pass_seg(
            x_norm=x_norm, bg_mask=bg_mask, bg_supp_x_norm=bg_supp_x_norm,
            bg_suppression_opts=bg_suppression_opts
        )
        
        return x_norm, mask, logits
    
    
    @torch.inference_mode()
    def generate_pseudo_labels(
        self,
        dae_dataloader: DataLoader,
    ) -> torch.Tensor:  
        self._evaluation_mode()
        
        other_fwd_pass_seg_kwargs = {
                'manually_norm_img_before_seg': self._manually_norm_img_before_seg_tta
        }            

        _, masks, _ = self.predict(
            dae_dataloader, output_vol_format='1CDHW',
            bg_supp_x_norm=self._bg_supp_x_norm_tta_dae,
            bg_suppression_opts=self._bg_suppression_opts_tta,
            other_fwd_pass_seg_kwargs=other_fwd_pass_seg_kwargs
        )

        if self._rescale_factor is not None:
            masks = F.interpolate(masks, scale_factor=self._rescale_factor, mode='trilinear')

        dae_output, _ = self._dae(masks)
        dice_denoised = dice_score(masks, dae_output, soft=True, reduction='mean', foreground_only=False)
        dice_atlas = dice_score(masks, self._atlas, soft=True, reduction='mean',  foreground_only=False)
        
        if self._debug_mode:
            print(f'DEBUG: dice_denoised: {dice_denoised}, dice_atlas: {dice_atlas}')
        
        if (dice_denoised / dice_atlas >= self._alpha and dice_atlas >= self._beta) \
            or self._state.use_only_dae_pl \
            or (self._state.use_atlas_only_for_init and self._state.atlas_init_done):
            print('Using DAE output as pseudo label')
            dice = dice_denoised.item()
            self._state.use_dae_pl(dae_output)

        else:
            print('Using Atlas as pseudo label')
            dice = dice_atlas.item()
            self._state.use_atlas_pl(self._atlas)
            
        self._state.check_new_best_score(dice)
        
        self._tta_fit_mode()

        return self._state.y_pl
        
    def _normalize_image_intensities_to_sd(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Normalize image intensities to match the source domain statistics.

        Args:
            x_norm (torch.Tensor): Normalized image resulting from the normalization model.

        Returns:
            torch.Tensor: Preprocess normalized image intensities to match statistics of the source domain.
        """
        if self._normalization_strategy == 'standardize':
            x_norm_standardized = normalize(
                type='standardize', data=x_norm,
                mean=self._state.norm_td_statistics.mean,
                std=self._state.norm_td_statistics.std
                )
            x_norm_norm_to_sd = self._norm_sd_statistics.std * x_norm_standardized + self._norm_sd_statistics.mean
        
        elif self._normalization_strategy == 'min_max':
            x_norm_btw_0_1 = normalize(
                type='min_max', data=x_norm,
                min=self._state.norm_td_statistics.min,
                max=self._state.norm_td_statistics.max
                )
            x_norm_norm_to_sd = (self._norm_sd_statistics.max - self._norm_sd_statistics.min) * x_norm_btw_0_1  +\
                self._norm_sd_statistics.min
                
        elif self._normalization_strategy == 'histogram_eq':
            raise NotImplementedError('Histogram equalization is not implemented yet')

        else:
            raise ValueError(f'Normalization strategy {self._normalization_strategy} is not valid')
        
        return x_norm_norm_to_sd
    
    def _define_custom_wandb_metrics(self):
        wandb.define_metric("tta_step")
        wandb.define_metric('dice_score_fg_sklearn_mean/*', step_metric='tta_step')
        wandb.define_metric('dice_score_classes_of_interest/*', step_metric='tta_step')
        wandb.define_metric('tta_loss/*', step_metric='tta_step')
        
        if self._classes_of_interest is not None:
            wandb.define_metric('dice_score_classes_of_interest/*', step_metric='tta_step')

    def _save_checkpoint(self, logdir: str, dataset_name: str, index: int) -> None:
        cpt_dir = os.path.join(logdir, 'checkpoints')
        os.makedirs(cpt_dir, exist_ok=True)
        
        # Save normalizer weights with the highest agreement with the pseudo label
        save_checkpoint(
            os.path.join(cpt_dir, f'checkpoint_tta_{dataset_name}_{index:02d}_best_score.pth'), 
            **self.get_best_state(as_dict=True, remove_initial_state=True)
            )
        
        save_checkpoint(
            os.path.join(cpt_dir, f'checkpoint_tta_{dataset_name}_{index:02d}_last_step.pth'),
            **self.get_current_state(as_dict=True, remove_initial_state=True)
        )

    def load_best_state_norm(self) -> None:
        self._state.reset_to_state(self._state.best_state)
        self.load_current_norm_state_dict()

    def reset_state(self) -> None:
        self._tta_fit_mode()

        # Reset optimizer state
        self._optimizer = torch.optim.Adam(
            self._norm.parameters(),
            lr=self._learning_rate
        )
        
        # Reset TTADAEState to initial state
        self._state.reset()

        # Reset normalization model
        self.load_current_norm_state_dict()
    
    def load_current_norm_state_dict(self) -> None:
        self._norm.load_state_dict(self._state.norm_state_dict)
        self._state.norm_state_dict = self._norm.state_dict()
    
    def get_current_pseudo_label(self) -> Optional[torch.Tensor]:
        return self._state.y_pl
    
    def get_loss(self, loss_name: str) -> OrderedDict[int, float]:
        return self._state.get_loss(loss_name)
    
    def get_score(self, metric_name: str) -> OrderedDict[int, float]:
        return self._state.get_score(metric_name)
    
    def get_best_state(
        self,
        as_dict: bool = True,
        remove_initial_state: bool = True,
        down_cast_y_pl: bool = False
        ) -> TTADAEState | dict:
        if as_dict:
            best_state_dict = asdict(self._state.best_state)
            if remove_initial_state:
                best_state_dict['_initial_state'] = None 
            if down_cast_y_pl:
                best_state_dict['y_pl'] = best_state_dict['y_pl'].half()
            return best_state_dict
        
        else:
            best_state = self._state.best_state
            if remove_initial_state or down_cast_y_pl:
                best_state = best_state.current_state # create deep copy of best_state
                if remove_initial_state:
                    best_state._initial_state = None
                if down_cast_y_pl:
                    best_state.y_pl = best_state.y_pl.half() 
                
            return self._state.best_state
        
    def get_current_state(
        self,
        as_dict: bool = True,
        remove_initial_state: bool = True,
        remove_best_state: bool = True,
        down_cast_y_pl: bool = True
        ) -> TTADAEState | dict:
        if as_dict:
            current_state_dict = asdict(self._state)
            if remove_initial_state:
                current_state_dict['_initial_state'] = None 
            if remove_best_state:
                current_state_dict['_best_state'] = None
            if down_cast_y_pl:
                current_state_dict['y_pl'] = current_state_dict['y_pl'].half()
            return current_state_dict
        else:
            current_state = self._state
            if remove_initial_state or remove_best_state:
                current_state = current_state.current_state
                
                if remove_initial_state:
                    current_state._initial_state = None
                
                if remove_best_state:
                    current_state._best_state = None

            if down_cast_y_pl:
                current_state = current_state.current_state
                current_state.y_pl = current_state.y_pl.half()
            
            return current_state
                

    def get_model_selection_score(self) -> float:
        return self._state.model_selection_score

    def get_current_iteration(self) -> int:
        return self._state.iteration

    def load_state(self, path: str) -> None:
        raise NotImplementedError('Method not implemented for TTADAE')
    
    def save_state(self, path: str) -> None:
        raise NotImplementedError('Method not implemented for TTADAE')
