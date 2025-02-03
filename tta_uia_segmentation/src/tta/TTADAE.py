import os
from tqdm import tqdm
from dataclasses import asdict, dataclass, field
from typing import Callable, OrderedDict, Optional, Any, Literal

import wandb
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from tta_uia_segmentation.src.models import BaseSeg
from tta_uia_segmentation.src.utils.loss import dice_score, DiceLoss, onehot_to_class
from tta_uia_segmentation.src.utils.utils import (
    get_seed, generate_2D_dl_for_vol, resize_volume)
from tta_uia_segmentation.src.utils.io import save_checkpoint
from tta_uia_segmentation.src.dataset.aug_tensor_dataset import AgumentedTensorDataset
from tta_uia_segmentation.src.dataset import Dataset
from tta_uia_segmentation.src.tta.BaseTTASeg import BaseTTAState, BaseTTASeg, EVAL_METRICS


ALLOWED_DATASET_CLASSES = (AgumentedTensorDataset, Dataset)


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
    norm_td_statistics : Optional[DomainStatistics]
        Statistics of the normalized target domain.
    """
    using_dae_pl: bool = False
    using_atlas_pl: bool = False
    use_only_dae_pl: bool = False
    use_only_atlas: bool = False
    use_atlas_only_for_init: bool = False
    y_pl: torch.Tensor = field(default=None)
           
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

        self.y_pl = y_dae.detach().cpu()
        
    def use_atlas_pl(self, y_atlas: torch.Tensor):
        """
        Switch to using Atlas pseudo-labels.

        This method sets the flags to use Atlas pseudo-labels and not use DAE pseudo-labels.
        """
        assert y_atlas is not None, "Atlas pseudo-labels cannot be None if using Atlas as pseudo-labels"
        assert not self.use_only_dae_pl, "Cannot use only Atlas pseudo-labels with only DAE pseudo-labels"
        
        self.using_atlas_pl = True
        self.using_dae_pl = False

        if self.use_atlas_only_for_init and self.atlas_init_done:
            raise RuntimeError("Cannot switch to Atlas pseudo-labels after initialization" +
                               " when use_atlas_only_for_intit is enabled")
    
        self.y_pl = y_atlas.detach().cpu()

    @property
    def current_state(self) -> 'TTADAEState':
        """
        Get a deep copy of the current state of the TTADAEState.

        state_dict's that correspond to model states are moved to the CPU.

        Returns:
            TTADAEState: The current state.
        """

        return TTADAEState(**super().current_state_as_dict)

    @property
    def y_pl_categorical(self) -> torch.Tensor:
        """
        Get the pseudo-labels in categorical format.

        Returns:
            torch.Tensor: Pseudo-labels in categorical format. N1DHW format.
        """
        return onehot_to_class(self.y_pl) if self.y_pl is not None else None


class TTADAE(BaseTTASeg):
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
        seg: BaseSeg,
        dae: torch.nn.Module, 
        n_classes: int, 
        atlas: Optional[Any] = None,
        rescale_factor: tuple[int, int, int] = (1, 1, 1),
        fit_at_test_time: Literal["normalizer", "bn_layers", "all"] = "normalizer",
        aug_params: Optional[dict[str, Any]] = None,
        loss_func: torch.nn.Module = DiceLoss(),
        learning_rate: float = 1e-3,
        max_grad_norm: Optional[float] = 0.5,
        alpha: float = 1.0,
        beta: float = 0.25,
        use_only_dae_pl: bool = False,
        use_only_atlas: bool = False,
        use_atlas_only_for_intit: bool = False,
        eval_metrics: OrderedDict[str, callable] = EVAL_METRICS,
        classes_of_interest: Optional[tuple[int, ...]] = tuple(),
        viz_interm_outs: tuple[str, ...] = tuple(),
        seed: Optional[int] = None,
        wandb_log: bool = False,
        device: str = 'cuda',
        optimizer: Optional[torch.optim.Optimizer] = None,
        bg_supp_x_norm_tta_dae: bool = False,
        bg_suppression_opts_tta_dae: Optional[dict] = None,
        debug_mode: bool = False
        ) -> None:

        if fit_at_test_time == "normalizer":
            viz_interm_outs = tuple(
                list(viz_interm_outs) + ["Normalized Image"]
            )

        super().__init__(
            seg=seg,
            n_classes=n_classes,
            fit_at_test_time=fit_at_test_time,
            aug_params=aug_params,
            classes_of_interest=classes_of_interest,
            eval_metrics=eval_metrics,
            viz_interm_outs=viz_interm_outs,
            wandb_log=wandb_log,
            device=device,
            seed=seed
        )

        # Models and objects used in TTA
        self._dae = dae
        self._atlas = atlas
        
        # Rescale factor for pseudo labels
        self._rescale_factor = rescale_factor
        
        # Thresholds for pseudo label selection
        if atlas is None:
            assert alpha == 0 and beta == 0, "Cannot use alpha and beta thresholds without an Atlas"
        self._alpha = alpha
        self._beta = beta      
          
        # Loss function and optimizer
        self._loss_func = loss_func
        self._learning_rate = learning_rate
        
        if optimizer is None:
            self._optimizer = torch.optim.Adam(
                self.tta_fitted_params,
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

        # Initialize the state of the class
        self._state = TTADAEState(
            using_dae_pl=False,
            using_atlas_pl=False,
            use_only_dae_pl=(self._alpha == 0 and self._beta == 0) or use_only_dae_pl,
            use_only_atlas=use_only_atlas,
            use_atlas_only_for_init=use_atlas_only_for_intit,
        )

        # Set the object in tta_fit_mode 
        self._tta_fit_mode()

        self._debug_mode = debug_mode
    
    def _evaluation_mode(self) -> None:
        """
        Set the models to evaluation mode.

        """
        self._seg.eval_mode()
        self._dae.eval()
        self._dae.requires_grad_(False)

    def _tta_fit_mode(self) -> None:
        """
        Set the model to TTA fit mode.
        """
        self._dae.eval()
        self._dae.requires_grad_(False)

        if self._fit_at_test_time == "normalizer":
            # Set everything but the normalizer to eval mode
            for module in self._seg.get_all_modules_except_normalizer().values():
                module.eval()
                module.requires_grad_(False)

            # Set the normalizer to train mode
            self._seg.get_normalizer_module().train()

        elif self._fit_at_test_time == "bn_layers":
            # Set everything but the bn layers to eval mode
            for module in self._seg.get_all_modules_except_bn_layers().values():
                module.eval()
                module.requires_grad_(False)

            # Set the batch normalization layers to train mode
            for _, m in self._seg.get_bn_layers().items():
                m.train()

        elif self._fit_at_test_time == "all":
            # Set everything in the model to train mode
            self._seg.train_mode()

    def tta(
        self,
        x: torch.Tensor | DataLoader,
        num_steps: int = 500,
        batch_size: int = 16,
        num_workers: int = 1,
        calculate_dice_every: int = 25,
        update_dae_output_every: int = 25,
        accumulate_over_volume: bool = True,
        const_aug_per_volume: bool = False,
        save_checkpoints: bool = True,
        output_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        store_visualization: bool = True,
        save_predicted_vol_as_nifti: bool = False,
        registered_x_preprocessed: Optional[torch.Tensor] = None,
        x_original: Optional[torch.Tensor] = None,
        y_gt: Optional[torch.Tensor] = None,
        preprocessed_pix_size: Optional[tuple[float, ...]] = None,
        gt_pix_size: Optional[tuple[float, ...]] = None,
        slice_vols_for_viz: Optional[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = None,
        metrics: Optional[dict[str, Callable]] = EVAL_METRICS,
    ) -> None:

        # Change batch size for the pseudo label (in case it has a smaller size)
        if self._rescale_factor is not None:
            assert (batch_size * self._rescale_factor[0]) % 1 == 0
            label_batch_size = int(batch_size * self._rescale_factor[0])
        else:
            label_batch_size = batch_size

        # Generate a DataLoader if a single volume is provided
        if isinstance(x, torch.Tensor):
            base_msg = "{arg} must be provided to create a DataLoader for the x Tensor."
            assert batch_size is not None, base_msg.format(arg="batch_size")
            assert num_workers is not None, base_msg.format(arg="num_workers")
            x = self.convert_volume_to_DCHW_dl(
                x, batch_size=batch_size, num_workers=num_workers, pin_memory=True
            )

        # Assert 
        assert isinstance(x, DataLoader), "x must be a DataLoader"
        assert isinstance(x.dataset, ALLOWED_DATASET_CLASSES), "The dataset must be an AgumentedTensorDataset or Dataset"
        
        y_pl = None
        pl_dataloader = None
        
        # Start TTA iterations 
        for step in tqdm(range(num_steps)):
            self._state.iteration = step

            # Update Pseudo label, with DAE or Atlas, depending on which has a better agreement
            if step % update_dae_output_every == 0:
                
                y_pl = self.generate_pseudo_labels(dae_dataloader=x)

                pl_dataloader = generate_2D_dl_for_vol(
                    y_pl,
                    batch_size=label_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    drop_last=False, 
                    pin_memory=True
                )
                
            # Test performance during adaptation.
            if (step % calculate_dice_every == 0 or step == num_steps - 1) and calculate_dice_every != -1:
                _ = self.evaluate(
                    x_preprocessed=registered_x_preprocessed,  # type: ignore
                    x_original=x_original,
                    y_gt=y_gt.float(),  # type: ignore
                    preprocessed_pix_size=preprocessed_pix_size,  # type: ignore
                    gt_pix_size=gt_pix_size,  # type: ignore
                    iteration=step,
                    metrics=metrics,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    classes_of_interest=self._classes_of_interest, # type: ignore
                    output_dir=output_dir,
                    file_name=file_name,  # type: ignore
                    store_visualization=store_visualization,
                    save_predicted_vol_as_nifti=save_predicted_vol_as_nifti,
                    slice_vols_for_viz=slice_vols_for_viz,
                )

            # Take optimization TTA step
            self._tta_fit_mode()
            x.dataset.augment = True  
            
            tta_loss = 0
            n_samples = 0
            
            if accumulate_over_volume:
                self._optimizer.zero_grad()

            if const_aug_per_volume:
                x.seed = get_seed()
                                                            
            for (x_b, *_), (y_pl,) in zip(x, pl_dataloader):

                if not accumulate_over_volume:
                    self._optimizer.zero_grad()

                x_b = x_b.to(self._device).float()
                y_pl = y_pl.to(self._device)
                
                mask, *_ = self._seg(x_b)

                if self._rescale_factor is not None:
                    # Downsample the mask to the size of the pseudo label
                    mask = resize_volume(
                        mask,
                        target_pix_size=[1., 1., 1.],
                        current_pix_size=self._rescale_factor,
                        only_inplane_resample=False,
                        input_format='DCHW',
                        output_format='DCHW')
                    
                loss = self._loss_func(mask, y_pl)

                if accumulate_over_volume:
                    loss /= len(x_b)
                
                loss.backward()

                if not accumulate_over_volume:
                    if self._max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.tta_fitted_params, self._max_grad_norm)
                    self._optimizer.step()

                with torch.no_grad():
                    tta_loss += loss.detach() * x_b.shape[0]
                    n_samples += x_b.shape[0]

            if accumulate_over_volume:
                if self._max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.tta_fitted_params, self._max_grad_norm)
                self._optimizer.step()

            tta_loss = (tta_loss / n_samples).item()

            self._state.add_test_loss(
                iteration=step, loss_name='tta_loss', loss_value=tta_loss)
            
            if self._wandb_log:
                wandb.log({f'tta_loss/{file_name}': tta_loss, 'tta_step': step})

        if save_checkpoints:
            assert output_dir is not None, "output_dir must be provided to save the checkpoints."
            assert file_name is not None, "file_name must be provided to save the checkpoints."
            self.save_state(path=os.path.join(output_dir, f"tta_dae_{file_name}_last.pth"))

        self._state.is_adapted = True

        if output_dir is not None:
            assert file_name is not None, "file_name must be provided to store the test scores."
            self._state.store_test_scores_in_dir(
                output_dir=os.path.join(output_dir, 'tta_score'),
                file_name_prefix=file_name,
                reduce='mean_accross_iterations'
            )        
    
    @torch.inference_mode()
    def generate_pseudo_labels(
        self,
        dae_dataloader: DataLoader,
    ) -> torch.Tensor:  
        self._evaluation_mode()
        
        masks, *_ = self.predict(
            dae_dataloader,
            include_interm_outs=False, 
            output_vol_format='1CDHW',
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
        
    
    def _define_custom_wandb_metrics(self):
        wandb.define_metric("tta_step")
        wandb.define_metric('dice_score_fg_sklearn_mean/*', step_metric='tta_step')
        wandb.define_metric('tta_loss/*', step_metric='tta_step')
        
        if self._classes_of_interest is not None:
            wandb.define_metric('dice_score_classes_of_interest/*', step_metric='tta_step')

    def reset_state(self) -> None:
        self._tta_fit_mode()

        # Reset optimizer state
        self._optimizer = torch.optim.Adam(
            self.tta_fitted_params,
            lr=self._learning_rate
        )
        
        # Reset state of the class and its fitted modules 
        super().reset_state()

    def get_current_pseudo_label(self) -> Optional[torch.Tensor]:
        return self._state.y_pl
    
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
                
    def load_state(self, path: str) -> None:
        """
        Load the state of the model from a file.

        Parameters
        ----------
        path : str
            Path to the file containing the model state.
        """
        state_dict = torch.load(path)
        self._seg.load_checkpoint_from_dict(state_dict["seg"])
        self._state.reset_to_state(state_dict["state"])

    def save_state(self, path: str) -> None:
        """
        Save the state of the model to a file.

        Parameters
        ----------
        path : str
            Path to the file where the model state will be saved.
        """
        save_checkpoint(
            path,
            seg=self._seg.checkpoint_as_dict(),
            state=self._state.current_state_as_dict,
        )    

