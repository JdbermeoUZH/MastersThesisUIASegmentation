import os
import copy
from tqdm import tqdm
from typing import Optional, Callable, Literal, Any
from dataclasses import asdict, dataclass, field

import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tta_uia_segmentation.src.dataset import Dataset
from tta_uia_segmentation.src.dataset.aug_tensor_dataset import AgumentedTensorDataset
from tta_uia_segmentation.src.dataset.utils import onehot_to_class, class_to_onehot
from tta_uia_segmentation.src.tta.BaseTTASeg import BaseTTASeg, EVAL_METRICS, BaseTTAState
from tta_uia_segmentation.src.models import BaseSeg
from tta_uia_segmentation.src.utils.utils import (
    default,
    get_seed,
    clone_state_dict_to_cpu
)
from tta_uia_segmentation.src.utils.loss import one_hot_score_to_onehot_pred


def quadratic_penalty_function(m1, m2):
    return F.relu(m1 - 0.9 * m2) ** 2 + F.relu(1.1 * m2 - m1) ** 2


def calculate_shape_moment(class_probs, n_classes, cx=0, cy=0, p=0, q=0, onehot=True, batch_size=-1):
    device = class_probs.device
    shape = class_probs.shape
    C, H, W = shape[-3:]

    class_probs = class_probs.reshape(-1, C, H, W)
    N = class_probs.shape[0]
    
    H_grid, W_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

    H_grid = (H_grid - cx) ** p
    W_grid = (W_grid - cy) ** q

    grid = H_grid * W_grid
    grid = grid.to(device)

    moments = []

    if batch_size == -1:
        batch_size = N
    
    loader = DataLoader(
        TensorDataset(class_probs),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    for probs, in loader:
        if not onehot:
            probs = class_to_onehot(probs, n_classes).to(device)

        moments.append((probs * grid).sum((2, 3)))

    moments = torch.cat(moments)

    return moments


class TTASLoss(torch.nn.Module):
    def __init__(
        self,
        source_class_ratios: torch.Tensor,
        descriptor_term_weight: Optional[torch.Tensor] = None,
        descriptor_func: Optional[Callable] = None,
        epsilon_pred: int = 10,
        weak_supervision: bool = False,
    ):
        super().__init__()

        self.source_class_ratios = source_class_ratios
        self.descriptor_term_weight = descriptor_term_weight
        self.epsilon_pred = epsilon_pred
        self.weak_supervision = weak_supervision

        weights = 1 / source_class_ratios
        weights /= weights.sum()
        self.weights = weights.reshape(1, -1, 1, 1)

        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

        self.descriptor_func = descriptor_func

    def forward(
        self,
        probs,
        target_descriptors: Optional[torch.Tensor] = None,
        add_descriptor_term: bool = False,
        labels: Optional[torch.Tensor] = None,
    ):

        N, C, H, W = probs.shape

        # Entropy loss
        # :============================================================================:
        entropy_term = -self.weights * probs * torch.log(probs + 1e-10)
        entropy_term = entropy_term.nansum(1).mean()

        # If one of the terms is NaN, make the entropy minimal
        if entropy_term.isnan().any():
            entropy_term = 0

        total_loss = entropy_term

        # KL Divergence loss
        # :============================================================================:
        source_class_ratios = torch.stack([self.source_class_ratios] * N)

        if self.weak_supervision:
            assert (
                labels is not None
            ), "Labels must be provided when weak supervision is used."
            soft_class_mask = labels.any(3).any(2).bool()
        else:
            soft_class_mask = (
                probs.sum((2, 3)) > 10
            )  # if True, class is considered to be present in the image

        source_class_ratios[~soft_class_mask] = 0
        source_class_ratios += 1e-10

        class_ratio = calculate_shape_moment(probs, C) / (H * W)
        kl_term = self.kl_loss(self.source_class_ratios.log(), class_ratio)
        kl_term = kl_term[:, 1:].sum() / N

        if kl_term.isnan().any():
            kl_term = 0

        total_loss += kl_term
        
        if not add_descriptor_term:
            return entropy_term + kl_term

        # Shape descriptor loss
        # :============================================================================:
        if add_descriptor_term:
            assert self.descriptor_func is not None, "Descriptor function must be provided."
            descriptor = self.descriptor_func(probs, C)
            descriptor = torch.stack(descriptor, dim=1)
            descriptor_term = quadratic_penalty_function(descriptor, target_descriptors)
            descriptor_term = (
                self.descriptor_term_weight * descriptor_term[:, :, 1:].nanmean() / (H * W)
            )

            if descriptor_term.isnan().any():
                descriptor_term = 0

            total_loss += descriptor_term

        return total_loss


@torch.jit.script
def softmax_entropy(logits: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    if weights is not None:
        return -(weights * logits.softmax(1) * logits.log_softmax(1)).sum(1)
    else:
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class EntropyMinLoss(torch.nn.Module):
    def __init__(
        self,
        use_kl_loss: bool = False,
        source_class_ratios: Optional[torch.Tensor] = None,
        weighted_loss: bool = False,
        filter_low_support_classes: bool = False,
        clases_to_exclude_ent_term: Optional[torch.Tensor | tuple[int, ...]] = None,
        classes_to_exclude_kl_term: Optional[torch.Tensor | tuple[int, ...]] = None,
        eps: float = 1e-10,
    ):
        super().__init__()

        self._use_kl_loss = use_kl_loss
        self._filter_low_support_classes = filter_low_support_classes
        self._classes_to_exclude_ent_term = clases_to_exclude_ent_term
        self._classes_to_exclude_kl_term = classes_to_exclude_kl_term
        self._eps = eps

        if use_kl_loss:
            assert source_class_ratios is not None, "Source class ratios must be provided if KL loss is used."
            self._source_class_ratios = source_class_ratios
        else:
            self._source_class_ratios = None
            
        if weighted_loss:
            assert self._source_class_ratios is not None, "Source class ratios must be provided if weighted loss is used."
            weights = 1 / self._source_class_ratios
            weights /= weights.sum()
            self.weights = weights.reshape(1, -1, 1, 1)
        else:
            self.weights = None

    def forward(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        pseudo_labels: Optional[torch.Tensor] = None,
        sum_prob_support_threshold: int = 10,
    ):
        assert logits is not None or probs is not None, "Either logits or probs must be provided."
        assert logits is None or probs is None, "Only one of logits or probs must be provided."
        
        if logits is not None:
            N, C, H, W = logits.shape
            probs = logits.softmax(1)
        else: 
            assert probs is not None, "Probs must be provided if logits are not."
            N, C, H, W = probs.shape
        
        # Entropy loss
        # :============================================================================:
        weights = default(self.weights, torch.ones((1, C, 1, 1), device=probs.device) / C)
        
        class_mask_ent_term = torch.ones(C, device=probs.device).bool()
        if self._classes_to_exclude_ent_term is not None:
            class_mask_ent_term[self._classes_to_exclude_ent_term] = False

        idx_slice = (slice(None), class_mask_ent_term, ...)

        if logits is not None:
            entropy_term = softmax_entropy(
                logits[idx_slice],
                weights[idx_slice]
            )
        else:
            weights_ = weights[idx_slice]
            probs_ = probs[idx_slice]
            entropy_term = - weights_ * probs_ * (probs_ + self._eps).log() 
            entropy_term = entropy_term.nansum(1).nanmean(0) + self._eps
      
        total_loss = entropy_term

        # KL Divergence loss
        # :============================================================================:
        if self._use_kl_loss:
            # Define mask of classes to filter out, if the option is chosen
            class_mask_kl_term = torch.ones(C, device=probs.device).bool()

            if self._classes_to_exclude_kl_term is not None:
                class_mask_kl_term[self._classes_to_exclude_kl_term] = False
            
            idx_slice = (slice(None), class_mask_kl_term, ...)

            assert self._source_class_ratios is not None, "Source class ratios must be provided if KL loss is used."
            source_class_ratios = torch.stack([self._source_class_ratios] * N)
            
            # Define mask of classes to filter out, if the option is chosen
            if self._filter_low_support_classes:
                if pseudo_labels is not None:
                    # They are at least present in the pseudo label map
                    low_supp = ~pseudo_labels.any(-1).any(-2).bool()
                else:
                    # They have at least some support
                    low_supp = probs.sum((-2, -1)) < sum_prob_support_threshold  
            else:
                low_supp = torch.zeros(C, device=probs.device).bool()
                
            # Set the class ratio to 0 for classes with very small support in 
            #  the pseudo label map or the predicted probabilities
            source_class_ratios[low_supp] = 0
            source_class_ratios += self.eps

            # Calculate kl divergence terms as KL(probs || source_class_ratios)
            #  and average over the batch
            log_class_ratio = (source_class_ratios + self._eps).log()
            log_probs = (probs + self._eps).log() if logits is None else logits.log_softmax(1)
            kl_batch = probs * (log_probs - log_class_ratio)

            kl_term = kl_batch[idx_slice].sum() / N # Exclude background class

            total_loss += kl_term

        return total_loss


class TTAEntropyMin(BaseTTASeg):

    def __init__(
        self,
        seg: BaseSeg,
        n_classes: int,
        class_prior_type: Literal["uniform", "data"] = "data",
        fit_at_test_time: Literal["normalizer", "bn_layers", "all"] = "bn_layers",
        learning_rate: float = 1e-6,
        weight_decay: float = 1e-3,
        lr_decay: bool = True,
        lr_scheduler_step_size: int = 20,
        lr_scheduler_gamma: float = 0.7,
        aug_params: Optional[dict] = None,
        classes_of_interest: Optional[tuple[int | str, ...]] = None,
        eval_metrics: dict[str, Callable] = EVAL_METRICS,
        viz_interm_outs: tuple[str, ...] = tuple(),
        wandb_log: bool = False,
        device: str | torch.device = "cuda",
        seed: Optional[int] = None,
        entropy_min_loss_kwargs: dict = {},
    ):

        # Check modules to fit at test time are present in the model
        assert fit_at_test_time in [
            "normalizer",
            "bn_layers",
            "all",
        ], "fit_at_test_time must be either 'normalizer' or 'bn_layers'."

        super().__init__(
            seg,
            n_classes,
            fit_at_test_time,
            classes_of_interest,
            eval_metrics,
            viz_interm_outs,
            wandb_log,
            device,
        )

        self._aug_params = aug_params
        self._entrop_min_loss_kwargs = entropy_min_loss_kwargs
        self._seed = default(seed, get_seed())

        # Class prior related attributes
        self._class_prior_type = class_prior_type

        if self._class_prior_type == "uniform":
            self._class_prior = (
                torch.ones(self._n_classes, device=device) / self._n_classes
            )

        elif self._class_prior_type == "data":
            self._class_prior = None

        else:
            raise ValueError("class_prior_type must be either 'uniform' or 'data'.")

        # Initialize optimizer and lr scheduler that will be used
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._optimizer = torch.optim.Adam(
                self.trainable_params_at_test_time,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        
        if lr_decay:
            self._lr_scheduler_step_size = lr_scheduler_step_size
            self._lr_scheduler_gamma = lr_scheduler_gamma
            self._secheduler = torch.optim.lr_scheduler.StepLR(
                self._optimizer,
                lr_scheduler_step_size,
                lr_scheduler_gamma)
        else:
            self._lr_scheduler_step_size = None
            self._lr_scheduler_gamma = None     
            self._secheduler = None

    def fit_class_prior(
        self,
        source_domain_data: Dataset | DataLoader,
        batch_size: int = 1,
        num_workers: int = 0,
        **other_dl_kwargs,
    ):
        """
        Fit the class prior using the source domain data.

        Computes the relative frequencies of each class a pixel can take in the source domain data.
        """

        assert (
            self._class_prior_type == "data"
        ), "Class prior type must be 'data' to fit the class prior using data."

        if isinstance(source_domain_data, Dataset):
            source_domain_data.augment = False
            source_domain_data = DataLoader(
                source_domain_data,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                **other_dl_kwargs,
            )
        elif not isinstance(source_domain_data, DataLoader):
            source_domain_data.dataset.augment = False

        class_counts = torch.zeros(self._n_classes, device=self._device)

        for _, y, *_ in source_domain_data:
            # Convert to class indices
            y = onehot_to_class(y, class_dim=1)

            # Count the number of times each class appears
            class_counts += torch.bincount(y.flatten(), minlength=self._n_classes)

        # Compute the class prior
        self._class_prior = class_counts / class_counts.sum()

    def _evaluation_mode(self) -> None:
        """
        Set the model to evaluation mode.
        """
        self._seg.eval_mode()

    def _tta_fit_mode(self) -> None:
        """
        Set the model to TTA fit mode.
        """

        if self._fit_at_test_time == "normalizer":
            # Set everything in the model to eval mode
            self._seg.eval_mode()

            # Set the normalizer to train mode
            self._seg.get_normalizer_module().train()

        elif self._fit_at_test_time == "bn_layers":
            # Set everything in the model to eval mode
            self._seg.eval_mode()

            # Set the batch normalization layers to train mode
            for m in self._seg.get_bn_layers():
                m.train()

        elif self._fit_at_test_time == "all":
            # Set everything in the model to train mode
            self._seg.train_mode()

    def tta(
        self,
        x: torch.Tensor | DataLoader,
        num_steps: int = 500,
        gradient_acc_steps: int = 1,
        evaluate_every: Optional[int] = None,
        registered_x_preprocessed: Optional[torch.Tensor] = None,
        x_original: Optional[torch.Tensor] = None,
        y_original_gt: Optional[torch.Tensor] = None,
        preprocessed_pix_size: Optional[tuple[float, ...]] = None,
        gt_pix_size: Optional[tuple[float, ...]] = None,
        metrics: Optional[dict[str, Callable]] = EVAL_METRICS,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        classes_of_interest: tuple[int] = tuple(),
        logdir: Optional[str] = None,
        save_checkpoints: bool = True,
        slice_vols_for_viz: Optional[
            tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        ] = None,
        file_name: Optional[str] = None,
        store_visualization: bool = True,
        save_predicted_vol_as_nifti: bool = False,
    ) -> None:
        # Check if prior exists or is fitted (happens for 'data' type prior)
        if self._class_prior_type == "data" and self._class_prior is None:
            raise ValueError(
                "Class prior must be fitted before running TTA "
                "when class_prior_type is 'data'."
            )

        # Initialize loss function object
        loss_fn = EntropyMinLoss(
            source_class_ratios=self._class_prior,
            **self._entrop_min_loss_kwargs
            )

        # Generate a DataLoader if a single volume is provided
        if isinstance(x, torch.Tensor):
            base_msg = "{arg} must be provided to create a DataLoader for the x Tensor."
            assert batch_size is not None, base_msg.format(arg="batch_size")
            assert num_workers is not None, base_msg.format(arg="num_workers")

            x = DataLoader(
                AgumentedTensorDataset(x, aug_params=self._aug_params, seed=self._seed),
                batch_size=batch_size,
                num_workers=num_workers,
            )

        # If evaluate_every is provided, verify the required args for eval are provided
        if evaluate_every is not None:
            base_msg = "{arg} must be provided to evaluate the model every {eval_every} iterations."
            assert registered_x_preprocessed is not None, base_msg.format(
                arg="registered_x_preprocessed", eval_every=evaluate_every
            )
            assert y_original_gt is not None, base_msg.format(
                arg="y_original_gt", eval_every=evaluate_every
            )
            assert preprocessed_pix_size is not None, base_msg.format(
                arg="preprocessed_pix_size", eval_every=evaluate_every
            )
            assert gt_pix_size is not None, base_msg.format(
                arg="gt_pix_size", eval_every=evaluate_every
            )

        # For loop for n-iterations
        pbar = tqdm(range(num_steps), desc="TTA-EntropyMin")

        for iter_i in pbar:
            # Check if it is the checkpoint to evaluate the model
            if evaluate_every is not None and iter_i % evaluate_every == 0:
                self.evaluate(
                    x_preprocessed=registered_x_preprocessed,  # type: ignore
                    x_original=x_original,
                    y_gt=y_original_gt.float(),  # type: ignore
                    preprocessed_pix_size=preprocessed_pix_size,  # type: ignore
                    gt_pix_size=gt_pix_size,  # type: ignore
                    iteration=iter_i,
                    metrics=metrics,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    classes_of_interest=classes_of_interest,
                    output_dir=logdir,
                    file_name=file_name,  # type: ignore
                    store_visualization=store_visualization,
                    save_predicted_vol_as_nifti=save_predicted_vol_as_nifti,
                    slice_vols_for_viz=slice_vols_for_viz,
                )

            step_loss = 0
            n_samples = 0

            # Backpropagation on entropy min loss
            self._tta_fit_mode()

            for step, (x_batch, ) in enumerate(x):
                x_batch = x_batch.to(self._device)
                
                # Forward pass
                _, y_logits, _ = self._seg(x_batch)

                # Calculate loss
                loss = loss_fn(logits=y_logits)

                # Backpropagate
                loss.backward()

                # Gradient accumulation
                if step % gradient_acc_steps == 0:
                    # Update the weights
                    self._optimizer.step()
                    self._optimizer.zero_grad()

                # Track the loss
                with torch.no_grad():
                    step_loss += loss.item() * x_batch.shape[0]
                    n_samples += x_batch.shape[0]

            # Update the learning rate
            if self._secheduler is not None:
                self._secheduler.step()    

            tta_loss = step_loss / n_samples

            self._state.add_test_loss(
                iteration=iter_i, loss_name='tta_loss', loss_value=tta_loss)

            if self._wandb_log:
                wandb.log({f'tta_loss/{file_name}': tta_loss, 'tta_step': iter_i})
    
        self._state.is_adapted = True

        if save_checkpoints:
            assert logdir is not None, "logdir must be provided to save the checkpoints."
            assert file_name is not None, "file_name must be provided to save the checkpoints."
            self.save_state(
                path=os.path.join(logdir, f"tta_entropy_min_{file_name}_last.pth")
            )

        if logdir is not None:
            assert file_name is not None, "file_name must be provided to store the test scores."
            self._state.store_test_scores_in_dir(
                output_dir=os.path.join(logdir, 'tta_score'),
                file_name_prefix=file_name,
                reduce='mean_accross_iterations'
            )          

    @property
    def trainable_params_at_test_time(self) -> list[torch.nn.Parameter]:
        """
        Return the trainable parameters at test time.
        """
        if self._fit_at_test_time == "normalizer":
            return list(self._seg.get_normalizer_module().parameters())

        elif self._fit_at_test_time == "bn_layers":
            return [
                param for ly in self._seg.get_bn_layers() for param in list(ly.parameters())]

        elif self._fit_at_test_time == "all":
            # Set everything in the model to train mode
            return self._seg.trainable_params
        else:
            raise ValueError("fit_at_test_time must one of ['normalizer', 'bn_layers', 'all'].")

    def reset_state(self) -> None:
        self._tta_fit_mode()

        # Reset optimizer state
        self._optimizer = torch.optim.Adam(
            self.trainable_params_at_test_time,
            lr=self._learning_rate,
            weight_decay=self._weight_decay
        )
        
        # Reset scheduler state, if necessary
        if self._secheduler is not None:
            self._secheduler = torch.optim.lr_scheduler.StepLR(
                self._optimizer,
                self._lr_scheduler_step_size,   # type: ignore
                self._lr_scheduler_gamma        # type: ignore
            )

        # Reset state of the class and its fitted modules 
        super().reset_state()