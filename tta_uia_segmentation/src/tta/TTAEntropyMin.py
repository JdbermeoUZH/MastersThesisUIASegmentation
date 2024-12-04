from typing import Optional, Callable, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from tta_uia_segmentation.src.dataset import Dataset
from tta_uia_segmentation.src.dataset.utils import onehot_to_class
from tta_uia_segmentation.src.tta.BaseTTASeg import BaseTTASeg, EVAL_METRICS
from tta_uia_segmentation.src.models import BaseSeg


class TTAEntropyMin(BaseTTASeg):

    def __call__(
        self,
        seg: BaseSeg,
        n_classes: int,
        class_prior_type: Literal['uniform', 'data'],
        fit_at_test_time: Literal['normalizer', 'bn_layers', 'all'], 
        classes_of_interest: Optional[tuple[int | str, ...]] = None,
        eval_metrics: dict[str, Callable] = EVAL_METRICS,
        viz_interm_outs: tuple[str, ...] = tuple(),
        wandb_log: bool = False,
        device: str | torch.device = "cuda",
    ):

        super().__init__(
            seg,
            n_classes,
            classes_of_interest,
            eval_metrics,
            viz_interm_outs,
            wandb_log,
            device,
        )

        # Check modules to fit at test time are present in the model
        assert fit_at_test_time in ['normalizer', 'bn_layers', 'all'], "fit_at_test_time must be either 'normalizer' or 'bn_layers'."

        if fit_at_test_time == 'normalizer':
            assert self._seg.has_normalizer_module(), "Model does not have a normalizer module to fit at test time."

        if fit_at_test_time == 'bn_layers':
            assert self._seg.has_bn_layers(), "Model does not have batch normalization layers to fit at test time."

        self._fit_at_test_time = fit_at_test_time
    
        # Class prior related attributes
        self._class_prior_type = class_prior_type

        if self._class_prior_type == 'uniform':
            self._class_prior = torch.ones(self._n_classes, device=device) / self._n_classes


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

        assert self._class_prior_type == 'data', "Class prior type must be 'data' to fit the class prior using data."

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

        if self._fit_at_test_time == 'normalizer':
            # Set everything in the model to eval mode
            self._seg.eval_mode()

            # Set the normalizer to train mode
            self._seg.get_normalizer_module().train()
        
        elif self._fit_at_test_time == 'bn_layers':
            # Set everything in the model to eval mode
            self._seg.eval_mode()

            # Set the batch normalization layers to train mode
            for m in self._seg.get_bn_layers():
                m.train()

        elif self._fit_at_test_time == 'all':
            # Set everything in the model to train mode
            self._seg.train_mode()

    def tta(
        self,
        dataset: Dataset,
        vol_idx: int,
        batch_size: int,
        num_workers: int,
        metrics: Optional[dict[str, Callable]] = None,
        classes_of_interest: tuple[int] = tuple(),
        logdir: Optional[str] = None,
        slice_vols_for_viz: Optional[
            tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        ] = None,
        file_name: Optional[str] = None,
        store_visualization: bool = True,
        save_predicted_vol_as_nifti: bool = True,
    ) -> None:
        
        pass

    def evaluate(
        self,
        dataset: Dataset,
        vol_idx: int,
        batch_size: int,
        num_workers: int,
        metrics: Optional[dict[str, Callable]] = None,
        classes_of_interest: tuple[int] = tuple(),
        logdir: Optional[str] = None,
        slice_vols_for_viz: Optional[
            tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        ] = None,
        file_name: Optional[str] = None,
        store_visualization: bool = True,
        save_predicted_vol_as_nifti: bool = True,
    ) -> None:

        self._evaluation_mode()
        dataset.augment = False

        # Get the preprocessed vol for that has the same position as
        # the original vol (preprocessed vol may have a translation in xy)
        x_preprocessed, *_ = dataset.get_preprocessed_original_volume(vol_idx)

        x_original, y_original_gt = dataset.get_original_volume(vol_idx)

        file_name = (
            f"{dataset.dataset_name}_vol_{vol_idx:03d}"
            if file_name is None
            else file_name
        )

        # Evaluate the model on the volume
        eval_metrics = super().evaluate(
            x_preprocessed=x_preprocessed,
            x_original=x_original,
            y_original_gt=y_original_gt.float(),
            preprocessed_pix_size=dataset.get_processed_pixel_size_w_orientation(),
            gt_pix_size=dataset.get_original_pixel_size_w_orientation(vol_idx),
            metrics=metrics,
            batch_size=batch_size,
            num_workers=num_workers,
            classes_of_interest=classes_of_interest,
            output_dir=logdir,
            file_name=file_name,
            store_visualization=store_visualization,
            save_predicted_vol_as_nifti=save_predicted_vol_as_nifti,
            slice_vols_for_viz=slice_vols_for_viz,
        )

        for eval_metric_name, eval_metric_values in eval_metrics.items():
            self.state.add_test_score(
                iteration=0, metric_name=eval_metric_name, score=eval_metric_values
            )

        # Print mean dice score of the foreground classes
        dices_fg_mean = np.mean(eval_metrics["dice_score_fg_classes"]).mean().item()
        print(f"dice score_fg_classes (vol{vol_idx}): {dices_fg_mean}")

        dices_fg_mean_sklearn = (
            np.mean(eval_metrics["dice_score_fg_classes_sklearn"]).mean().item()
        )
        print(f"dice score_fg_classes_sklearn (vol{vol_idx}): {dices_fg_mean_sklearn}")



    