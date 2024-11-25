from typing import Optional, Callable

import numpy as np
import torch

from tta_uia_segmentation.src.dataset import Dataset
from tta_uia_segmentation.src.tta.BaseTTASeg import BaseTTASeg, EVAL_METRICS
from tta_uia_segmentation.src.models import BaseSeg


class NoTTASeg(BaseTTASeg):
    """
    A class that implements no test-time adaptation, simply evaluating a segmentation model.

    This class extends the BaseTTA abstract class and overrides its methods to provide
    a no-adaptation implementation.

    """

    def __init__(
        self,
        seg: BaseSeg,
        n_classes: int,
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
        eval_metrics = self.evaluate(
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
