from typing import Optional
import torch
from torch.utils.data import Dataset

from tta_uia_segmentation.src.tta import TTAInterface
from tta_uia_segmentation.src.utils.loss import dice_score
from tta_uia_segmentation.src.utils.utils import resize_volume
from tta_uia_segmentation.src.utils.io import save_nii_image
from tta_uia_segmentation.src.utils.visualization import export_images


dice_score_fn_dict = {
    'dice_score_all_classes': lambda y_pred, y_gt: dice_score(y_pred, y_gt, soft=False, reduction='none', smooth=1e-5),
    'dice_score_fg_classes': lambda y_pred, y_gt: dice_score(y_pred, y_gt, soft=False, reduction='none', 
        foreground_only=True, smooth=1e-5)
}


class BaseTTA(TTAInterface):
    """
    Base class for Test-Time Adaptation (TTA) that implements the TTAInterface.
    
    Attributes
    ----------
    _adapted : bool
        Indicates whether the model has been adapted.
    """

    def __init__(self) -> None:
        """
        Initializes the BaseTTA class.
        
        """
        self._adapted = False
    
    def tta(self, x: torch.Tensor) -> None:
        """
        Perform test-time adaptation on the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data for TTA.
        """
        raise NotImplementedError("The method 'tta' is not implemented.")
    
    def predict(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions on the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data for prediction.
        """
        raise NotImplementedError("The method 'predict' is not implemented.")
    
    def evaluate(
        self,
        x_preprocessed: torch.Tensor, 
        y_original_gt: torch.Tensor,
        preprocessed_pix_size: tuple[float, ...],
        gt_pix_size: tuple[float, ...], 
        metrics: dict[str, callable] = dice_score_fn_dict,
        output_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        store_visualization: bool = False,
        x_original: Optional[torch.Tensor] = None,
        other_volumes_to_visualize: Optional[dict[str, torch.Tensor]] = None,
        save_predicted_vol_as_nifti: bool = False,
        slice_vols_for_viz: Optional[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = None,
        **kwargs
        ) -> dict[str, float]:
        """
        Evaluate the model on the input data and labels.

        Parameters
        ----------
        x_preprocessed : torch.Tensor
            Input image at the preprocessing resolution.
        y_gt : torch.Tensor
            Ground truth segmentation labels at the original resolution.
        preprocessed_pix_size : tuple[float, ...]
            Pixel size of the preprocessed input image.
        gt_pix_size : tuple[float, ...]
            Pixel size of the ground truth labels.
        metrics : dict[str, callable], optional
            Dictionary of metric functions to evaluate the model's performance.
        output_dir : str, optional
            Directory to save output visualizations.
        file_name : str, optional
            Base name for output files.
        store_visualization : bool, optional
            Whether to store visualizations of the results.
        other_volumes_to_visualize : dict[str, torch.Tensor], optional
            Additional volumes to include in the visualization.
        save_predicted_vol_as_nifti : bool, optional
            Whether to save the predicted volume as a NIfTI file.
        slice_vols_for_viz : tuple[tuple[int, int], tuple[int, int], tuple[int, int]], optional
            Slicing indices for visualization of volumes.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        dict[str, float]
            Dictionary of metric names and their corresponding values.
        """

        # Predict segmentation for x_preprocessed
        x_norm, y_pred, _ = self.predict(x_preprocessed, **kwargs)

        # Resize x_norm and y_pred to the original resolution
        x_norm, y_pred = [
            resize_volume(
                vol,
                current_pix_size=preprocessed_pix_size,
                target_pix_size=gt_pix_size,
                target_img_size=None,  # We assume no padding or cropping is needed to match image sizes
                mode='trilinear',
                only_inplane_resample=True     
            ) for vol in (x_norm, y_pred)
        ]

        # Measure the performance of the model
        metrics_values = {}
        for metric_name, metric_fn in metrics.items():
            metric_value = metric_fn(y_pred, y_original_gt)

            if isinstance(metric_value, torch.Tensor):
                if metric_value.ndim <= 1:
                    metric_value = metric_value.mean().item()
                else:   
                    metric_value = metric_value.tolist()

            metrics_values[metric_name] = metric_value

        # Save visualizations
        # TODO: Check if the volumes are in the correct format for visualization (e.g., shape, dtype)
        if store_visualization:
            assert output_dir is not None, "The output directory must be provided to store visualizations."
            assert file_name is not None, "The file name must be provided to store visualizations."

            # Slice volumes for visualization
            if slice_vols_for_viz is not None:
                slice_indices = tuple(slice(start, end) for start, end in slice_vols_for_viz)
                x_original, x_norm, y_original_gt, y_pred = [ 
                    vol[..., slice_indices[0], slice_indices[1], slice_indices[2]]
                    for vol in [x_original, x_norm, y_original_gt, y_pred]
                ]

            # Resize other volumes to the original resolution
            if other_volumes_to_visualize is not None:
                for vol_name, vol in other_volumes_to_visualize.items():
                    other_volumes_to_visualize[vol_name] = resize_volume(
                        vol,
                        current_pix_size=preprocessed_pix_size,
                        target_pix_size=gt_pix_size,
                        target_img_size=y_original_gt.shape[-3: ], # We assume no padding or cropping is needed to match image sizes
                        mode='trilinear',
                        only_inplane_resample=True     
                    )
            
            export_images(
                x_original,
                x_norm,
                y_original_gt,
                y_pred,
                output_dir=output_dir,
                image_name=file_name,
                **other_volumes_to_visualize
            )

        # Save the predicted volume as a NIfTI file
        if save_predicted_vol_as_nifti:
            assert output_dir is not None, "The output directory must be provided to save the predicted volume."
            assert file_name is not None, "The file name must be provided to save the predicted volume."

            # TODO: Check the format matches what the function expects
            save_nii_image(dir=output_dir, file_name=file_name + '.nii.gz')

        return metrics_values
    
    def evaluate_dataset(self, ds: Dataset) -> None:
        """
        Evaluate the model on a dataset.

        Parameters
        ----------
        ds : torch.utils.data.Dataset
            Dataset for evaluation.
        """
        raise NotImplementedError("The method 'evaluate_dataset' is not implemented.")
    
    def load_state(self, path: str) -> None:
        """
        Load the state of the model from a file.

        Parameters
        ----------
        path : str
            Path to the file containing the model state.
        """
        raise NotImplementedError("The method 'load_state' is not implemented.")
    
    def save_state(self, path: str) -> None:
        """
        Save the state of the model to a file.

        Parameters
        ----------
        path : str
            Path to the file where the model state will be saved.
        """
        raise NotImplementedError("The method 'save_state' is not implemented.")
    
    def reset_state(self) -> None:
        """
        Reset the state of the model.
        """
        raise NotImplementedError("The method 'reset_state' is not implemented.")
    
    def _evaluation_mode(self) -> None:
        """
        Set the model to evaluation mode.
        """
        raise NotImplementedError("The method '_evaluation_mode' is not implemented.")

    def set_adapted(self, adapted: bool) -> None:
        """
        Set the adapted state of the model.

        Parameters
        ----------
        adapted : bool
            Indicates whether the model has been adapted.
        """
        self._adapted = adapted

    def is_adapted(self) -> bool:
        """
        Check if the model has been adapted.

        Returns
        -------
        bool
            True if the model has been adapted, False otherwise.
        """
        return self._adapted