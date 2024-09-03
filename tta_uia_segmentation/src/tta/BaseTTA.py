import torch
from torch.utils.data import Dataset

from tta_uia_segmentation.src.tta import TTAInterface
from tta_uia_segmentation.src.utils.loss import dice_score


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
    
    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
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
        y_gt: torch.Tensor,
        preprocessed_pix_size: tuple[float, ...],
        gt_pix_size: tuple[float, ...], 
        metrics: dict[str, callable] = dice_score_fn_dict,
        **kwargs
        ) -> float:
        """
        Evaluate the model on the input data and labels.

        Parameters
        ----------
        x : torch.Tensor
            Input image at the preprocessing resolution in DCHW format
        y : torch.Tensor
            Ground truth segmentation labels at the original resolution in CDHW format

        Returns
        -------
        float
            Accuracy of the model on the input data.
        """

        # Predict segmentation for x_preprocessed
        y_pred = self.predict(x_preprocessed, **kwargs)

        # Resize original images to preprocessed resolution
        scale_factor = gt_pix_size / preprocessed_pix_size

        output_size = (y_.shape[2:] * scale_factor).round().astype(int).tolist()


        return 
    
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