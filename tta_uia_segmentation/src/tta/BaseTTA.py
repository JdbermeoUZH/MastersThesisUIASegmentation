import os
import copy
from typing import Optional, Literal, Union
from collections import OrderedDict
from dataclasses import dataclass, field, asdict

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from tta_uia_segmentation.src.tta import TTAInterface, TTAStateInterface
from tta_uia_segmentation.src.utils.loss import dice_score, onehot_to_class
from tta_uia_segmentation.src.utils.utils import resize_volume
from tta_uia_segmentation.src.utils.io import save_nii_image
from tta_uia_segmentation.src.utils.visualization import export_images


dice_score_fn_dict = {
    'dice_score_all_classes': lambda y_pred, y_gt: dice_score(y_pred, y_gt, soft=False, reduction='none', smooth=1e-5),
    'dice_score_fg_classes': lambda y_pred, y_gt: dice_score(y_pred, y_gt, soft=False, reduction='none', 
        foreground_only=True, smooth=1e-5)
}

def _preprocess_volumes_for_viz(
        *vols,
        convert_to_categorical: bool = False,
        slice_idxs: Optional[tuple] = None,
    ) -> tuple[torch.Tensor]:
    """
    Preprocess volumes for visualization by slicing and squeezing them.

    Parameters
    ----------
    *vols : tuple of torch.Tensor
        Volumes to preprocess.
    convert_to_categorical : bool, optional
        Whether to convert volumes to categorical, by default False.
    slice_idxs : tuple of tuple of int, optional
        Slicing indices for the volumes, by default None.

    Returns
    -------
    tuple of torch.Tensor
        Preprocessed volumes.
    """
    preprocessed_vols = []
    
    for vol in vols:
        # Slice vol
        if slice_idxs is not None:
            slice_idxs = tuple(slice(start, end) for start, end in slice_idxs)
            vol = vol[..., slice_idxs[0], slice_idxs[1], slice_idxs[2]]
        
        # Convert to categorical
        if convert_to_categorical:
            vol = onehot_to_class(vol)

        # Convert from NCDHW to NDCHW
        vol = vol.permute(0, 2, 1, 3, 4)

        # Squeeze vol
        vol = vol.squeeze()

        preprocessed_vols.append(vol)

    return preprocessed_vols


@dataclass
class BaseTTAState(TTAStateInterface):
    """
    Dataclass to store the state of the BaseTTA.

    Attributes
    ----------
    is_adapted : bool
        Flag indicating whether adaptation has been performed.
    iteration : int
        Current iteration of the TTA process.
    model_selection_score : float
        Best score achieved during TTA, used for model selection.
    tta_losses : dict
        Dictionary to store TTA losses for different metrics.
    test_scores : dict
        Dictionary to store test scores for different metrics.
    _create_initial_state : bool
        Flag to create initial state.
    _initial_state : BaseTTAState, optional
        Initial state of the TTA process.
    _best_state : BaseTTAState, optional
        Best state achieved during TTA.
    """
    is_adapted: bool = False
    iteration: int = 0
    model_selection_score: float = float('-inf')
    tta_losses: dict[str | int, OrderedDict[int, list | float]] = field(default_factory=dict)
    test_scores: dict[str | int, OrderedDict[int, list | float]] = field(default_factory=dict)

    _create_initial_state: bool = field(default=True)
    _initial_state: Optional[Union['BaseTTAState', dict]] = field(default=None)
    _best_state: Optional[Union['BaseTTAState', dict]] = field(default=None) 

    def reset(self) -> None:
        """
        Reset the state of the class to the initial state.
        """
        self.reset_to_state(self._initial_state)
        self._initial_state = self.current_state

    def reset_to_state(self, state: 'BaseTTAState') -> None:
        """
        Reset the state of the class to the given state.

        Parameters
        ----------
        state : BaseTTAState
            The state to reset to.
        """
        self.__dict__.update(state.__dict__)
    
    def __post_init__(self):
        """Create the initial state of the class."""
        # Store the initial state of the class
        if self._create_initial_state:
            self._initial_state = self.current_state
            #self._initial_state._initial_state = self._initial_state.current_state

        # Convert the _initial_state and _best_state attributes to TTAState objects
        if isinstance(self._initial_state, dict):
            self._initial_state = self.__class__(**self._initial_state)
        
        if isinstance(self._best_state, dict):
            self._best_state = self.__class__(**self._best_state)
    
    @property
    def current_state(self) -> 'BaseTTAState':
        """
        Get the current state of the BaseTTAState.

        Returns
        -------
        BaseTTAState
            The current state.
        """
        # # Copy the _initial_state and _best_state attributes
        # _initial_state = self._initial_state.current_state \
        #     if self._initial_state is not None else None
        # _best_state = self._best_state.current_state \
        #     if self._best_state is not None else None
        
        # Get the dict of the remaining attributes and deep copy them
        current_state_dict = asdict(self)
        # del current_state_dict['_initial_state']
        # del current_state_dict['_best_state']
        current_state_dict['_create_initial_state'] = False

        current_state_dict = {key: copy.deepcopy(value) 
                              for key, value in current_state_dict.items()}

        # Create a new BaseTTAState object with the copied attributes
        current_state = BaseTTAState(**current_state_dict)
        # current_state._initial_state = _initial_state
        # current_state._best_state = _best_state

        return current_state

    @property
    def initial_state(self) -> 'BaseTTAState':
        """
        Get the initial state of the BaseTTAState.

        Returns
        -------
        BaseTTAState
            The initial state.
        """
        return self._initial_state

    @property
    def best_state(self) -> 'BaseTTAState':
        """
        Get the best state of the BaseTTAState.

        Returns
        -------
        BaseTTAState
            The best state.
        """
        return self._best_state

    @property
    def best_iteration(self) -> int:
        """
        Get the iteration number of the best state.

        Returns
        -------
        int
            The iteration number of the best state.
        """
        return self._best_state.iteration
    
    @property
    def best_model_selection_score(self) -> float:
        """
        Get the model selection score of the best state.

        Returns
        -------
        float
            The model selection score of the best state.
        """
        return self._best_state.model_selection_score
    
    def add_test_score(self, iteration: int, metric_name: str, score: float | list):
        """
        Add a test score to the test_scores OrderedDict.

        Parameters
        ----------
        iteration : int
            The current iteration number.
        metric_name : str
            The name of the metric.
        score : float or list
            The score value.
        """
        if metric_name not in self.test_scores:
            self.test_scores[metric_name] = OrderedDict()
        self.test_scores[metric_name][iteration] = score

    def add_test_loss(self, iteration: int, loss_name: str, loss_value: float | list):
        """
        Add a test loss to the tta_losses OrderedDict.

        Parameters
        ----------
        iteration : int
            The current iteration number.
        loss_name : str
            The name of the loss.
        loss_value : float or list
            The loss value.
        """
        if loss_name not in self.tta_losses:
            self.tta_losses[loss_name] = OrderedDict()
        self.tta_losses[loss_name][iteration] = loss_value
    
    def check_new_best_score(self, new_score: float) -> None:
        """
        Check if the new score is the best score and update the best score if necessary.
        
        As these classes are specific to segmentation, the best score is the highest dice score 
        of the foreground classes.

        Parameters
        ----------
        new_score : float
            New score to be checked.
        """
        if self.model_selection_score < new_score:
            self.model_selection_score = new_score

            self._best_state = self.current_state

    def get_loss(self, loss_name: str) -> float | list:
        """
        Retrieve the loss values for a given loss name.

        Parameters
        ----------
        loss_name : str
            The name of the loss to retrieve.

        Returns
        -------
        float or list
            Loss values for the given loss name.
        """
        return self.tta_losses[loss_name]

    def get_score(self, score_name: str) -> float | list:
        """
        Retrieve the latest score for a given score name.

        Parameters
        ----------
        score_name : str
            The name of the score to retrieve.

        Returns
        -------
        float or list
            The latest score value for the given score name.
        """
        return self.test_scores[score_name]

    def get_score_as_df(self, score_name) -> pd.DataFrame:
        """
        Convert the test_scores dictionary to a pandas DataFrame.

        Parameters
        ----------
        score_name : str
            The name of the score to retrieve.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with the test_scores.
        """
        df = pd.DataFrame.from_dict(self.get_score(score_name),
                                    orient='index')
        
        if isinstance(df.iloc[0, 0],list):
            # Expand the list into columns
            df = pd.DataFrame(df[0].tolist(), index=df.index)
        
        return df
    
    def get_loss_as_df(self, loss_name) -> pd.DataFrame:
        """
        Convert the tta_losses dictionary to a pandas DataFrame.

        Parameters
        ----------
        loss_name : str
            The name of the loss to retrieve.

        Returns
        -------
        pd.DataFrame
            DataFrame with the tta_losses
        """
        df = pd.DataFrame.from_dict(self.get_loss(loss_name),
                                    orient='index')
        if isinstance(df.iloc[0, 0], list):
            # Expand the list into columns
            df = pd.DataFrame(df[0].tolist(), index=df.index)
        
        return df
    
    def get_all_test_scores_as_df(self, name_contains: Optional[str] = None) -> dict[pd.DataFrame]:
        """
        Convert all test_scores to pandas DataFrames.

        Parameters
        ----------
        name_contains : str, optional
            If provided, only return scores whose names contain this string.

        Returns
        -------
        dict of pd.DataFrame
            DataFrames with the test_scores.
        """
        if name_contains is not None:
            return {score_name: self.get_score_as_df(score_name)
                    for score_name in self.test_scores.keys()
                    if name_contains in score_name}
        else:
            return {score_name: self.get_score_as_df(score_name)
                    for score_name in self.test_scores.keys()}
    
    def store_test_scores_in_dir(
            self,
            output_dir: str, file_name_prefix: str,
            reduce: Literal['mean_accross_iterations', 'mean', None] = None) -> None:
        """
        Store the test scores in a directory as CSV files.

        Parameters
        ----------
        output_dir : str
            Directory to store the test scores.
        file_name_prefix : str
            Prefix for the file names.
        """
        os.makedirs(output_dir, exist_ok=True)
        for score_name, score_df in self.get_all_test_scores_as_df().items():
            # replace '/' with '_'
            score_name = score_name.replace('/', '__')
            filepath = os.path.join(output_dir, f'{file_name_prefix}_{score_name}.csv')
            
            if reduce == 'mean_accross_iterations':
                score_df = score_df.mean(axis=1)
            elif reduce == 'mean':
                score_df = score_df.values.mean()
            elif reduce is not None:
                raise ValueError(f"Unknown reduction method: {reduce}")
            
            score_df.to_csv(filepath, index=False, header=False)
            
    def get_all_losses_as_df(self, name_contains: Optional[str]) -> tuple[pd.DataFrame]:
        """
        Convert all tta_losses to pandas DataFrames.

        Returns
        -------
        tuple[pd.DataFrame]
            DataFrames with the tta_losses.
        """
        if name_contains is not None:
            return {loss_name: self.get_loss_as_df(loss_name)
                    for loss_name in self.tta_losses.keys()
                    if name_contains in loss_name}
        else:
            return {loss_name: self.get_loss_as_df(loss_name)
                    for loss_name in self.tta_losses.keys()}
    
class BaseTTA(TTAInterface):
    """
    Base class for Test-Time Adaptation (TTA) that implements the TTAInterface.
    
    Attributes
    ----------
    _adapted : bool
        Indicates whether the model has been adapted.
    """
    
    def tta(self, x: torch.Tensor) -> None:
        """
        Perform test-time adaptation on the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data for TTA.
        """
        raise NotImplementedError("The method 'tta' is not implemented.")
    
    def predict(
            self,
            x: torch.Tensor | DataLoader,
            output_vol_format: Literal['DCHW', '1CDHW'] = '1CDHW',
            **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions on the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data for prediction.
        """
        raise NotImplementedError("The method 'predict' is not implemented.")
    
    @torch.inference_mode()
    def evaluate(
        self,
        x_preprocessed: torch.Tensor, 
        y_original_gt: torch.Tensor,
        n_classes: int,
        preprocessed_pix_size: tuple[float, ...],
        gt_pix_size: tuple[float, ...], 
        metrics: dict[str, callable] = dice_score_fn_dict,
        output_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        store_visualization: bool = False,
        save_predicted_vol_as_nifti: bool = False,
        x_original: Optional[torch.Tensor] = None,
        other_volumes_to_visualize: Optional[dict[str, torch.Tensor]] = None,
        slice_vols_for_viz: Optional[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = None,
        **kwargs
        ) -> dict[str, float]:
        """
        Evaluate the model on the input data and labels.

        Parameters
        ----------
        x_preprocessed : torch.Tensor
            Input image at the preprocessing resolution. In NCDHW format.
        y_gt : torch.Tensor
            Ground truth segmentation labels at the original resolution. In NCDHW format.
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
        other_volumes_to_visualize : dict[str, dict], optional
            Additional volumes to include in the visualization. Volumes in each dict in NCDHW format.
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
        # Put all the necessary modules in evaluation mode
        self._evaluation_mode()

        # Predict segmentation for x_preprocessed
        x_norm, y_pred, _ = self.predict(x_preprocessed, output_vol_format='1CDHW', **kwargs)

        assert all(vol.ndim == 5 for vol in (x_norm, y_pred, y_original_gt)), "The volumes must have 5 dimensions (NCDHW)."

        y_original_gt = y_original_gt.to(y_pred.device)

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
        if store_visualization:
            assert output_dir is not None, "The output directory must be provided to store visualizations."
            assert file_name is not None, "The file name must be provided to store visualizations."

            # Preprocess volumes for visualization
            x_original, x_norm = _preprocess_volumes_for_viz(
                x_original, x_norm, slice_idxs=slice_vols_for_viz, convert_to_categorical=False)        
            
            y_original_gt, y_pred = _preprocess_volumes_for_viz(
                y_original_gt, y_pred, slice_idxs=slice_vols_for_viz, convert_to_categorical=True)

            # Resize other volumes to the original resolution and preprocess them
            other_vols_preprocessed = {}
            if other_volumes_to_visualize is not None:
                for vol_name, vol_dict in other_volumes_to_visualize.items():
                    other_vols_preprocessed[vol_name] = resize_volume(
                        vol_dict['vol'],
                        current_pix_size=vol_dict['current_pix_size'],
                        target_pix_size=vol_dict['target_pix_size'],
                        target_img_size=vol_dict['target_img_size'], 
                        mode='trilinear',
                        only_inplane_resample=False     
                    )

                    convert_to_categorical = vol_name[0].lower() == 'y'
                    slice_idxs = slice_vols_for_viz if 'slice_idxs' in vol_dict else None
                    other_vols_preprocessed[vol_name] = _preprocess_volumes_for_viz(
                        other_vols_preprocessed[vol_name], 
                        slice_idxs=slice_idxs, 
                        convert_to_categorical=convert_to_categorical)[0]
                   
            export_images(
                x_original,
                x_norm,
                y_original_gt,
                y_pred,
                output_dir=os.path.join(output_dir, 'segmentation'),
                image_name=file_name + '.png',
                n_classes=n_classes,
                **other_vols_preprocessed
            )

        # Save the predicted volume as a NIfTI file
        if save_predicted_vol_as_nifti:
            assert output_dir is not None, "The output directory must be provided to save the predicted volume."
            assert file_name is not None, "The file name must be provided to save the predicted volume."

            save_nii_image(
                dir=os.path.join(output_dir, 'segmentation_nifti'),
                filename=file_name + '.nii.gz',
                image=y_pred.detach().cpu().numpy().astype('uint8'),
            )

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
    
    def load_best_state(self) -> None:
        """
        Load the best state of the model.
        """
        raise NotImplementedError("The method 'load_best_state' is not implemented.")
    
    def _evaluation_mode(self) -> None:
        """
        Set the model to evaluation mode.
        """
        raise NotImplementedError("The method '_evaluation_mode' is not implemented.")
    
    def _tta_fit_mode(self) -> None:
        """
        Set the model to TTA fit mode.
        """
        raise NotImplementedError("The method '_tta_fit_mode' is not implemented.")