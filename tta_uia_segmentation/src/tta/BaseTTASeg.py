import os
import copy
from functools import partial
from typing import Optional, Literal, Union, Callable
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field, asdict

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import f1_score

from tta_uia_segmentation.src.models import BaseSeg
from tta_uia_segmentation.src.tta.TTAInterface import TTAInterface, TTAStateInterface
from tta_uia_segmentation.src.utils.loss import (
    dice_score,
    onehot_to_class,
    onehot_to_class,
)
from tta_uia_segmentation.src.utils.utils import (
    default,
    resize_volume,
    torch_to_numpy,
    from_DCHW_to_NCDHW,
    generate_2D_dl_for_vol,
)
from tta_uia_segmentation.src.utils.io import save_nii_image, write_to_csv
from tta_uia_segmentation.src.utils.visualization import export_images


# Default metric functions for evaluation
EVAL_METRICS = {
    "dice_score_all_classes": lambda y_pred, y_gt: dice_score(
        y_pred,
        y_gt,
        soft=False,
        reduction="none",
        bg_channel=0,
        smooth=0,
        epsilon=0,
    ),
    "dice_score_fg_classes": lambda y_pred, y_gt: dice_score(
        y_pred,
        y_gt,
        soft=False,
        reduction="none",
        foreground_only=True,
        bg_channel=0,
        smooth=0,
        epsilon=0,
    ),
    "dice_score_fg_classes_sklearn": lambda y_pred, y_gt: f1_score(
        torch_to_numpy(onehot_to_class(y_gt)).flatten(),
        torch_to_numpy(onehot_to_class(y_pred)).flatten(),
        average=None,
    )[1:],
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
            slice_list = [slice(start, end) for start, end in slice_idxs]
            vol = vol[..., slice_list[0], slice_list[1], slice_list[2]]

        # Convert to categorical
        if convert_to_categorical:
            vol = onehot_to_class(vol)

        # Convert from NCDHW to NDCHW
        vol = vol.permute(0, 2, 1, 3, 4)

        # Squeeze vol
        vol = vol.squeeze()

        preprocessed_vols.append(vol)

    return preprocessed_vols


def _visualize_predictions(
    x_original: torch.Tensor,
    interm_outs: dict[str, torch.Tensor],
    y_original_gt: torch.Tensor,
    y_pred: torch.Tensor,
    output_dir: str,
    file_name: str,
    output_dir_suffix: str = "",
    other_volumes_to_visualize: Optional[dict[str, torch.Tensor]] = None,
    save_predicted_vol_as_nifti: bool = False,
    slice_idxs: Optional[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ] = None,
) -> None:
    """
    Visualize the predictions of the model.

    Parameters
    ----------
    x_original : torch.Tensor
        Original input image.
    interm_outs : dict[str, torch.Tensor],
        Intermediate outputs of the forward pass to visualize.
    y_original_gt : torch.Tensor
        Ground truth segmentation.
    y_pred : torch.Tensor
        Predicted segmentation.
    output_dir : str
        Directory to store the visualizations.
    file_name : str
        Base name for the output files.
    other_volumes_to_visualize : dict[str, torch.Tensor], optional
        Additional volumes to visualize.
    save_predicted_vol_as_nifti : bool, optional
        Whether to save the predicted volume as a NIfTI file.
    slice_idxs : tuple[tuple[int, int], tuple[int, int], tuple[int, int]], optional
        Slicing indices for the volumes.
    """
    # Get the number of classes
    n_classes = y_pred.shape[1]

    # Preprocess volumes for visualization
    preproc_img_vol_viz = lambda vol: _preprocess_volumes_for_viz(
        vol,
        slice_idxs=slice_idxs,
        convert_to_categorical=False,
    )[0]

    x_original = preproc_img_vol_viz(x_original)

    interm_outs = {key: preproc_img_vol_viz(val) for key, val in interm_outs.items()}

    y_original_gt, y_pred = _preprocess_volumes_for_viz(
        y_original_gt, y_pred, slice_idxs=slice_idxs, convert_to_categorical=True
    )

    # Resize other volumes to the original resolution and preprocess them
    other_vols_preprocessed = {}
    if other_volumes_to_visualize is not None:
        for vol_name, vol_dict in other_volumes_to_visualize.items():
            other_vols_preprocessed[vol_name] = resize_volume(
                vol_dict["vol"],
                current_pix_size=vol_dict["current_pix_size"],
                target_pix_size=vol_dict["target_pix_size"],
                target_img_size=vol_dict["target_img_size"],
                mode="trilinear",
                only_inplane_resample=False,
            )

            convert_to_categorical = vol_name[0].lower() == "y"
            slice_idxs = slice_idxs if "slice_idxs" in vol_dict else None
            other_vols_preprocessed[vol_name] = _preprocess_volumes_for_viz(
                other_vols_preprocessed[vol_name],
                slice_idxs=slice_idxs,
                convert_to_categorical=convert_to_categorical,
            )[0]

    export_images(
        x_original,
        interm_outs,
        y_original_gt,
        y_pred,
        output_dir=os.path.join(output_dir, "segmentation" + output_dir_suffix),
        image_name=file_name + ".png",
        n_classes=n_classes,
        **other_vols_preprocessed,
    )

    # Save the predicted volume as a NIfTI file
    if save_predicted_vol_as_nifti:
        assert (
            output_dir is not None
        ), "The output directory must be provided to save the predicted volume."
        assert (
            file_name is not None
        ), "The file name must be provided to save the predicted volume."

        vols_to_save = {
            "x_original": x_original,
            "y_pred": y_pred,
            "y_gt": y_original_gt,
        }

        for vol_name, vol in vols_to_save.items():
            save_nii_image(
                dir=os.path.join(
                    output_dir,
                    "segmentation_nifti" + output_dir_suffix,
                    f"vol_{vol_name}",
                ),
                filename=file_name + vol_name + ".nii.gz",
                image=vol.detach().cpu().numpy().astype("uint8"),
            )


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
    model_selection_score: float = float("-inf")
    tta_losses: dict[str | int, OrderedDict[int, list | float]] = field(
        default_factory=dict
    )
    test_scores: dict[str | int, OrderedDict[int, list | float]] = field(
        default_factory=dict
    )

    _create_initial_state: bool = field(default=True)
    _initial_state: Optional[Union["BaseTTAState", dict]] = field(default=None)
    _best_state: Optional[Union["BaseTTAState", dict]] = field(default=None)

    def reset(self) -> None:
        """
        Reset the state of the class to the initial state.
        """
        self.reset_to_state(self._initial_state)
        self._initial_state = self.current_state

    def reset_to_state(self, state: "BaseTTAState") -> None:
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

        # Convert the _initial_state and _best_state attributes to TTAState objects
        if isinstance(self._initial_state, dict):
            self._initial_state = self.__class__(**self._initial_state)

        if isinstance(self._best_state, dict):
            self._best_state = self.__class__(**self._best_state)

    @property
    def current_state(self) -> "BaseTTAState":
        """
        Get the current state of the BaseTTAState.

        Returns
        -------
        BaseTTAState
            The current state.
        """

        # Get the dict of the remaining attributes and deep copy them
        current_state_dict = asdict(self)
        current_state_dict["_create_initial_state"] = False

        current_state_dict = {
            key: copy.deepcopy(value) for key, value in current_state_dict.items()
        }

        # Create a new BaseTTAState object with the copied attributes
        current_state = BaseTTAState(**current_state_dict)

        return current_state

    @property
    def initial_state(self) -> "BaseTTAState":
        """
        Get the initial state of the BaseTTAState.

        Returns
        -------
        BaseTTAState
            The initial state.
        """
        return self._initial_state

    @property
    def best_state(self) -> "BaseTTAState":
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
            self._best_state._best_state = None

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
        df = pd.DataFrame.from_dict(self.get_score(score_name), orient="index")

        if isinstance(df.iloc[0, 0], list):
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
        df = pd.DataFrame.from_dict(self.get_loss(loss_name), orient="index")
        if isinstance(df.iloc[0, 0], list):
            # Expand the list into columns
            df = pd.DataFrame(df[0].tolist(), index=df.index)

        return df

    def get_all_test_scores_as_df(
        self, name_contains: Optional[str] = None
    ) -> dict[pd.DataFrame]:
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
            return {
                score_name: self.get_score_as_df(score_name)
                for score_name in self.test_scores.keys()
                if name_contains in score_name
            }
        else:
            return {
                score_name: self.get_score_as_df(score_name)
                for score_name in self.test_scores.keys()
            }

    def store_test_scores_in_dir(
        self,
        output_dir: str,
        file_name_prefix: str,
        reduce: Literal["mean_accross_iterations", "mean", None] = None,
    ) -> None:
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
            score_name = score_name.replace("/", "__")
            filepath = os.path.join(output_dir, f"{file_name_prefix}_{score_name}.csv")

            if reduce == "mean_accross_iterations":
                score_df = score_df.mean(axis=1)
            elif reduce == "mean":
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
            return {
                loss_name: self.get_loss_as_df(loss_name)
                for loss_name in self.tta_losses.keys()
                if name_contains in loss_name
            }
        else:
            return {
                loss_name: self.get_loss_as_df(loss_name)
                for loss_name in self.tta_losses.keys()
            }


class BaseTTASeg(TTAInterface):
    """
    Base class for Test-Time Adaptation (TTA) that implements the TTAInterface.

    Attributes
    ----------
    _adapted : bool
        Indicates whether the model has been adapted.
    """

    def __init__(
        self,
        seg: BaseSeg,
        n_classes: int,
        classes_of_interest: Optional[tuple[int | str, ...]] = tuple(),
        eval_metrics: dict[str, Callable] = EVAL_METRICS,
        viz_interm_outs: tuple[str, ...] = tuple(),
        wandb_log: bool = False,
        device: str | torch.device = "cuda",
    ):
        self._state = BaseTTAState()

        self._seg = seg

        # Information about the problem
        self._n_classes = n_classes
        self._classes_of_interest = (
            (classes_of_interest,)
            if isinstance(classes_of_interest, int)
            else classes_of_interest
        )

        # Logging settings
        self._wandb_log = wandb_log

        # Evaluation metrics
        self._eval_metrics = eval_metrics

        # Visualization settings
        self._viz_interm_outs = viz_interm_outs

        # Device
        self._device = device

    @torch.inference_mode()
    def predict(
        self,
        x: torch.Tensor | DataLoader,
        include_interm_outs: bool = True,
        output_vol_format: Literal["DCHW", "1CDHW"] = "1CDHW",
        **preprocess_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        """
        Make predictions on the input data.

        Parameters
        ----------
        x : torch.Tensor | DataLoader
            Input volume for which to make a prediction.
        output_vol_format : Literal['DCHW', '1CDHW'], optional
            Format of the output volume, by default '1CDHW'.
        """

        y_mask_list, y_logits_list = [], []
        interm_outs_dict = defaultdict(list)

        # Convert the input volume to a DataLoader (DCHW) if necessary
        if isinstance(x, torch.Tensor):
            # Verify that the input volume has the correct number of dimensions
            assert x.ndim == 5, "The input volume must have 5 dimensions (NCDHW)."

            # Create a DataLoader for the input volume
            x = generate_2D_dl_for_vol(
                x, 
                batch_size=preprocess_kwargs['batch_size'],
                num_workers=preprocess_kwargs['num_workers']
            )

        for x_b, *_ in x:
            x_b = x_b.to(self._device).float()
            y_mask, y_logits, interm_outs = self._seg.predict(
                x_b,
                include_interm_outs=include_interm_outs,
                **preprocess_kwargs,
            )
            y_mask_list.append(y_mask)
            y_logits_list.append(y_logits)

            # Append intermediate outputs
            if interm_outs is not None:
                for key in self._viz_interm_outs:
                    assert (
                        key in interm_outs
                    ), f"{key} is not an intermediate output of the model's forward pass."
                    interm_outs_dict[key].append(interm_outs[key])

        # Concatenate the masks and logits
        y_mask = torch.cat(y_mask_list)
        y_logits = torch.cat(y_logits_list)
        interm_outs = {key: torch.cat(val) for key, val in interm_outs_dict.items()}

        # Rearrange the output volume format
        if output_vol_format == "DCHW":
            pass

        elif output_vol_format == "1CDHW":
            y_mask = from_DCHW_to_NCDHW(y_mask)
            y_logits = from_DCHW_to_NCDHW(y_logits)
            interm_outs = {
                key: from_DCHW_to_NCDHW(val) for key, val in interm_outs.items()
            }

        return y_mask, y_logits, interm_outs

    @torch.inference_mode()
    def evaluate(
        self,
        x_preprocessed: torch.Tensor,
        y_original_gt: torch.Tensor,
        preprocessed_pix_size: tuple[float, ...],
        gt_pix_size: tuple[float, ...],
        metrics: Optional[dict[str, Callable]] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        classes_of_interest: tuple[int] = tuple(),
        output_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        store_visualization: bool = False,
        save_predicted_vol_as_nifti: bool = False,
        x_original: Optional[torch.Tensor] = None,
        other_volumes_to_visualize: Optional[dict[str, torch.Tensor]] = None,
        slice_vols_for_viz: Optional[
            tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        ] = None,
        predict_kwargs: dict = dict(),
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
        metrics : dict[str, Callable], optional
            Dictionary of metric functions to evaluate the model's performance.
        classes_of_interest : tuple[int], optional
            List of classes to visualize.
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
        predict_kwargs: dict, optional
            Additional keyword arguments to pass to the predict method.

        Returns
        -------
        dict[str, float]
            Dictionary of metric names and their corresponding values.
        """
        # Put all the necessary modules in evaluation mode
        self._evaluation_mode()

        # Set default values of arguments if necessary
        metrics = default(metrics, self._eval_metrics)
        classes_of_interest = default(classes_of_interest, self._classes_of_interest)

        # Predict segmentation for x_preprocessed
        # :===================================================================:

        # Generate a DataLoader if a single volume is provided
        if isinstance(x_preprocessed, torch.Tensor):
            x_preprocessed = generate_2D_dl_for_vol(
                x_preprocessed, batch_size=batch_size, num_workers=num_workers
            )

        y_pred, _, interm_outs = self.predict(
            x_preprocessed, output_vol_format="1CDHW", **predict_kwargs
        )

        assert all(
            vol.ndim == 5 for vol in (y_pred, *interm_outs.values())
        ), "The volumes must have 5 dimensions (NCDHW)."

        # Resize y_pred and interm_outs to the original resolution
        # :===================================================================:
        resize_to_original = partial(
            resize_volume,
            current_pix_size=preprocessed_pix_size,
            target_pix_size=gt_pix_size,
            target_img_size=None,  # We assume no padding or cropping is needed to match image sizes
            mode="bilinear",
            only_inplane_resample=True,
        )

        y_pred = resize_to_original(y_pred)
        interm_outs = {key: resize_to_original(val) for key, val in interm_outs.items()}

        # Measure the performance of the model
        # :===================================================================:
        y_original_gt = y_original_gt.to(y_pred.device)

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
        # :===================================================================:
        if store_visualization:
            assert (
                output_dir is not None
            ), "The output directory must be provided to store visualizations."
            assert (
                file_name is not None
            ), "The file name must be provided to store visualizations."

            _visualize_predictions(
                x_original=x_original,
                interm_outs=interm_outs,
                y_original_gt=y_original_gt,
                y_pred=y_pred,
                output_dir=output_dir,
                file_name=file_name,
                other_volumes_to_visualize=other_volumes_to_visualize,
                save_predicted_vol_as_nifti=save_predicted_vol_as_nifti,
                slice_idxs=slice_vols_for_viz,
            )

            if len(classes_of_interest) > 0:
                # Visualize the classes of interest
                classes_of_interest_str = [str(cls) for cls in classes_of_interest]
                output_dir_suffix = "_classes_of_interest_" + "_".join(
                    classes_of_interest_str
                )

                _visualize_predictions(
                    x_original=x_original,
                    interm_outs=interm_outs,
                    y_original_gt=y_original_gt[:, [0] + classes_of_interest],
                    y_pred=y_pred[:, [0] + classes_of_interest],
                    output_dir=output_dir,
                    file_name=file_name,
                    output_dir_suffix=output_dir_suffix,
                    other_volumes_to_visualize=other_volumes_to_visualize,
                    save_predicted_vol_as_nifti=save_predicted_vol_as_nifti,
                    slice_idxs=slice_vols_for_viz,
                )

        return metrics_values

    def _evaluation_mode(self) -> None:
        """
        Set the model to evaluation mode.
        """
        self._seg.eval_mode()

    def write_current_dice_scores(
        self, vol_idx: int, logdir: str, dataset_name: str, iteration_type: str
    ):
        """
        Write dice scores to CSV files.

        Parameters
        ----------
        logdir : str
            Directory to save the CSV files.
        dataset_name : str
            Name of the dataset.
        iteration_type : str
            Type of iteration (e.g., 'last_iteration', 'best_scoring_iteration').

        """
        test_scores_fg = self._state.get_all_test_scores_as_df(
            name_contains="dice_score"
        )

        for score_name, score_df in test_scores_fg.items():
            score_sub_dir = os.path.dirname(score_name)
            score_name = os.path.basename(score_name)
            file_name = f"{score_name}_{dataset_name}_{iteration_type}.csv"
            last_scores = score_df.iloc[-1].values
            write_to_csv(
                os.path.join(logdir, score_sub_dir, file_name),
                np.hstack([[[f"volume_{vol_idx:02d}"]], [last_scores]]),
                mode="a",
            )

            # Store the average
            file_name = "summary_per_vol" + file_name
            mean = np.mean(last_scores)
            std = np.std(last_scores)
            write_to_csv(
                os.path.join(logdir, score_sub_dir, file_name),
                np.array([[f"volume_{vol_idx:02d}", mean, std]]),
                mode="a",
            )

    def get_current_average_test_score(self, score_name: str) -> dict[str, float]:
        """
        Get the average test scores for the current iteration.

        Parameters
        ----------
        score_name_contains : str, optional
            If provided, only return scores whose names contain this string.

        Returns
        -------
        dict[str, float]
            Dictionary of average test scores.
        """
        test_scores_df = self._state.get_score_as_df(score_name)

        return np.mean(test_scores_df.iloc[-1].values)

    def get_current_test_score(
        self, score_name: str, class_idx: Optional[int] = None
    ) -> dict[str, float]:
        """
        Get the test scores for the current iteration.

        Parameters
        ----------
        score_name : str
            Name of the score to retrieve.

        Returns
        -------
        dict[str, float]
            Dictionary of test scores.
        """
        if class_idx is None:
            return self._state.get_score_as_df(score_name).iloc[-1].values
        else:
            return self._state.get_score_as_df(score_name).iloc[-1, class_idx]

    @property
    def state(self) -> BaseTTAState:
        return self._state

    def get_current_state(
        self,
        as_dict: bool = True,
        remove_initial_state: bool = True,
        remove_best_state: bool = True,
    ) -> BaseTTAState | dict:

        if as_dict:
            current_state = asdict(self._state)
            if remove_initial_state:
                current_state["_initial_state"] = None
            if remove_best_state:
                current_state["_best_state"] = None

        else:
            current_state = self._state
            if remove_initial_state or remove_best_state:
                current_state = current_state.current_state

                if remove_initial_state:
                    current_state._initial_state = None

                if remove_best_state:
                    current_state._best_state = None

        return current_state

    @property
    def best_state(self) -> BaseTTAState:
        return self._state.best_state

    def get_best_state(
        self,
        as_dict: bool = True,
        remove_initial_state: bool = True,
    ) -> BaseTTAState | dict:
        if as_dict:
            best_state = asdict(self._state.best_state)
            if remove_initial_state:
                best_state["_initial_state"] = None
        else:
            best_state = self._state.best_state
            if remove_initial_state:
                best_state = best_state.current_state  # create deep copy of best_state
                best_state._initial_state = None

        return best_state

    def load_state(self, path: str) -> None:
        """
        Load the state of the model from a file.

        Parameters
        ----------
        path : str
            Path to the file containing the model state.
        """
        state_dict = torch.load(path)
        self._seg.load_state_dict(state_dict["seg"])

    def save_state(self, path: str) -> None:
        """
        Save the state of the model to a file.

        Parameters
        ----------
        path : str
            Path to the file where the model state will be saved.
        """
        state_dict = {"seg": self._seg.state_dict(), "state": self._state}
        torch.save(state_dict, path)

    def reset_state(self) -> None:
        """
        Reset the state of the model.
        """
        self._state.reset()

    def load_best_state(self) -> None:
        """
        Load the best state of the model.
        """
        self._state.reset_to_state(self._state.best_state)

    @property
    def model_selection_score(self) -> float:
        return self._state.model_selection_score

    @property
    def current_iteration(self) -> int:
        return self._state.iteration

    def get_loss(self, loss_name: str) -> OrderedDict[int, float]:
        return self._state.get_loss(loss_name)

    def get_score(self, metric_name: str) -> OrderedDict[int, float]:
        return self._state.get_score(metric_name)

    def tta(self, x: torch.Tensor | DataLoader) -> None:
        """
        Perform test-time adaptation on the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data for TTA.
        """
        raise NotImplementedError("The method 'tta' is not implemented.")

    def _tta_fit_mode(self) -> None:
        """
        Set the model to TTA fit mode.
        """
        raise NotImplementedError("The method '_tta_fit_mode' is not implemented.")

    def evaluate_dataset(self, ds: Dataset) -> None:
        """
        Evaluate the model on a dataset.

        Parameters
        ----------
        ds : torch.utils.data.Dataset
            Dataset for evaluation.
        """
        raise NotImplementedError("The method 'evaluate_dataset' is not implemented.")
