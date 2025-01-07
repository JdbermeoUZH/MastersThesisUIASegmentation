from typing import Optional

import torch
import numpy as np
import joblib
from sklearn.decomposition import IncrementalPCA as IncrementalPCASklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .BasePCA import BasePCA
from .utils import flatten_pixels, unflatten_pixels
from tta_uia_segmentation.src.utils.utils import torch_to_numpy, default


import time

def time_it(func, *args, **kwargs):
    """
    Wraps a function call and measures its execution time.
    
    Parameters:
        func (callable): The function to be timed.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        result: The result of the function call.
        elapsed_time: The time taken to execute the function in seconds.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    print(f"Execution time for {func.__name__}: {elapsed_time:.4f} seconds")
    return result, elapsed_time

@torch.jit.script
def _compiled_to_pcs_with_torch(
    x: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    principal_components: torch.Tensor,
) -> torch.Tensor:
    # Normalize with the scalers information
    x = (x - mean) / scale

    return torch.mm(x, principal_components.t())
class IncrementalPCA(BasePCA):
    def __init__(self, n_components: Optional[int] = None):
        self._pca = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", IncrementalPCASklearn(n_components=n_components)),
            ]
        )
        self._principal_components: Optional[torch.Tensor] = None
        self._mean: Optional[torch.Tensor] = None
        self._scale: Optional[torch.Tensor] = None
        self._use_torch = False
        self._device = None

    def fit(self, x: torch.Tensor):
        x_np = torch_to_numpy(x)
        self._pca.fit(x_np)

    def partial_fit(self, x: torch.Tensor):
        x_np = torch_to_numpy(x)

        # Partial fit of the scaler
        self.scaler.partial_fit(x_np)

        # Normalize the data
        x_np = self.scaler.transform(x_np)

        # Partial fit of the PCA
        self.ipca.partial_fit(x_np)

    @property
    def n_components(self) -> Optional[int]:
        return self.ipca.n_components_

    @n_components.setter
    def n_components(self, value: Optional[int]):
        self.ipca.set_params(n_components=value)
        self.ipca.n_components_ = value

        if value is not None:
            assert (
                value <= self.ipca.n_components_
            ), f"Cannot set n_components to {value} as it is greater than the current number of components {self.ipca.n_components_}"

        if self._use_torch and isinstance(value, int):
            if (
                isinstance(self._principal_components, torch.Tensor)
                and len(self._principal_components) > value
            ):
                self._principal_components = self._principal_components[:value]
            else:
                self._principal_components = (
                    torch.from_numpy(self.ipca.components_)
                    .float()
                    .to(self._device)[:value]
                )

    def to_pcs(self, x: torch.Tensor) -> torch.Tensor:

        if not self._use_torch:
            device, dtype = x.device, x.dtype
            x_np = torch_to_numpy(x)
            return torch.tensor(
                self._pca.transform(x_np), 
                device=device,
                dtype=dtype
            )[:, : self.n_components]
        else:
            assert isinstance(
                self._principal_components, torch.Tensor
            ), "The principal components must be a torch.Tensor to use torch"

            return _compiled_to_pcs_with_torch(
                x,
                self._mean,
                self._scale,
                self._principal_components,
                )

    def from_pcs(self, z: torch.Tensor) -> torch.Tensor:
        if not self._use_torch:
            if self.n_components != len(self.ipca.components_):
                raise ValueError(
                    f"Inverting the transformation with a different number of " 
                    + f" components than the fitted model is not supported.\n"
                    + f" The model was fitted with {self.n_components} components,"
                    + f" but you are trying to invert with {len(self.ipca.components_)} components"
                )

            z_np = torch_to_numpy(z)
            return torch.from_numpy(self._pca.inverse_transform(z_np))
        else:
            assert isinstance(
                self._principal_components, torch.Tensor
            ), "The principal components must be a torch.Tensor to use torch"

            x_recon = torch.mm(z.to(self._device), self._principal_components)

            # Denormalize the data
            return x_recon * self._scale + self._mean

    def save(self, path):
        joblib.dump(self._pca, path)

    def load(self, path):
        self._pca = joblib.load(path)

    def serialize_to_dict(self) -> dict:
        """
        Serializes a scikit-learn pipeline with a StandardScaler and IncrementalPCA.

        Returns:
            dict: A dictionary containing the serialized state of the pipeline.
        """
        
        serialized_pipeline = {}
        for name, step in self._pca.steps:
            if isinstance(step, StandardScaler):
                serialized_pipeline[name] = {
                    "type": "StandardScaler",
                    "mean_": step.mean_.tolist() if hasattr(step, "mean_") else None,
                    "scale_": step.scale_.tolist() if hasattr(step, "scale_") else None,
                    "var_": step.var_.tolist() if hasattr(step, "var_") else None,
                    "n_samples_seen_": int(step.n_samples_seen_) if hasattr(step, "n_samples_seen_") else None
                }
            elif isinstance(step, IncrementalPCA):
                serialized_pipeline[name] = {
                    "type": "IncrementalPCA",
                    "hyperparameters": step.get_params(),
                    "fitted_parameters": {
                        "components_": step.components_.tolist() if hasattr(step, "components_") else None,
                        "explained_variance_": step.explained_variance_.tolist() if hasattr(step, "explained_variance_") else None,
                        "explained_variance_ratio_": step.explained_variance_ratio_.tolist() if hasattr(step, "explained_variance_ratio_") else None,
                        "singular_values_": step.singular_values_.tolist() if hasattr(step, "singular_values_") else None,
                        "mean_": step.mean_.tolist() if hasattr(step, "mean_") else None,
                        "n_components_": step.n_components_ if hasattr(step, "n_components_") else None,
                        "n_samples_seen_": int(step.n_samples_seen_) if hasattr(step, "n_samples_seen_") else None
                    }
                }
            else:
                raise ValueError(f"Unsupported step type: {type(step)} in pipeline.")

        # Add the other class attributes
        serialized_pipeline["n_components"] = self.n_components
        serialized_pipeline["principal_components"] = self._principal_components.tolist() if self._principal_components is not None else None
        serialized_pipeline["mean"] = self._mean.tolist() if self._mean is not None else None
        serialized_pipeline["scale"] = self._scale.tolist() if self._scale is not None else None
        serialized_pipeline["use_torch"] = self._use_torch
        serialized_pipeline["device"] = self._device

        return serialized_pipeline

    @classmethod
    def load_pipeline_from_dict(cls, serialized_pipeline: dict) -> "IncrementalPCA":
        """
        Loads a scikit-learn pipeline from a serialized state dictionary.

        Parameters:
            serialized_pipeline (dict): The serialized state of the pipeline.
        """

        self = cls()
        
        for name, step_state in serialized_pipeline.items():
            step = self._pca.named_steps[name]

            if step_state["type"] == "StandardScaler":
                if step_state.get("mean_") is not None:
                    step.mean_ = np.array(step_state["mean_"])
                if step_state.get("scale_") is not None:
                    step.scale_ = np.array(step_state["scale_"])
                if step_state.get("var_") is not None:
                    step.var_ = np.array(step_state["var_"])
                if step_state.get("n_samples_seen_") is not None:
                    step.n_samples_seen_ = step_state["n_samples_seen_"]
            
            elif step_state["type"] == "IncrementalPCA":
                step.set_params(**step_state["hyperparameters"])
                fitted_params = step_state["fitted_parameters"]
                for param, value in fitted_params.items():
                    if value is not None:
                        setattr(
                            step,
                            param, np.array(value) 
                            if isinstance(value, list) else value
                        )
            else:
                raise ValueError(f"Unsupported step type: {step_state['type']} in pipeline.")

        # Add the other class attributes
        self.n_components = serialized_pipeline["n_components"]
        self._use_torch = serialized_pipeline["use_torch"]
        self._device = serialized_pipeline["device"]

        if self._use_torch:
            self._principal_components = torch.Tensor(serialized_pipeline["principal_components"], device=self._device)
            self._mean = torch.Tensor(serialized_pipeline["mean"], device=self._device)
            self._scale = torch.Tensor(serialized_pipeline["scale"], device=self._device)

        return self

    def reconstruct(
        self, x: torch.Tensor, num_components: Optional[int] = None
    ) -> torch.Tensor:
        if not self._use_torch and self.n_components != len(self.ipca.components_):
            raise ValueError(
                f"Reconstructing the data with a different number of " 
                + f" components than the fitted model is not supported.\n"
                + f" The model was fitted with {self.n_components} components,"
                + f" but you are trying to reconstruct with {len(self.ipca.components_)} components"
            )

        old_n_components = self.n_components

        num_components = default(num_components, self.n_components)
        self.n_components = num_components

        x_recon = self.from_pcs(self.to_pcs(x))
        self.n_components = old_n_components

        return x_recon

    def num_components_to_keep(self, variance_to_keep: float) -> int:
        assert (
            self._is_fitted()
        ), "The PCA model must be fitted before calling this method"

        cumulative_variance = np.cumsum(self.ipca.explained_variance_ratio_)

        # Find the minimum number of components
        return int(np.searchsorted(cumulative_variance, variance_to_keep) + 1)

    def img_to_pcs(self, x: torch.Tensor) -> torch.Tensor:
        x, (b, h, w) = flatten_pixels(x)

        # Map the data to the PCA space
        x_pcs = self.to_pcs(x)

        # Unflatten the data to the original shape
        return unflatten_pixels(x_pcs, b, h, w)

    def img_from_pcs(self, z: torch.Tensor) -> torch.Tensor:
        z, (b, h, w) = flatten_pixels(z)

        # Map the data back to the original space
        z_recon = self.from_pcs(z)

        # Unflatten the data to the original shape
        return unflatten_pixels(z_recon, b, h, w)

    def img_reconstruct(
        self, x: torch.Tensor, num_components: Optional[int] = None
    ) -> torch.Tensor:
        x, (b, h, w) = flatten_pixels(x)

        # Reconstruct the image
        x_recon = self.reconstruct(x, num_components)

        # Unflatten the data to the original shape
        return unflatten_pixels(x_recon, b, h, w)

    def to_device(self, device):
        self._device = device
        self.use_torch = True

        assert isinstance(
            self._principal_components, torch.Tensor
        ), "The principal components must be initialized as a torch.Tensor"
        assert isinstance(
            self._mean, torch.Tensor
        ), "The mean must be initialized as a torch.Tensor"
        assert isinstance(
            self._scale, torch.Tensor
        ), "The scale must be initialized as a torch.Tensor"
        self._principal_components = self._principal_components.to(device)
        self._mean = self._mean.to(device)
        self._scale = self._scale.to(device)

    @property
    def device(self):
        return self._device

    @property
    def scaler(self) -> StandardScaler:
        return self._pca.named_steps["scaler"]

    @property
    def ipca(self) -> IncrementalPCASklearn:
        return self._pca.named_steps["pca"]

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.ipca.explained_variance_ratio_

    @property
    def use_torch(self) -> bool:
        return self._use_torch

    @use_torch.setter
    def use_torch(self, value: bool):
        self._use_torch = value

        if self._use_torch:
            assert (
                self._is_fitted()
            ), "The PCA model must be fitted to have PCs to be used in torch to begin with"
            self._principal_components = torch.from_numpy(self.ipca.components_).float()
            self._mean = torch.Tensor(self.scaler.mean_) + torch.Tensor(
                self.ipca.mean_
            ) * torch.Tensor(self.scaler.scale_)
            self._scale = torch.Tensor(self.scaler.scale_)

            # Make sure the number of components is set correctly
            self.n_components = self.n_components

    def _is_fitted(self):
        return hasattr(self.ipca, "components_")

    @classmethod
    def load_pca(cls, path: str) -> "IncrementalPCA":
        pca = cls()
        pca.load(path)
        return pca

    def __print__(self):
        return f"IncrementalPCA(n_components={self.n_components})"
