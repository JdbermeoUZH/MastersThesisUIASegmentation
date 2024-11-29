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
            x_np = torch_to_numpy(x)
            return torch.from_numpy(self._pca.transform(x_np))[:, : self.n_components]
        else:
            assert isinstance(
                self._principal_components, torch.Tensor
            ), "The principal components must be a torch.Tensor to use torch"

            # Normalize with the scalers information
            x = (x - self._mean) / self._scale

            return torch.mm(x.to(self._device), self._principal_components.t())

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
