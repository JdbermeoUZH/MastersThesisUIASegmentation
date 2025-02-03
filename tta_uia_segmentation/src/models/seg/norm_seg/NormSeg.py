from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import Normalization
from .UNet import UNet
from ..BaseSeg import BaseSeg


class NormSeg(BaseSeg):
    def __init__(
        self,
        norm: Normalization | nn.Module,
        seg: UNet | nn.Module,
    ):
        super().__init__()

        self._norm = norm
        self._seg = seg

    def select_necessary_extra_inputs(self, extra_input_dict: dict[str, Any]) -> dict[str, Any]:
        return {}
    
    def _preprocess_x(
        self,
        x: torch.Tensor | list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Preprocess input tensor

        Returns:
        -------
        x_preproc : torch.Tensor
            Preprocessed input tensor.

        intermediate_outputs : dict[str, torch.Tensor]
            Dictionary containing intermediate outputs of preprocessing.
        """
        x_norm = self._norm(x)

        return x_norm, {"Normalized Image": x_norm}

    def _forward(
        self,
        x_preproc: torch.Tensor | list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model assuming a preprocessed input.

        Returns:
        --------
        y_mask : torch.Tensor
            Predicted segmentation mask.

        y_logits : torch.Tensor
            Predicted logits.
        """
        return self._seg(x_preproc)

    def checkpoint_as_dict(self, **kwargs) -> dict:
        """
        Returns the model checkpoint as a dictionary.
        """
        return dict(
            norm_state_dict=self.norm.state_dict(),
            seg_state_dict=self.seg.state_dict(),
            **kwargs,
        )

    def load_checkpoint_from_dict(self, checkpoint_dict: dict, device: Optional[str | torch.device] = None) -> None:
        self.norm.load_state_dict(checkpoint_dict["norm_state_dict"])
        self.seg.load_state_dict(checkpoint_dict["seg_state_dict"])

    @property
    def norm(self):
        return self._norm

    @property
    def seg(self):
        return self._seg

    def eval_mode(
        self,
    ) -> None:
        """
        Sets the model to evaluation mode.
        """
        self.eval()
        self._seg.eval()
        self._norm.eval()

    def train_mode(
        self,
    ) -> None:
        """
        Sets the model to training mode.
        """
        self.train()
        self._seg.train()
        self._norm.train()

    @property
    def trainable_modules(self) -> dict[str, torch.nn.Module]:
        """
        Returns the trainable parameters of the model.
        """
        return {
            "_norm": self._norm,
            "_seg": self._seg
        }

    def has_normalizer_module(self) -> bool:
        return True

    def get_normalizer_module(self) -> torch.nn.Module:
        return self._norm
    
    def get_all_modules_except_normalizer(self) -> dict[str, torch.nn.Module]:
        return {'_seg': self._seg}
    
    def get_normalizer_state_dict(self) -> dict[str, Any]:
        return {'_norm': self._norm.state_dict()}