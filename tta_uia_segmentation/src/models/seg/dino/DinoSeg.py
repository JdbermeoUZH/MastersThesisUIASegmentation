from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DinoV2FeatureExtractor import DinoV2FeatureExtractor
from .ResNetDecoder import ResNetDecoder
from ..BaseSeg import BaseSeg
from tta_uia_segmentation.src.utils.io import save_checkpoint


class DinoSeg(BaseSeg):
    def __init__(
        self,
        decoder: ResNetDecoder,
        dino_fe: Optional[DinoV2FeatureExtractor] = None,
        precalculated_fts: bool = False,
    ):
        super().__init__()

        self._decoder = decoder
        self._dino_fe = dino_fe

        self._precalculated_fts = precalculated_fts

    def forward(
        self, x: torch.Tensor, **preprocess_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass of the model

        Returns:
        --------
        y_mask : torch.Tensor
            Predicted segmentation mask.

        y_logits : torch.Tensor
            Predicted logits.

        intermediate_outputs : dict[str, torch.Tensor]
            Dictionary containing intermediate outputs of preprocessing.
        """
        # Make output size of the decoder match the input's size if necessary
        if self._decoder.output_size != x.shape[-2:]:
            self._decoder.output_size = tuple(x.shape[-2:])

        return super().forward(x, **preprocess_kwargs)
        
    def _forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the Dino features to a segmentation mask

        Returns:
        --------
        y_mask : torch.Tensor
            Predicted segmentation mask.

        y_logits : torch.Tensor
            Predicted logits.
        """

        return self.decoder(x)

    @torch.inference_mode()
    def _preprocess_x(
        self,
        x,
        mask: Optional[torch.Tensor] = None,
        pre: bool = False,
        hierarchy: int = 0,
    ):

        # Calculate dino features if necessary
        if not self.precalculated_fts:
            # Convert grayscale to RGB, required by DINO
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

            assert (
                self._dino_fe is not None
            ), "Dino feature extractor is required when features are not precalculated"
            dino_out = self._dino_fe(x, mask, pre, hierarchy)
            x = dino_out["patch"].permute(
                0, 3, 1, 2
            )  # N x np x np x df -> N x df x np x np

        return x, {"Dino Features": x}

    def save_checkpoint(self, path: str, **kwargs) -> None:
        save_checkpoint(
            path=path,
            decoder_state_dict=self.decoder.state_dict(),
            dino_model_name=self.dino_fe.model_name,
            **kwargs,
        )

    def load_checkpoint(
        self,
        path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:

        checkpoint = torch.load(path, map_location=device)

        # Load Model weights
        if "decoder_state_dict" in checkpoint:
            self._decoder.load_state_dict(checkpoint["decoder_state_dict"])
        elif "seg_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["seg_state_dict"])
        else:
            raise ValueError("No decoder state dict found in the checkpoint")

        # Load Dino feature extractor if specified in the checkpoint
        if "dino_model_name" in checkpoint:
            dino_model_name = checkpoint["dino_model_name"]
            self._dino_fe = DinoV2FeatureExtractor(model=dino_model_name)

    @property
    def decoder(self):
        return self._decoder

    @property
    def dino_fe(self):
        return self._dino_fe

    @property
    def precalculated_fts(self):
        return self._precalculated_fts

    @precalculated_fts.setter
    def precalculated_fts(self, value: bool):
        self._precalculated_fts = value

    def eval_mode(
        self,
    ) -> None:
        """
        Sets the model to evaluation mode.
        """
        self.eval()
        self._decoder.eval_mode()

    def train_mode(self):
        """
        Sets the model to training mode.

        We keep the Dino encoder always frozen
        """
        self._decoder.train_mode()

    @property
    def trainable_params(self) -> list[nn.Parameter]:
        return list(self._decoder.parameters())
    
    @property
    def trainable_modules(self) -> list[torch.nn.Module]:
        """
        Returns the trainable parameters of the model.
        """
        return (self._decoder,)