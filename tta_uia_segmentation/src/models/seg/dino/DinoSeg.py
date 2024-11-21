from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DinoV2FeatureExtractor import DinoV2FeatureExtractor
from .BaseDecoder import BaseDecoder
from ..BaseSeg import BaseSeg
from tta_uia_segmentation.src.utils.io import save_checkpoint


class DinoSeg(BaseSeg):
    def __init__(
        self,
        decoder: BaseDecoder,
        dino_fe: Optional[DinoV2FeatureExtractor] = None,
        dino_model_name: Optional[str] = None,
        precalculated_fts: bool = False,
        hierarchy_level: int = 0,
    ):
        super().__init__()

        self._decoder = decoder
        self._dino_fe = dino_fe
        self._hierarchy_level_dino_fe = hierarchy_level

        if self._dino_fe is not None:
            assert (
                dino_model_name is None
            ), "Dino model name is not required when a Dino feature extractor is provided"
            self._dino_model_name = self._dino_fe.model_name
        else:
            assert (
                dino_model_name is not None
            ), "Dino model name is required when a Dino feature extractor is not provided"
            self._dino_model_name = dino_model_name

        self._precalculated_fts = precalculated_fts

    def forward(
        self, x: torch.Tensor | List[torch.Tensor], output_size: Optional[tuple[int, ...]] = None, **preprocess_kwargs
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
    
        if not self.precalculated_fts:
            assert isinstance(x, torch.Tensor), "If calculating features, x must be a tensor of image(s)"
            # Make output size of the decoder match the input's size if necessary
            output_size = tuple(x.shape[-2:])
        else:       
            assert output_size is not None, (
                "Output size is required when features are precalculated"
            )

        if self._decoder.output_size != output_size:
                self._decoder.output_size = output_size

        return super().forward(x, **preprocess_kwargs)

    def select_necessary_extra_inputs(self, extra_input_dict):
        assert 'output_size' in extra_input_dict, "Output size is required"
        assert isinstance(extra_input_dict['output_size'], tuple | list), "Output size must be a tuple"
        
        # Check all elements the batch have the same output size
        h = extra_input_dict['output_size'][0].unique()
        w = extra_input_dict['output_size'][1].unique()

        assert len(h) == 1 and len(w) == 1, "All images in the batch must have the same output size"

        output_size = (h[0].item(), w[0].item())

        return {'output_size': output_size}
    
    def _forward( # type: ignore
        self,
        x_preproc: torch.Tensor,
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

        return self._decoder(x_preproc)

    @torch.inference_mode()
    def _preprocess_x(
        self,
        x: torch.Tensor | List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        pre: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        # Calculate dino features if necessary
        if not self.precalculated_fts:
            # Convert grayscale to RGB, required by DINO
            assert isinstance(x, torch.Tensor), "If calculating features, x must be a tensor of image(s)"

            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

            assert (
                self._dino_fe is not None
            ), "Dino feature extractor is required when features are not precalculated"
            dino_out = self._dino_fe(x, mask, pre, self._hierarchy_level_dino_fe)
            x = dino_out["patch"].permute(
                0, 3, 1, 2
            )  # N x np x np x df -> N x df x np x np
        else:
            assert isinstance(x, list), "If features are precalculated, x must be a list of tensors"
            x = x[self._hierarchy_level_dino_fe]

        assert isinstance(x, torch.Tensor), "x must be a tensor"

        return x, {"Dino Features": x}

    def save_checkpoint(self, path: str, **kwargs) -> None:
        save_checkpoint(
            path=path,
            decoder_state_dict=self.decoder.state_dict(),
            dino_model_name=self._dino_model_name,
            **kwargs,
        )

    def load_checkpoint(
        self,
        path: str,
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
    def trainable_params(self) -> List[nn.Parameter]:
        return list(self._decoder.parameters())

    @property
    def trainable_modules(self) -> List[torch.nn.Module]:
        """
        Returns the trainable parameters of the model.
        """
        return [self._decoder]
