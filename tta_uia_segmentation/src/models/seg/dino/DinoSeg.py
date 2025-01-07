from typing import Optional, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DinoV2FeatureExtractor import DinoV2FeatureExtractor
from .BaseDecoder import BaseDecoder
from ..BaseSeg import BaseSeg
from tta_uia_segmentation.src.models.pca.BasePCA import BasePCA
from tta_uia_segmentation.src.utils.utils import (
    min_max_normalize_channelwise,
    inference_mode_if_enabled,
    default
)

class DinoSeg(BaseSeg):
    def __init__(
        self,
        decoder: BaseDecoder,
        dino_fe: Optional[DinoV2FeatureExtractor] = None,
        pca: Optional[BasePCA] = None,
        pc_norm_type: Literal["bn_layer", "per_img", None] = "per_img",
        dino_model_name: Optional[str] = None,
        dino_emb_dim: Optional[int] = None,
        precalculated_fts: bool = False,
        hierarchy_level: int = 0,
        encoder_inference_mode: bool = True,
        return_intermediate_outputs: bool = False,
    ):
        super().__init__()

        self._decoder = decoder
        self._dino_fe = dino_fe
        self._pca = pca
        self._hierarchy_level_dino_fe = hierarchy_level

        if self._dino_fe is not None:
            assert dino_model_name is None, (
                "Dino model name is not required when a "
                "Dino feature extractor is provided"
            )
            assert dino_emb_dim is None, (
                "Dino embedding dimension is not required when a "
                "Dino feature extractor is provided"
            )
            self._dino_model_name = self._dino_fe.model_name
            self._dino_emb_dim = self._dino_fe.emb_dim

        else:
            assert dino_model_name is not None, (
                "Dino model name is required when a "
                "Dino feature extractor is not provided"
            )
            assert dino_emb_dim is not None, (
                "Dino embedding dimension is required when a "
                "Dino feature extractor is not provided"
            )

            self._dino_model_name = dino_model_name
            self._dino_emb_dim = dino_emb_dim

        self._precalculated_fts = precalculated_fts
        
        self._encoder_inference_mode = encoder_inference_mode

        self._pc_norm_type = pc_norm_type

        self._bn_dino_features = None

        self._return_intermediate_outputs = return_intermediate_outputs

        if self._pc_norm_type is not None:
            assert self._pca is not None, "PCA model is required if normalization is expected"

            if self._pc_norm_type == "bn_layer":
                n_components = default(self._pca.n_components, self._dino_emb_dim)
                self._bn_dino_features = nn.BatchNorm2d(n_components)

    def forward(
        self,
        x: torch.Tensor | List[torch.Tensor],
        output_size: Optional[tuple[int, ...]] = None,
        **preprocess_kwargs,
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

        if not self._precalculated_fts:
            assert isinstance(
                x, torch.Tensor
            ), "If calculating features, x must be a tensor of image(s)"
            # Make output size of the decoder match the input's size if necessary
            output_size = tuple(x.shape[-2:])
        else:
            assert (
                output_size is not None
            ), "Output size is required when features are precalculated"

        if self._decoder.output_size != output_size:
            self._decoder.output_size = output_size

        return super().forward(x, **preprocess_kwargs)

    def select_necessary_extra_inputs(self, extra_input_dict):
        if not self._precalculated_fts:
            return {}

        assert "output_size" in extra_input_dict, "Output size is required"
        assert isinstance(
            extra_input_dict["output_size"], tuple | list
        ), "Output size must be a tuple"

        # Check all elements the batch have the same output size
        h = extra_input_dict["output_size"][0].unique()
        w = extra_input_dict["output_size"][1].unique()

        assert (
            len(h) == 1 and len(w) == 1
        ), "All images in the batch must have the same output size"

        output_size = (h[0].item(), w[0].item())

        return {"output_size": output_size}

    def _forward(  # type: ignore
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
        
        if x_preproc.is_inference():
            x_preproc = x_preproc.clone()
                
        return self._decoder(x_preproc)

    def _preprocess_x(
        self,
        x: torch.Tensor | List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        pre: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with inference_mode_if_enabled(self._encoder_inference_mode):
            # Calculate dino features if necessary
            # :=====================================================================:
            x = self._extract_dino_features(x, mask, pre)

            # Apply PCA if necessary
            # :=====================================================================:
            if self._pca is not None:
                x = self._get_pc_dino_features(x)

        if self._return_intermediate_outputs:
            interm_outs = {"Dino Features": x}
        else:
            interm_outs = {}

        return x, interm_outs

    def _extract_dino_features(
        self,
        x: torch.Tensor | List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        pre: bool = False,
        hierarchy: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract Dino features from a batch of images

        Returns:
        --------
        x : torch.Tensor
            Extracted Dino features.
        """

        hierarchy_: int = default(hierarchy, self._hierarchy_level_dino_fe)

        # Get Dino Features
        # :=========================================================================:
        if not self._precalculated_fts:
            # Convert grayscale to RGB, required by DINO
            assert isinstance(
                x, torch.Tensor
            ), "If calculating features, x must be a tensor of image(s)"

            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

            assert (
                self._dino_fe is not None
            ), "Dino feature extractor is required when features are not precalculated"

            dino_out = self._dino_fe(x, mask, pre, hierarchy_)
            x_out = dino_out["patch"].permute(
                0, 3, 1, 2
            )  # N, np, np, df -> N, df, np, np

        else:
            assert isinstance(
                x, list
            ), "If features are precalculated, x must be a list of tensors"
            x_out = x[hierarchy_]

        assert isinstance(x_out, torch.Tensor), "x must be a tensor"

        return x_out

    def _get_pc_dino_features(self, x: torch.Tensor) -> torch.Tensor:
        assert self._pca is not None, "PCA model is required"

        # Map to principal components
        x = self._pca.img_to_pcs(x)
        if self._pc_norm_type == "bn_layer":
            assert (
                self._bn_dino_features is not None
            ), "BatchNorm layer was not initialized"
            x = self._bn_dino_features(x)

        elif self._pc_norm_type == "per_img":
            x = min_max_normalize_channelwise(x, spatial_dims=(-2, -1))

        elif self._pc_norm_type is not None:
            pass
        else:
            raise ValueError(f"Invalid normalization type: {self._pc_norm_type}")

        return x

    def checkpoint_as_dict(self, **kwargs) -> dict:
        """
        Returns the model checkpoint as a dictionary.
        """
        return dict(
            decoder_state_dict=self.decoder.state_dict(),
            dino_model_name=self._dino_model_name,
            **kwargs,
        )

    def load_checkpoint_from_dict(
        self,
        checkpoint_dict: dict,
        device: Optional[str | torch.device] = None,
    ) -> None:
        # Load Model weights
        if "decoder_state_dict" in checkpoint_dict:
            self._decoder.load_state_dict(checkpoint_dict["decoder_state_dict"])
        elif "seg_state_dict" in checkpoint_dict:
            self.load_state_dict(checkpoint_dict["seg_state_dict"])
        else:
            raise ValueError("No decoder state dict found in the checkpoint")

        # Load Dino feature extractor if specified in the checkpoint
        if "dino_model_name" in checkpoint_dict:
            dino_model_name = checkpoint_dict["dino_model_name"]
            self._dino_fe = DinoV2FeatureExtractor(model=dino_model_name)

        if device is not None:
            self = self.to(device)        

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
    def trainable_modules(self) -> dict[str, torch.nn.Module]:
        """
        Returns the trainable parameters of the model.
        """
        return {"_decoder": self._decoder}

    @property
    def pca_n_components(self) -> Optional[int]:
        if self._pca is not None:
            return self._pca.n_components
        return None

    @pca_n_components.setter
    def pca_n_components(self, value: Optional[int]):
        if self._pca is not None:
            self._pca.n_components = value
        else:
            raise ValueError("No PCA model initialized")

    def has_normalizer_module(self) -> bool:
        return False

    def get_normalizer_module(self) -> nn.Module:
        raise ValueError("Model does not have a normalizer module")

    def get_normalizer_state_dict(self) -> dict[str, torch.Tensor]:
        raise ValueError("Model does not have a normalizer module")
