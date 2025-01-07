from typing import Any, Optional, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DinoV2FeatureExtractor import DinoV2FeatureExtractor
from .BaseDecoder import BaseDecoder
from .DinoSeg import DinoSeg
from tta_uia_segmentation.src.models.pca.BasePCA import BasePCA
from tta_uia_segmentation.src.utils.utils import min_max_normalize_channelwise, default


class NormDinoSeg(DinoSeg):
    def __init__(
        self,
        norm: nn.Module,
        decoder: BaseDecoder,
        dino_fe: Optional[DinoV2FeatureExtractor] = None,
        pca: Optional[BasePCA] = None,
        pc_norm_type: Literal["bn_layer", "per_img", None] = "per_img",
        dino_model_name: Optional[str] = None,
        dino_emb_dim: Optional[int] = None,
        hierarchy_level: int = 0,
    ):

        super(NormDinoSeg, self).__init__(
            decoder=decoder,
            dino_fe=dino_fe,
            pca=pca,
            pc_norm_type=pc_norm_type,
            dino_model_name=dino_model_name,
            dino_emb_dim=dino_emb_dim,
            precalculated_fts=False,
            hierarchy_level=hierarchy_level,
            encoder_inference_mode=False
        )

        self._norm = norm
        
        assert self._dino_fe is not None, (
            "Dino Feature Extractor must be provided to calculate the Jacobian"
        )
        self._dino_fe.inference_mode = False
        self._dino_fe.freeze()

    def _preprocess_x(
        self,
        x: torch.Tensor | List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        pre: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        
        # Normalize input
        # :=========================================================================:
        if isinstance(x, torch.Tensor):
            x_norm = self._norm(x)

        elif isinstance(x, list):
            x_norm = self._norm(x[self._hierarchy_level_dino_fe])

        x_dino_features, *_ = super()._preprocess_x(x_norm, mask, pre)

        return x_dino_features, {"Dino Features": x_dino_features, "Normalized Image": x_norm}

    def checkpoint_as_dict(self, **kwargs) -> dict:
        """
        Returns the model checkpoint as a dictionary.
        """
        
        return dict(
            norm_state_dict=self._norm.state_dict(),
            decoder_state_dict=self.decoder.state_dict(),
            dino_model_name=self._dino_model_name,
            pca=self._pca.serialize_to_dict() if self._pca is not None else None,
            **kwargs,
        )

    def load_checkpoint_from_dict(
        self,
        checkpoint_dict: dict,
        device: Optional[str | torch.device] = None,
    ) -> None:
        # Load Normalizer weights
        self._norm.load_state_dict(checkpoint_dict["norm_state_dict"])
        
        # Load Decoder weights
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

        # Load PCA model if specified in the checkpoint
        if "pca" in checkpoint_dict and checkpoint_dict["pca"] is not None:
            self._pca = BasePCA.load_pipeline_from_dict(checkpoint_dict["pca"])
        else:
            self._pca = None

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
        raise NotImplementedError(
            "Pre-calculated features are not supported as we fit a normalization module "
            "that modifies the image before the getting Dino Features"
            )
    
    @precalculated_fts.setter
    def precalculated_fts(self, value: bool):
        raise NotImplementedError(
            "Pre-calculated features are not supported as we fit a normalization module "
            "that modifies the image before the getting Dino Features"
            )
    
    def eval_mode(
        self,
    ) -> None:
        """
        Sets the model to evaluation mode.
        """
        self.eval()
        self._decoder.eval_mode()
        self._norm.eval()

        if self._dino_fe is not None:
            self._dino_fe.eval()

    def train_mode(self):
        """
        Sets the model to training mode.

        We keep the Dino encoder always frozen
        """
        self._decoder.train_mode()
        self._norm.train()
        if self._dino_fe is not None:
            self._dino_fe.eval()

    @property
    def trainable_modules(self) -> dict[str, torch.nn.Module]:
        """
        Returns the trainable parameters of the model.
        """
        return {"_decoder": self._decoder, "_norm": self._norm}

    def has_normalizer_module(self) -> bool:
        return True

    def get_normalizer_module(self) -> nn.Module:
        return self._norm

    def get_normalizer_state_dict(self) -> dict[str, Any]:
        return {'_norm': self._norm.state_dict()}
