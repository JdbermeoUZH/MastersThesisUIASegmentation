from typing import Any, Optional

import torch
import torch.nn as nn

from .DinoV2FeatureExtractor import DinoV2FeatureExtractor


class DinoSeg(nn.Module):
    def __init__(
            self,
            decoder: nn.Module,
            dino_fe: Optional[DinoV2FeatureExtractor] = None,
            features_are_precalculated: bool = False,
            ):
        super().__init__()

        self._decoder = decoder
        self._dino_fe = dino_fe
        
        self._features_are_precalculated = features_are_precalculated

        # TODO: Assertions to verify that decoder matches the dino features

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None, pre: bool = True, hierarchy: list[int] = [0]) -> torch.Tensor:
        if not self._features_are_precalculated:
            dino_out = self._dino_fe(image, mask, pre, hierarchy)
            image = ... # Get the dino features

        # Decode features
        self