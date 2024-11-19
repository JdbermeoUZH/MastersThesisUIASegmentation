from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DinoV2FeatureExtractor import DinoV2FeatureExtractor
from .HierarchichalResNetDecoder import HierarchichalResNetDecoder
from .DinoSeg import DinoSeg
from tta_uia_segmentation.src.utils.io import save_checkpoint


class HierarchichalDinoSeg(DinoSeg):
    def __init__(
        self,
        decoder: HierarchichalResNetDecoder,
        hierarchy_levels: int,
        dino_fe: Optional[DinoV2FeatureExtractor] = None,
        precalculated_fts: bool = False,
    ):
        self._hierarchy_levels = hierarchy_levels

        super(HierarchichalDinoSeg, self).__init__(
            decoder=decoder,
            dino_fe=dino_fe,
            precalculated_fts=precalculated_fts,
        )
        
    @torch.inference_mode()
    def _preprocess_x(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pre: bool = False,
        hierarchy: int = 0,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:

        # Calculate dino features if necessary
        if not self.precalculated_fts:
            # Convert grayscale to RGB, required by DINO
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

            assert (
                self._dino_fe is not None
            ), "Dino feature extractor is required when features are not precalculated"
            
            
            x_preproc_list = []
            for hier_i in range(self._hierarchy_levels):
                x_preproc = self._dino_fe(x, mask, pre, hierarchy=hier_i)
                x_preproc_list.append(x_preproc["patch"].permute(0, 3, 1, 2))

        else:
            x_preproc_list = x
        
        intermediate_outputs = {f"Dino Features (hier {i})": x_preproc_list[i] for i in range(self._hierarchy_levels)}

        return x, intermediate_outputs
