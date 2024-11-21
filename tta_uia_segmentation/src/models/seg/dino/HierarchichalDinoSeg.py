from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DinoV2FeatureExtractor import DinoV2FeatureExtractor
from .DinoSeg import DinoSeg
from .BaseDecoder import BaseDecoder
from tta_uia_segmentation.src.utils.io import save_checkpoint


class HierarchichalDinoSeg(DinoSeg):
    def __init__(
        self,
        decoder: BaseDecoder,
        hierarchy_levels: int,
        dino_fe: Optional[DinoV2FeatureExtractor] = None,
        precalculated_fts: bool = False,
    ):
        self._hierarchy_levels = hierarchy_levels

        super(HierarchichalDinoSeg, self).__init__(
            decoder=decoder,
            dino_fe=dino_fe,
            precalculated_fts=precalculated_fts,
            hierarchy_level=0
        )
        
    @torch.inference_mode()
    def _preprocess_x( # type: ignore
        self,
        x: torch.Tensor | List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        pre: bool = False,
    ) -> tuple[List[torch.Tensor], dict[str, torch.Tensor]]:

        # Calculate dino features if necessary
        if not self.precalculated_fts:
            assert isinstance(x, torch.Tensor), "If calculating features, x must be a tensor of image(s)"
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
            assert isinstance(x, list), "If features are precalculated, x must be a list of tensors"
            x_preproc_list = x
        
        intermediate_outputs = {f"Dino Features (hier {i})": dino_fe for i, dino_fe in enumerate(x_preproc_list)}

        return x_preproc_list, intermediate_outputs
