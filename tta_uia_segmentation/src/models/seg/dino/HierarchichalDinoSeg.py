from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DinoV2FeatureExtractor import DinoV2FeatureExtractor
from .DinoSeg import DinoSeg
from .BaseDecoder import BaseDecoder
from tta_uia_segmentation.src.models.pca.BasePCA import BasePCA
from tta_uia_segmentation.src.utils.io import save_checkpoint


class HierarchichalDinoSeg(DinoSeg):
    def __init__(
        self,
        decoder: BaseDecoder,
        hierarchy_levels: int,
        dino_fe: Optional[DinoV2FeatureExtractor] = None,
        pca: Optional[BasePCA] = None,
        precalculated_fts: bool = False,
        **kwargs,
    ):
        self._hierarchy_levels = hierarchy_levels

        super(HierarchichalDinoSeg, self).__init__(
            decoder=decoder,
            dino_fe=dino_fe,
            pca=pca,
            precalculated_fts=precalculated_fts,
            hierarchy_level=0,
            **kwargs,
        )

    @torch.inference_mode()
    def _preprocess_x(  # type: ignore
        self,
        x: torch.Tensor | List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        pre: bool = False,
    ) -> tuple[List[torch.Tensor], dict[str, torch.Tensor]]:

        # Get Dino Features
        # :=========================================================================:
        x_preproc_list = []
        for hier_i in range(self._hierarchy_levels + 1):
            x_preproc_list.append(
                self._extract_dino_features(x, mask, pre, hierarchy=hier_i)
            )

        # Apply PCA if necessary
        # :=========================================================================:
        if self._pca is not None:
            x_preproc_list = [self._get_pc_dino_features(x) for x in x_preproc_list]

        intermediate_outputs = {
            f"Dino Features (hier {i})": dino_fe
            for i, dino_fe in enumerate(x_preproc_list)
        }

        return x_preproc_list, intermediate_outputs
