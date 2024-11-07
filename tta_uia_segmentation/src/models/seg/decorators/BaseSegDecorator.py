from typing import Optional, Any

import torch

from ..BaseSeg import BaseSeg


class BaseSegDecorator(BaseSeg):
    def __init__(self, base_seg: BaseSeg):
        super(BaseSegDecorator, self).__init__()
        self._base_seg = base_seg

    def _preprocess_x(
        self,
        x,
        **preprocess_kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._base_seg._preprocess_x(x, **preprocess_kwargs)

    def _forward(self, x_preproc) -> tuple[torch.Tensor, torch.Tensor]:
        return self._base_seg._forward(x_preproc)

    def forward(
        self, x: torch.Tensor, **preprocess_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return self._base_seg.forward(x, **preprocess_kwargs)

    @torch.inference_mode()
    def predict_mask(self, x: torch.Tensor, **preprocess_kwargs) -> torch.Tensor:
        return self._base_seg.predict_mask(x, **preprocess_kwargs)

    def eval_mode(
        self,
    ) -> None:
        self._base_seg.eval_mode()

    def train_mode(
        self,
    ) -> None:
        self._base_seg.train_mode()
