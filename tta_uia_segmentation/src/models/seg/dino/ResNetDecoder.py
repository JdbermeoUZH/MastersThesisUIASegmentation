import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..norm_seg.UNet import get_conv, DoubleConv
from tta_uia_segmentation.src.utils.utils import assert_in, default

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ResNetDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor: Optional[float] = 2.0,
        n_dimensions=2,
    ):
        super().__init__()

        assert_in(n_dimensions, "n_dimensions", [1, 2, 3])

        if n_dimensions == 1:
            self.interpolation_mode = "linear"
        elif n_dimensions == 2:
            self.interpolation_mode = "bilinear"
        else:
            self.interpolation_mode = "trilinear"

        self.scale_factor = scale_factor

        self.double_conv = DoubleConv(
            in_channels, out_channels, n_dimensions=n_dimensions
        )

        self.residual_conv = get_conv(
            in_channels, out_channels, 1, n_dimensions=n_dimensions
        )

    def forward(self, x, scale_factor=None, size=None):
        size = default(size, None)
        scale_factor = default(scale_factor, self.scale_factor)

        if isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.cpu().numpy().tolist()

        x = F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=self.interpolation_mode,
            align_corners=True,
        )

        x = self.double_conv(x) + self.residual_conv(x)

        return x


class ResNetDecoder(nn.Module):
    def __init__(
        self,
        n_classes: int,
        output_size: tuple[int, int],
        channels=[128, 64, 32, 16],
        n_dimensions=2,
    ):

        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResNetDecoderBlock(
                    channels[i], channels[i + 1], 2, n_dimensions=n_dimensions
                )
                for i in range(len(channels) - 2)
            ]
        )

        self.last_block = ResNetDecoderBlock(
            channels[-2], channels[-1], None, n_dimensions=n_dimensions
        )

        self.output_conv = get_conv(
            channels[-1], n_classes, 1, n_dimensions=n_dimensions
        )
        self.softmax = nn.Softmax(dim=1)

        self._output_size = output_size
        self._in_train_mode = True

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        # Pass trough decoder blocks
        for block in self.blocks:
            x = block(x)

        # Upsample to original size
        scale_factor = torch.tensor(self._output_size) / torch.tensor(x.shape[-2:])
        
        if (scale_factor > 2.0).any():
            msg = (
                f"Upsampling is too large: {scale_factor},"
                + " Consider adding another block to the decoder instead"
            )
            if self._in_train_mode:
                raise ValueError(msg)
            else:
                logger.warning(msg)

        x = self.last_block(x, size=self._output_size)

        # Output to n_classes
        x = self.output_conv(x)

        return self.softmax(x), x

    @property
    def output_size(self) -> tuple[int, int]:
        return self._output_size

    @output_size.setter
    def output_size(self, output_size: tuple[int, int]) -> None:
        self._output_size = output_size

    @property
    def in_train_mode(self) -> bool:
        return self._in_train_mode

    @in_train_mode.setter
    def in_train_mode(self, mode: bool) -> None:
        self._in_train_mode = mode

    def eval_mode(self) -> None:
        """
        Sets the model to evaluation mode.
        """
        self.eval()
        self._in_train_mode = False
    
    def train_mode(self) -> None:
        """
        Sets the model to training mode.
        """
        self.train()
        self._in_train_mode = True
