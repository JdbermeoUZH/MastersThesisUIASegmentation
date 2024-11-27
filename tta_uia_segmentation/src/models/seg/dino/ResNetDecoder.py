import logging
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseDecoder import BaseDecoder
from ..norm_seg.UNet import get_conv, DoubleConv
from tta_uia_segmentation.src.utils.utils import assert_in, default

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class UpResNetDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        convs_per_block: int = 2,
        scale_factor: Optional[float | tuple[float, ...]] = None,
        size: Optional[tuple[int, ...]] = None,
        n_dimensions=2,
    ):
        super().__init__()

        assert_in(n_dimensions, "n_dimensions", [1, 2, 3])

        # Upsample module
        # :====================================================================:
        if n_dimensions == 1:
            interpolation_mode = "linear"
        elif n_dimensions == 2:
            interpolation_mode = "bilinear"
        else:
            interpolation_mode = "trilinear"

        self._up = nn.Upsample(
            scale_factor=scale_factor,
            size=size,
            mode=interpolation_mode,
            align_corners=True,
        )

        # Channel reduction module
        # :====================================================================:
        if in_channels != out_channels:
            self._channel_reduction = get_conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                n_dimensions=n_dimensions,
            )
        else:
            self._channel_reduction = None

        # Convolutional blocks
        # :====================================================================:

        self._conv_blocks = nn.ModuleList(
            [
                DoubleConv(out_channels, out_channels, n_dimensions=n_dimensions)
                for _ in range(convs_per_block)
            ]
        )

    def forward(self, x, scale_factor=None, size=None):
        if size is not None:
            self._up.size = size

        if scale_factor is not None:
            if isinstance(scale_factor, torch.Tensor):
                scale_factor = scale_factor.cpu().numpy().tolist()

            self._up.scale_factor = scale_factor

        x = self._up(x)

        if self._channel_reduction is not None:
            x = self._channel_reduction(x)

        for conv_block in self._conv_blocks:
            x = x + conv_block(x)

        return x


class ResNetDecoder(BaseDecoder):
    def __init__(
        self,
        embedding_dim: int,
        n_classes: int,
        num_channels: tuple[int, ...],
        output_size: tuple[int, ...],
        n_dimensions: int = 2,
        convs_per_block: int = 2,
    ):

        super(ResNetDecoder, self).__init__(output_size=output_size)

        self._embedding_dim = embedding_dim
        self._output_size = output_size
        self._n_classes = n_classes
        self._num_channels = num_channels
        self._n_dimensions = n_dimensions
        self._in_train_mode = True

        # If number of channels per level is provided, then check it matches the number of levels
        # :====================================================================:
        num_ch = [embedding_dim] + list(num_channels)

        block_list = []
        out_ch = None
        for level_i, (in_ch, out_ch) in enumerate(zip(num_ch[:-1], num_ch[1:])):
            scale_factor = 2 if level_i < len(num_ch) - 2 else None
            block_list.append(
                UpResNetDecoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    scale_factor=scale_factor,
                    n_dimensions=n_dimensions,
                    convs_per_block=convs_per_block,
                )
            )

        self.blocks = nn.ModuleList(block_list)

        self.output_conv = get_conv(
            in_channels=out_ch,
            out_channels=n_classes,
            kernel_size=1,
            n_dimensions=n_dimensions,
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        # Pass trough decoder blocks
        for block_i, block in enumerate(self.blocks):
            if block_i < len(self.blocks) - 1:
                x = block(x)
            else:
                # Upsample to original size
                scale_factor = torch.tensor(self._output_size) / torch.tensor(
                    x.shape[-self._n_dimensions :]
                )

                if (scale_factor > 2.0).any():
                    msg = (
                        f"Upsampling is too large: {scale_factor},"
                        + " Consider adding another block to the decoder instead"
                    )
                    if self._in_train_mode:
                        raise ValueError(msg)
                    else:
                        logger.warning(msg)
                        
                x = block(x, size=self._output_size)

        # Output to n_classes
        x = self.output_conv(x)

        return self.softmax(x), x

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
