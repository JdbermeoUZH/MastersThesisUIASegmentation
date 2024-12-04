import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..norm_seg.UNet import get_conv, DoubleConv
from .BaseDecoder import BaseDecoder
from .ResNetDecoder import UpResNetDecoderBlock
from tta_uia_segmentation.src.utils.utils import assert_in, default

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class UpDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        convs_per_block: int = 2,
        scale_factor: Optional[float] = 2.0,
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

        # Convolutional blocks
        # :====================================================================:
        conv_block_list = list()
        for i in range(convs_per_block):
            if i == 0:
                in_channels = in_channels
            else:
                in_channels = out_channels

            conv_block_list.append(
                DoubleConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_dimensions=n_dimensions,
                )
            )

        self._conv_blocks = nn.ModuleList(conv_block_list)

    def forward(self, x, scale_factor=None, size=None):
        if size is not None:
            self._up.size = size

        if scale_factor is not None:
            if isinstance(scale_factor, torch.Tensor):
                scale_factor = scale_factor.cpu().numpy().tolist()

            self._up.scale_factor = scale_factor

        x = self._up(x)
        for conv_block in self._conv_blocks:
            x = conv_block(x)

        return x


class HierarchichalDecoder(BaseDecoder):
    def __init__(
        self,
        embedding_dim: int,
        n_classes: int,
        output_size: tuple[int, ...],
        num_channels: tuple[int, ...],
        hierarchy_level: int,
        n_dimensions: int = 2,
        convs_per_block: int = 2,
    ):

        super(HierarchichalDecoder, self).__init__(output_size=output_size)
        self._embedding_dim = embedding_dim
        self._hierarchy_level = hierarchy_level
        self._n_classes = n_classes
        self._num_channels = num_channels
        self._in_train_mode = True

        # Check we have at least as many channels as hierarchy levels + 1
        # :====================================================================:
        assert len(num_channels) >= hierarchy_level + 1, (
            f"Number of channels per hierarchy level must be at least"
            + f" hierarchy_level + 1: {hierarchy_level + 1}"
        )

        # Define modules for each hierarchy level
        # :====================================================================:
        conv_blocks = list()
        out_ch = None
        num_ch = [self._embedding_dim] + list(num_channels)
        for level_i, (in_ch, out_ch) in enumerate(zip(num_ch[:-1], num_ch[1:])):
            # Add decoder blocks for each hierarchy level
            if level_i <= self._hierarchy_level:
                if level_i > 0:
                    in_ch += self._embedding_dim

                conv_blocks.append(
                    UpDecoderBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        scale_factor=2,
                        n_dimensions=n_dimensions,
                        convs_per_block=convs_per_block,
                    )
                )
            # Add Resnet blocks to continue upsampling
            else:
                scale_factor = 2 if level_i < len(num_ch) - 2 else None
                conv_blocks.append(
                    UpResNetDecoderBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        scale_factor=scale_factor,
                        n_dimensions=n_dimensions,
                        convs_per_block=convs_per_block,
                    )
                )

        self.conv_blocks = nn.ModuleList(conv_blocks)

        # Define modules for classification head
        # :====================================================================:
        self.output_conv = get_conv(
            in_channels=out_ch,
            out_channels=n_classes,
            kernel_size=1,
            n_dimensions=n_dimensions,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Pass trough decoder blocks for each hierarchy level
        # :====================================================================:
        out_hier_i: Optional[torch.Tensor] = None
        for level_i, conv_block in enumerate(self.conv_blocks):
            # Use Decoder blocks that take as input dino features from upsampled images
            if level_i <= self._hierarchy_level:
                if level_i > 0:
                    # Pad spatial dimensions of the output of the previous hierarchy level
                    # to match the spatial dimensions of the current hierarchy level
                    assert isinstance(out_hier_i, torch.Tensor)

                    pad_x = out_hier_i.shape[-1] - x[level_i].shape[-1]
                    pad_y = out_hier_i.shape[-2] - x[level_i].shape[-2]

                    x[level_i] = F.pad(
                        x[level_i], (0, pad_x, 0, pad_y), mode="constant", value=0
                    )
                    x_in = torch.cat([out_hier_i, x[level_i]], dim=1)
                else:
                    x_in = x[level_i]

            # Use ResNet blocks to continue upsampling
            else:
                x_in = out_hier_i

            if level_i < len(self.conv_blocks) - 1:
                out_hier_i = conv_block(x_in)
            else:
                out_hier_i = conv_block(x_in, size=self._output_size)

        # Classify
        # :====================================================================:
        out = self.output_conv(out_hier_i)

        return self.softmax(out), out

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
