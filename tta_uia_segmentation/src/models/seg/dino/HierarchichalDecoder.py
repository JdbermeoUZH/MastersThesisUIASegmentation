import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..norm_seg.UNet import get_conv, DoubleConv
from .BaseDecoder import BaseDecoder
from tta_uia_segmentation.src.utils.utils import assert_in, default

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class DecoderBlock(nn.Module):
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

    def forward(self, x, scale_factor=None, size=None):
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

        return self.double_conv(x)


    

class HierarchichalDecoder(BaseDecoder):
    def __init__(
        self,
        embedding_dim: int,
        n_classes: int,
        num_channels_last_upsample: int,
        output_size: tuple[int, ...],
        hierarchy_level: int,
        num_channels_per_hier: Optional[list[int]] = None,
        n_dimensions: int = 2
    ):

        super(HierarchichalDecoder, self).__init__(output_size=output_size)
        self._embedding_dim = embedding_dim
        self._hierarchy_level = hierarchy_level
        self._n_classes = n_classes
        self._num_channels_last_upsample = num_channels_last_upsample
        self._in_train_mode = True

        # If number of channels per level is provided, then check it matches the number of levels
        # :====================================================================:
        if num_channels_per_hier is not None:
            assert len(num_channels_per_hier) == hierarchy_level + 1, (
                "Number of channels per level should match the number hierarchies",
            )
        else:
            num_channels_per_hier = [
                embedding_dim / (2**hier_i) for hier_i in range(1, hierarchy_level + 1)
            ]
            breakpoint()
            print("Check the number of channels is defined correctly")
        self._num_channels_per_hier = num_channels_per_hier

        # Define modules for each hierarchy level
        # :====================================================================:
        hier_blocks = list()
        num_channels = [self._embedding_dim] + num_channels_per_hier 
        
        for hier_i, (in_channels, out_channels) in enumerate(zip(num_channels[:-1], num_channels[1:])):
        
            if hier_i > 0:
                in_channels += self._embedding_dim
            hier_blocks.append(
                DecoderBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                scale_factor=2,
                n_dimensions=n_dimensions,
            )
            )

        self.hier_blocks = nn.ModuleList(hier_blocks)

        # Define modules for last upsampling block and classification head
        # :====================================================================:
        self.last_block = DecoderBlock(
            in_channels=num_channels_per_hier[-1],
            out_channels=num_channels_last_upsample,
            scale_factor=None, # Determined dinamically based on the output size
            n_dimensions=n_dimensions,
        )

        self.output_conv = get_conv(
            in_channels=num_channels_last_upsample,
            out_channels=n_classes,
            kernel_size=1,
            n_dimensions=n_dimensions
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Pass trough decoder blocks for each hierarchy level
        # :====================================================================:
        out_hier_i = None
        for hier_i, hier_block in enumerate(self.hier_blocks):
            if hier_i > 0:
                # Pad spatial dimensions of the output of the previous hierarchy level
                # to match the spatial dimensions of the current hierarchy level
                x[hier_i] = F.pad(
                    x[hier_i],
                    (0, out_hier_i.shape[-1]- x[hier_i].shape[-1], 0, out_hier_i.shape[-2] - x[hier_i].shape[-2]),
                    mode='constant',
                    value=0
                )
                x_in = torch.cat([out_hier_i, x[hier_i]], dim=1)
            else:
                x_in = x[hier_i]   

            out_hier_i = hier_block(x_in)

        # Last upsampling block to original size
        # :====================================================================:
        out = self.last_block(out_hier_i, size=self._output_size)

        # Output to n_classes
        out = self.output_conv(out)

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
