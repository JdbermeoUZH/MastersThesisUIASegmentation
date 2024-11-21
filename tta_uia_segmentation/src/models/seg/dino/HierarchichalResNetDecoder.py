import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..norm_seg.UNet import get_conv, DoubleConv
from .ResNetDecoder import ResNetDecoderBlock
from .BaseDecoder import BaseDecoder
from tta_uia_segmentation.src.utils.utils import assert_in, default

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class HierarchichalResNetDecoder(BaseDecoder):
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

        super(HierarchichalResNetDecoder, self).__init__(output_size=output_size)
        self._embedding_dim = embedding_dim
        self._hierarchy_level = hierarchy_level
        self._n_classes = n_classes
        self._num_channels_last_upsample = num_channels_last_upsample
        self._in_train_mode = True

        # If number of channels per level is provided, then check it matches the number of levels
        # :====================================================================:
        if num_channels_per_hier is not None:
            assert len(num_channels_per_hier) == hierarchy_level, (
                "Number of channels per level should match the number hierarchi",
            )
        else:
            num_channels_per_hier = [
                embedding_dim / (2**hier_i) for hier_i in range(hierarchy_level)
            ]

        self._num_channels_per_hier = num_channels_per_hier

        # Define modules for each hierarchy level
        # :====================================================================:
        module_dict = {}
        num_channels_per_level = num_channels_per_hier + [num_channels_last_upsample]
        for hier_i in range(hierarchy_level):
            hier_i_modules = {}

            # Append 1x1 conv layer to reduce the number of channels if needed
            if embedding_dim != num_channels_per_level[hier_i]:
                hier_i_modules["conv1"] = get_conv(
                    in_channels=embedding_dim,
                    out_channels=num_channels_per_level[hier_i],
                    kernel_size=1,
                    n_dimensions=n_dimensions,
                )

            # Append ResNetDecoderBlock that
            #  reduces the number of channels by a factor of 2
            #  and upsamples the input by a factor of 2
            hier_i_modules["res_dec_block"] = ResNetDecoderBlock(
                in_channels=num_channels_per_level[hier_i],
                out_channels=num_channels_per_level[hier_i + 1],
                scale_factor=2,
                n_dimensions=n_dimensions,
            )

            module_dict[f"hier_{hier_i}"] = nn.ModuleDict(hier_i_modules)

        self.blocks = module_dict

        # Define modules for last upsampling block and classification head
        # :====================================================================:
        self.last_block = ResNetDecoderBlock(
            in_channels=num_channels_last_upsample,
            out_channels=num_channels_last_upsample / 2,
            scale_factor=None, # Determined dinamically based on the output size
            n_dimensions=n_dimensions,
        )

        self.output_conv = get_conv(
            in_channels=num_channels_last_upsample / 2,
            out_channels=n_classes,
            kernel_size=1,
            n_dimensions=n_dimensions
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:

        # Pass trough decoder blocks for each hierarchy level
        # :====================================================================:
        for hier_i in range(self._hierarchy_level):
            hier_block = self.blocks[f"hier_{hier_i}"]

            if "conv1" in hier_block:
                # Reduce the number of channels if needed
                x_i = hier_block["conv1"](x[hier_i])
            else:
                x_i = x[hier_i]

            # Upsample and map
            x_i = hier_block["res_dec_block"](x_i)

            # Add to previous level, if not the first level
            if hier_i == 0:
                prev_level = x_i
            else:
                x_i += prev_level

        # Last upsampling block to original size
        # :====================================================================:
        x = self.last_block(x, size=self._output_size)

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
