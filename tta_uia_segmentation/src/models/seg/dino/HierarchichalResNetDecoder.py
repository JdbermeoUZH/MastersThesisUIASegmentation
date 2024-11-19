import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..norm_seg.UNet import get_conv, DoubleConv
from .ResNetDecoder import ResNetDecoderBlock
from tta_uia_segmentation.src.utils.utils import assert_in, default

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class HierarchichalResNetDecoder(nn.Module):
    _N_DIM = 2

    def __init__(
        self,
        embedding_dim: int,
        hierarchy_levels: int,
        num_channels_last_upsample: int,
        n_classes: int,
        output_size: tuple[int, int],
        num_channels_per_hier: Optional[list[int]] = None,
    ):

        super(HierarchichalResNetDecoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._hierarchy_levels = hierarchy_levels
        self._n_classes = n_classes
        self._output_size = output_size
        self._num_channels_last_upsample = num_channels_last_upsample
        self._in_train_mode = True

        # If number of channels per level is provided, then check it matches the number of levels
        # :====================================================================:
        if num_channels_per_hier is not None:
            assert len(num_channels_per_hier) == hierarchy_levels, (
                "Number of channels per level should match the number hierarchi",
            )
        else:
            num_channels_per_hier = [
                embedding_dim / (2**hier_i) for hier_i in range(hierarchy_levels)
            ]

        self._num_channels_per_hier = num_channels_per_hier

        # Define modules for each hierarchy level
        # :====================================================================:
        module_dict = {}
        num_channels_per_level = num_channels_per_hier + [num_channels_last_upsample]
        for hier_i in range(hierarchy_levels):
            hier_i_modules = {}

            # Append 1x1 conv layer to reduce the number of channels if needed
            if embedding_dim != num_channels_per_level[hier_i]:
                hier_i_modules["conv1"] = get_conv(
                    in_channels=embedding_dim,
                    out_channels=num_channels_per_level[hier_i],
                    kernel_size=1,
                    n_dimensions=self._N_DIM,
                )

            # Append ResNetDecoderBlock that
            #  reduces the number of channels by a factor of 2
            #  and upsamples the input by a factor of 2
            hier_i_modules["res_dec_block"] = ResNetDecoderBlock(
                in_channels=num_channels_per_level[hier_i],
                out_channels=num_channels_per_level[hier_i + 1],
                scale_factor=2,
                n_dimensions=self._N_DIM,
            )

            module_dict[f"hier_{hier_i}"] = nn.ModuleDict(hier_i_modules)

        self.blocks = module_dict

        # Define modules for last upsampling block and classification head
        # :====================================================================:
        self.last_block = ResNetDecoderBlock(
            in_channels=num_channels_last_upsample,
            out_channels=num_channels_last_upsample / 2,
            scale_factor=None, # Determined dinamically based on the output size
            n_dimensions=self._N_DIM,
        )

        self.output_conv = get_conv(
            in_channels=num_channels_last_upsample / 2,
            out_channels=n_classes,
            kernel_size=1,
            n_dimensions=self._N_DIM
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:

        # Pass trough decoder blocks for each hierarchy level
        # :====================================================================:
        for hier_i, hier_block in enumerate(self._hierarchy_levels):
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
