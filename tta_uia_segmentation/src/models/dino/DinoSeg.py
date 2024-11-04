from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DinoV2FeatureExtractor import DinoV2FeatureExtractor
from ..norm_seg.UNet import Decoder, get_conv, DoubleConv
from ...utils.utils import assert_in, default


class DinoSeg(nn.Module):
    def __init__(
            self,
            decoder: nn.Module,
            dino_fe: Optional[DinoV2FeatureExtractor] = None,
            features_are_precalculated: bool = False,
            ):
        super().__init__()

        self._decoder = decoder
        self._dino_fe = dino_fe
        
        self._features_are_precalculated = features_are_precalculated

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None, pre: bool = True, hierarchy: list[int] = [0]) -> torch.Tensor:
        if not self._features_are_precalculated:
            assert self._dino_fe is not None, "Dino feature extractor is required when features are not precalculated"
            dino_out = self._dino_fe(image, mask, pre, hierarchy)
            image = ... # Get the dino features

        # Decode features
        self._decoder(image)
    
    @property
    def decoder(self):
        return self._decoder
    
    @property
    def dino_fe(self):
        return self._dino_fe


class ResNetDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            scale_factor: Optional[float] = 2.,
            n_dimensions=2
        ):
        super().__init__()
        
        assert_in(n_dimensions, 'n_dimensions', [1, 2, 3])

        if n_dimensions == 1:
            self.interpolation_mode = 'linear'
        elif n_dimensions == 2:
            self.interpolation_mode = 'bilinear'
        else:
            self.interpolation_mode = 'trilinear'

        self.scale_factor = scale_factor

        self.residual_conv = get_conv(
            in_channels, out_channels, 1, n_dimensions=n_dimensions)

        self.double_conv = DoubleConv(
            in_channels, out_channels, n_dimensions=n_dimensions)

    def forward(self, x, scale_factor=None):
        scale_factor = default(scale_factor, self.scale_factor)

        if isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.cpu().numpy().tolist()

        x = F.interpolate( 
            x, 
            scale_factor=scale_factor, 
            mode=self.interpolation_mode, 
            align_corners=True
        )
        
        x = self.double_conv(x) + self.residual_conv(x)
        
        return x


class ResNetDecoder(nn.Module):
    def __init__( 
            self,
            n_classes: int,
            output_size: tuple[int, int],
            channels=[128, 64, 32, 16],
            n_dimensions=2):
        
        super().__init__()

        self.blocks = nn.ModuleList(
            [ResNetDecoderBlock(channels[i], channels[i+1], 2, n_dimensions=n_dimensions)
             for i in range(len(channels) - 2)]
        )

        self.output_size = output_size        
        self.last_block = ResNetDecoderBlock(
            channels[-2], channels[-1], None,
            n_dimensions=n_dimensions)

        self.output_conv = get_conv(channels[-1], n_classes, 1, n_dimensions=n_dimensions)

    def forward(self, x):
        # Pass trough decoder blocks
        for block in self.blocks:
            x = block(x)

        # Upsample to original size
        scale_factor = torch.tensor(self.output_size) / torch.tensor(x.shape[-2:])
        assert (scale_factor < 2.).any(), \
            f"upsampling is too large: {scale_factor}, " + \
            "add another block to the decoder instead"
        
        x = self.last_block(x, scale_factor=scale_factor)

        # Output to n_classes
        x = self.output_conv(x)

        return x
