from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DinoV2FeatureExtractor import DinoV2FeatureExtractor
from ..norm_seg.UNet import get_conv, DoubleConv
from ..BaseSeg import BaseSeg
from tta_uia_segmentation.src.utils.utils import assert_in, default
from tta_uia_segmentation.src.utils.io import save_checkpoint


class DinoSeg(BaseSeg):
    def __init__(
        self,
        decoder: nn.Module,
        dino_fe: Optional[DinoV2FeatureExtractor] = None,
        precalculated_fts: bool = False,
    ):
        super().__init__()

        self._decoder = decoder
        self._dino_fe = dino_fe

        self._precalculated_fts = precalculated_fts

    def _forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the Dino features to a segmentation mask

        Returns:
        --------
        y_mask : torch.Tensor
            Predicted segmentation mask.

        y_logits : torch.Tensor
            Predicted logits.
        """

        return self.decoder(x)

    def _preprocess_x(
        self,
        x,
        mask: Optional[torch.Tensor] = None,
        pre: bool = False,
        hierarchy: list[int] = [0],
    ):

        # Calculate dino features if necessary
        if not self.precalculated_fts:
            # Convert grayscale to RGB, required by DINO
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

            assert (
                self._dino_fe is not None
            ), "Dino feature extractor is required when features are not precalculated"
            dino_out = self._dino_fe(x, mask, pre, hierarchy)
            x = dino_out["patch"].permute(
                0, 3, 1, 2
            )  # N x np x np x df -> N x df x np x np

        return x, {"Dino Features": x}

    def save_checkpoint(self, path: str, **kwargs) -> None:
        save_checkpoint(
            path=path,
            decoder_state_dict=self.decoder.state_dict(),
            dino_model_name=self.dino_fe.model_name,
            **kwargs,
        )

    def load_checkpoint(
        self,
        path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:

        checkpoint = torch.load(path, map_location=device)
        decoder_state_dict = checkpoint["decoder_state_dict"]
        self._decoder.load_state_dict(decoder_state_dict)
        
        # Load Dino feature extractor if specified in the checkpoint
        if "dino_model_name" in checkpoint:
            dino_model_name = checkpoint["dino_model_name"]
            self._dino_fe = DinoV2FeatureExtractor(model=dino_model_name)   

    @property
    def decoder(self):
        return self._decoder

    @property
    def dino_fe(self):
        return self._dino_fe

    @property
    def precalculated_fts(self):
        return self._precalculated_fts

    @precalculated_fts.setter
    def precalculated_fts(self, value: bool):
        self._precalculated_fts = value


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

    def forward(self, x, scale_factor=None):
        scale_factor = default(scale_factor, self.scale_factor)

        if isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.cpu().numpy().tolist()

        x = F.interpolate(
            x,
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

        self.output_size = output_size
        self.last_block = ResNetDecoderBlock(
            channels[-2], channels[-1], None, n_dimensions=n_dimensions
        )

        self.output_conv = get_conv(
            channels[-1], n_classes, 1, n_dimensions=n_dimensions
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass trough decoder blocks
        for block in self.blocks:
            x = block(x)

        # Upsample to original size
        scale_factor = torch.tensor(self.output_size) / torch.tensor(x.shape[-2:])
        assert (scale_factor < 2.0).any(), (
            f"upsampling is too large: {scale_factor}, "
            + "add another block to the decoder instead"
        )

        x = self.last_block(x, scale_factor=scale_factor)

        # Output to n_classes
        x = self.output_conv(x)

        return self.softmax(x), x
