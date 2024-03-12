from typing import Optional

import torch

from improved_diffusion.unet import UNetModel


def create_model_conditioned_on_seg_mask(
    image_size,
    image_channels: int,
    seg_cond: bool,
    num_channels,
    channel_mult,
    num_res_blocks,
    learn_sigma,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    n_classes: Optional[int] = None,   
):
    if seg_cond and n_classes is None:
        raise ValueError("num_classes must be specified if seg_cond is True.")
    
    in_channels = image_channels + (n_classes if seg_cond else 0)
    out_channels = image_channels * (1 if not learn_sigma else 2)
    
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModelConditionedOnSegMask(
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        num_classes = None # To not use class conditioning
    )
    
    
class UNetModelConditionedOnSegMask(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, x, timesteps, seg_mask=None):
        if seg_mask is not None:
            x = torch.cat([x, seg_mask], dim=1)
        return super().forward(x, timesteps)

