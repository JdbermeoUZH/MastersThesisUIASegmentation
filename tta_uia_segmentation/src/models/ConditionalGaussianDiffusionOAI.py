import os
import sys

import torch

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

from tta_uia_segmentation.src.models.UNetModelOAI import UNetModelConditionedOnSegMask
from improved_diffusion import dist_util
from improved_diffusion.respace import SpacedDiffusion
from improved_diffusion.resample import LossAwareSampler, ScheduleSampler


def diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        sigma_small=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
    )


class ConditionalGaussianDiffusionOAI(torch.nn.Module):
    def __init__(
        self,
        model: UNetModelConditionedOnSegMask,
        ddpm: SpacedDiffusion,
        schedule_sampler: ScheduleSampler,
        device: torch.device,
        ):
        super().__init__()
        self.model = model
        self.ddpm = ddpm
        self.schedule_sampler = schedule_sampler
        self.device = device
        
        dist_util.setup_dist()
    
    def forward(self, img, cond_img):
        t, weights = self.schedule_sampler.sample(img.shape[0], self.device)
        
        # Return the estimation loss of the diffusion model
        losses = self.ddpm.training_losses(self.model, img, t,  model_kwargs={"x_cond": cond_img})

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )
        
        loss = (losses["loss"] * weights).mean()
            
        return loss
                
    def p_sample_loop(self, img_shape, x_cond, **kwargs):
        return self.ddpm.p_sample_loop(
            self.model, img_shape, model_kwargs={"x_cond": x_cond},
            **kwargs
            )

    def ddim_sample_loop(self, img_shape, x_cond, **kwargs):
        return self.ddpm.ddim_sample_loop(
            self.model, img_shape, model_kwargs={"x_cond": x_cond},
            **kwargs
            )
        
    def set_sampling_timesteps(self, sampling_timesteps):
        return 