import random
import itertools
from functools import partial
from tqdm import tqdm
from typing import Optional, Literal, Union

import torch
from torch.nn.functional import mse_loss
from diffusers import (
    UNet2DConditionModel,
    UNet2DModel,
    DDPMScheduler,
    CosineDPMSolverMultistepScheduler,
    DDIMScheduler,
    SchedulerMixin,
)
from diffusers.training_utils import compute_snr

from tta_uia_segmentation.src.models import BaseConditionalGaussianDiffusion
from tta_uia_segmentation.src.models.ddpm.utils import (
    sample_t,
    sample_noise,
    generate_unconditional_mask,
    normalize,
    unnormalize,
)
from tta_uia_segmentation.src.utils.utils import default

objective_to_prediction_type = {
    "pred_noise": "epsilon",
    "pred_x_t_m_1": "sample",
    "pred_v": "v_prediction",
    "pred_x0": "pred_original_sample",
}


class ConditionalGaussianDiffusion(BaseConditionalGaussianDiffusion):
    """
    Conditional Latent GaussianDiffusion model

    Attributes:
    ----------


    """

    def __init__(
        self,
        unet: UNet2DConditionModel | UNet2DModel,
        train_image_size: int,
        img_channels: int,
        cond_img_channels: int,
        beta_schedule: Literal["cosine", "linear", "sigmoid"] = "sigmoid",
        objective: Literal["pred_noise", "pred_xt_m_1", "pred_v", "pred_x0"] = "pred_v",
        forward_type: Literal["train_loss", "model_output", "sds"] = "model_output",
        num_train_timesteps: int = 1000,
        num_sample_timesteps: int = 100,
        unconditional_rate: Optional[float] = None,
        w_cfg: float = 0.0,
        cfg_rescale: float = 0.0,
        cond_type: Literal["concat"] = "concat",
        clamp_after_norm: bool = True,
        snr_weighting_gamma: Optional[float] = None,
        rescale_betas_zero_snr: bool = False,
        mixed_precision: Optional[Literal["fp16", "bf16"]] = None,
    ):

        # super(ConditionalLatentGaussianDiffusion, self).__init__()
        super(ConditionalGaussianDiffusion, self).__init__()

        # Check the image is large enough for the downsample blocks
        self._num_downsamples_unet = len(unet.config.down_block_types) - 1
        min_img_size = 2 ** (self._num_downsamples_unet)
        assert (
            train_image_size >= min_img_size
        ), f"Image size must be at least {min_img_size}"

        self._unet = unet

        # If network uses cross attention, create an encoding
        #  that matches the cross_attention_dim
        if isinstance(self._unet, UNet2DConditionModel):
            self._uses_x_attention = True

            # TODO: define module that maps from mask encoding network to cross attention dim adding first 2D positional embeddings, then flattening, and then linear layer to cross_attention_dim
            self._cond_emb_flat = None
        else:
            self._uses_x_attention = False

        # Parameters of the training
        self._train_image_size = train_image_size
        self._num_train_timesteps = num_train_timesteps

        # Training objective
        self._objective = objective

        # Noise schedule
        noise_schedule_args = dict(
            num_train_timesteps=num_train_timesteps,
            prediction_type=objective_to_prediction_type[objective],
        )
        if beta_schedule == "cosine":
            self._noise_scheduler = CosineDPMSolverMultistepScheduler(
                **noise_schedule_args
            )
        elif beta_schedule == "linear":
            self._noise_scheduler = DDPMScheduler(
                beta_schedule="linear", **noise_schedule_args
            )
        elif beta_schedule == "sigmoid":
            self._noise_scheduler = DDPMScheduler(
                beta_schedule="sigmoid", **noise_schedule_args
            )
        else:
            raise ValueError("Unknown beta_schedule")

        # Parameters of the model
        self._num_sample_timesteps = num_sample_timesteps

        self._img_channels = img_channels

        self._cond_type = cond_type
        self._cond_img_channels = cond_img_channels

        self._forward_type = forward_type
        self._clamp_after_norm = clamp_after_norm

        self._w_cfg = w_cfg
        self._cfg_rescale = cfg_rescale

        self._rescale_betas_zero_snr = rescale_betas_zero_snr
        self._snr_weighting_gamma = snr_weighting_gamma

        # Unconditional training rate parameters
        if unconditional_rate is not None:
            assert (
                unconditional_rate >= 0 and unconditional_rate <= 1
            ), "uncoditional_rate must be between 0 and 1"
            self._only_unconditional = unconditional_rate == 1
            self._also_unconditional = unconditional_rate > 0
        else:
            self._only_unconditional = False
            self._also_unconditional = False

        self.unconditional_rate = unconditional_rate

        # Mixed precision attributes
        if mixed_precision is not None:
            self._set_mixed_precision_attributes(mixed_precision)

    @property
    def img_dtype(self):
        return self._unet.dtype

    @property
    def cond_img_dtype(self):
        return torch.float32

    @property
    def num_train_timesteps(self):
        return self._num_train_timesteps

    @property
    def num_sample_timesteps(self):
        return self._num_sample_timesteps

    @property
    def device(self):
        return self._unet.device

    @property
    def train_image_size(self):
        return self._train_image_size

    @property
    def num_img_channels(self):
        return self._img_channels

    @property
    def also_unconditional(self):
        return self._also_unconditional

    @property
    def use_x_attention(self):
        return self._uses_x_attention

    @property
    def objective(self):
        return self._objective

    def train_mode(self):
        """
        Set the model in train mode
        """
        self._unet.train()

        if self._uses_x_attention:
            self._cond_emb_flat.train()

    def eval_mode(self):
        """
        Set the model in evaluation mode
        """
        self._unet.eval()

        if self._uses_x_attention:
            self._cond_emb_flat.eval()

    def get_modules_to_train(self):
        """
        Get the modules to train
        """
        modules_to_train = [self._unet]

        if self._uses_x_attention:
            modules_to_train.append(self._cond_emb_flat)

        return modules_to_train

    def get_params_of_modules_to_train(self):
        """
        Get the parameters of the modules to train
        """
        params_to_train = []
        for module in self.get_modules_to_train():
            params_to_train.extend(module.parameters())

        return itertools.chain(*params_to_train)

    def forward(
        self,
        img: torch.Tensor,
        cond_img: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        min_t: Optional[int] = None,
        max_t: Optional[int] = None,
        return_dict: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor:
        """
        Evaluate train_loss or tta loss for the given inputs

        Parameters
        ----------
        img : torch.Tensor
            The input image. Assumed to be normalized between 0 and 1. If not, it will print a warning.
        cond_img : torch.Tensor
            The conditional segmentation image. Assumed to be one hot encoded with the background class
             as an explicit class, thus normalized between 0 and 1. If not, it will print a warning.
        t : torch.Tensor, optional
            The time step tensor. If not passed, one is uniformly sampled.
        noise : torch.Tensor, optional
            The noise tensor. If not passed, one is sampled from a standard normal distribution.
        min_t : int, optional
            The minimum time step value to sample from.
        max_t : int, optional
            The maximum time step value to sample from.

        Returns
        -------
        dict[str, torch.Tensor] | tuple[torch.Tensor, ...]
            If 'forward_type' is 'model_output', returns a dictionary with the model prediction, time step and noise.
            If 'forward_type' is 'train_loss', returns the train_loss depending on the objective chosen.
            If 'forward_type' is 'sds', returns the sds loss.
        """
        assert (
            cond_img.shape[-2:] == img.shape[-2:]
        ), "cond_img and img must have the same shape in H and W"
        (
            batch_size,
            _,
            h,
            w,
        ) = img.shape
        num_channels_cond_img = cond_img.shape[1]

        also_return_t, also_return_noise = False, False
        output_dict = {}

        if num_channels_cond_img != self._cond_img_channels:
            print(
                f"Warning: cond_img has {num_channels_cond_img} channels, but DDPM trained on {self._cond_img_channels}"
            )

        if h != self._train_image_size or w != self._train_image_size:
            print(
                f"Warning: img has shape {h}x{w}, but DDPM trained on {self._train_image_size}x{self._train_image_size}"
            )

        # Preprocess images and conditioning to be in the correct range (-1, 1)
        if img.max() > 1 or img.min() < 0:
            print(
                "Warning: img is not normalized between 0 and 1 "
                f"max_value: {img.max()}, min_value: {img.min()}"
            )

        if cond_img.max() > 1 or cond_img.min() < 0:
            print(
                "Warning: cond_img is not normalized between 0 and 1"
                f"torch.unique(cond_img): {torch.unique(cond_img)}"
            )

        # If the model has an unconditional rate, randomly use the unconditional mask
        if self._only_unconditional or (
            self._also_unconditional and random.random() < self.unconditional_rate
        ):
            cond_img = generate_unconditional_mask(
                (batch_size, num_channels_cond_img, h, w), device=cond_img.device
            )

        # If no time step is given, sample it
        if t is None:
            also_return_t = True
            min_t = default(min_t, 0)
            max_t = default(max_t, self._num_train_timesteps)
            t = sample_t(min_t, max_t, batch_size, self.device)
        else:
            if min_t is not None and max_t is not None:
                assert (min_t <= t).all() and (
                    t <= max_t
                ).all(), f"t must be between {min_t} and {max_t}, but got {t}"

        # Noise the image latent
        #  If no noise is given, sample it
        if noise is None:
            also_return_noise = True
            noise = sample_noise(img.shape, self.device)

        noised_img_latents = self._noise_scheduler.add_noise(img, noise, t)

        # Join the latents with the conditioning image latents
        img_w_conditioning = self._apply_conditioning_to_input(
            noised_img_latents, cond_img
        )

        # Make the prediction with the model (typically noise or velocity estimation)
        model_output = self._unet(img_w_conditioning, t, return_dict=False)[0]

        if self._forward_type == "model_output":
            output_dict["model_pred"] = model_output
            if also_return_t:
                output_dict["t"] = t

            if also_return_noise:
                output_dict["noise"] = noise

        if self._forward_type == "train_loss":
            output_dict["loss"] = self._train_loss(img, model_output, t, noise)

        if self._forward_type == "sds":
            raise NotImplementedError("sds not implemented yet")

        if return_dict:
            return output_dict
        else:
            return (
                output_dict.values()
                if len(output_dict) > 1
                else list(output_dict.values())[0]
            )

    def _train_loss(
        self,
        img_latents: torch.Tensor,
        model_output: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ):
        """
        Compute the train_loss for the given inputs

        Parameters
        ----------
        input_latents : torch.Tensor
            The input latents tensor.
        model_output : torch.Tensor
            The output of the denoiser model.
        t : torch.Tensor
            The time step tensor.
        noise : torch.Tensor
            The noise tensor.
        """
        # Define the target
        if self._objective == "pred_noise":
            target = noise
        elif self._objective == "pred_v":
            target = self._noise_scheduler.get_velocity(img_latents, noise, t)
        elif self._objective == "pred_x_t_m_1":
            target = self._noise_scheduler.add_noise(
                img_latents, noise, (t - 1).clamp(min=0)
            )
        elif self._objective == "pred_x0":
            target = img_latents

        # Compute MSE with SNR weighting if gamma is set
        if self._snr_weighting_gamma is not None:
            return self._mse_loss_with_snr_weighting(model_output, target, t)
        else:
            return mse_loss(model_output.float(), target.float(), reduction="mean")

    @torch.inference_mode()
    def ddim_sample(
        self,
        img_shape,
        x_cond,
        unconditional_sampling: bool = False,
        return_all_timesteps: bool = False,
        num_sample_timesteps: Optional[int] = None,
        w_cfg: Optional[float] = None,
        show_progress: bool = False,
        **kwargs,
    ):

        if "rescale_betas_zero_snr" in kwargs:
            rescale_betas_zero_snr = kwargs.pop("rescale_betas_zero_snr")
        else:
            rescale_betas_zero_snr = self._rescale_betas_zero_snr

        ddim_scheduler = DDIMScheduler.from_config(
            self._noise_scheduler.config,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
            **kwargs,
        )

        return self.sample(
            img_shape,
            x_cond,
            unconditional_sampling=unconditional_sampling,
            return_all_timesteps=return_all_timesteps,
            sample_scheduler=ddim_scheduler,
            num_sample_timesteps=num_sample_timesteps,
            w_cfg=w_cfg,
            show_progress=show_progress,
        )

    @torch.inference_mode()
    def ddpm_sample(
        self,
        img_shape,
        x_cond,
        beta_schedule: Optional[Literal["cosine", "linear", "sigmoid"]] = None,
        unconditional_sampling: bool = False,
        return_all_timesteps: bool = False,
        num_sample_timesteps: Optional[int] = None,
        w_cfg: Optional[float] = None,
        show_progress: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:

        if isinstance(self._noise_scheduler, DDPMScheduler):
            beta_schedule = default(beta_schedule, self._noise_scheduler.beta_schedule)
        else:
            beta_schedule = default(beta_schedule, "linear")

        if "rescale_betas_zero_snr" in kwargs:
            rescale_betas_zero_snr = kwargs.pop("rescale_betas_zero_snr")
        else:
            rescale_betas_zero_snr = self._rescale_betas_zero_snr

        ddpm_scheduler = DDPMScheduler.from_config(
            self._noise_scheduler.config,
            beta_schedule=beta_schedule,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
            **kwargs,
        )

        return self.sample(
            img_shape,
            x_cond,
            unconditional_sampling=unconditional_sampling,
            return_all_timesteps=return_all_timesteps,
            sample_scheduler=ddpm_scheduler,
            num_sample_timesteps=num_sample_timesteps,
            w_cfg=w_cfg,
            show_progress=show_progress,
        )

    @torch.inference_mode()
    def sample(
        self,
        img_shape: tuple[int, int, int, int],
        x_cond: torch.Tensor,
        unconditional_sampling: bool = False,
        return_all_timesteps: bool = False,
        sample_scheduler: Optional[SchedulerMixin] = None,
        num_sample_timesteps: Optional[int] = None,
        w_cfg: Optional[float] = None,
        show_progress: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:

        if self._objective in ["pred_x0"]:
            raise NotImplementedError(
                "Sampling not implmented yet when"
                'model predicts x0, objective="pred_x0"'
            )

        w_cfg = default(w_cfg, self._w_cfg)
        with_cfg = w_cfg > 0

        # If a sample scheduler is provided, use it
        noise_scheduler = default(sample_scheduler, self._noise_scheduler)

        # Set the noise scheduler inference timesteps
        num_sample_timesteps = default(num_sample_timesteps, self.num_sample_timesteps)
        noise_scheduler.set_timesteps(num_sample_timesteps)
        timesteps = noise_scheduler.timesteps

        # Get the latent noise at step T
        latents = self._prepare_latent_noise(
            batch_size=img_shape[0],
            height=img_shape[2],
            width=img_shape[3],
            device=self.device,
        )

        # Get an unconditional mask if needed
        if unconditional_sampling:
            x_cond = generate_unconditional_mask(
                x_cond.shape, device=self.device, dtype=x_cond.dtype
            )

        # If using cfg, create a batch with the conditional and unconditional masks (prompts)
        x_cond = x_cond.to(self.device)

        if with_cfg:
            assert (
                not unconditional_sampling
            ), "Cannot sample with cfg when unconditional_sampling is True"
            x_cond = torch.cat(
                [x_cond, generate_unconditional_mask(x_cond.shape, device=self.device)]
            )

        # Sample the latents and apply denoising schedule or solver
        latent_history = []
        for t in tqdm(timesteps, disable=not show_progress):
            # If using cfg, batch the forward pass for the conditional and unconditional prompts
            latents = latents.repeat(2, 1, 1, 1) if with_cfg else latents
            latents = noise_scheduler.scale_model_input(latents, t)

            # Apply the conditioning to the latents
            latents_w_conditioning = self._apply_conditioning_to_input(latents, x_cond)

            # predict noise model_output
            model_pred = self._unet(latents_w_conditioning, t, return_dict=False)[0]

            # Correct the models prediction with CFG
            if with_cfg:
                model_pred_cond, model_pred_uncond = model_pred.chunk(2)
                model_pred = model_pred_cond + w_cfg * (
                    model_pred_cond - model_pred_uncond
                )

                if self._cfg_rescale > 0:
                    model_pred = self._rescale_noise_cfg(
                        model_pred, model_pred_cond, self._cfg_rescale
                    )

            # 2. compute previous image: x_t -> x_s, where s < t
            latents = noise_scheduler.step(model_pred, t, latents).prev_sample

            if return_all_timesteps:
                latent_history.append(latents)

        if return_all_timesteps:
            return latent_history

        return latents

    def _prepare_latent_noise(self, batch_size, height, width, device, latents=None):
        shape = (batch_size, self._img_channels, int(height), int(width))

        # Sample the initial noise
        latents = sample_noise(shape, device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self._noise_scheduler.init_noise_sigma

        return latents

    def _apply_conditioning_to_input(
        self, input: torch.Tensor, cond_img: torch.Tensor
    ) -> torch.Tensor:
        if self._cond_type == "concat":
            return torch.cat([input, cond_img], dim=1)
        else:
            raise ValueError("Unknown cond_type")

    def _rescale_noise_cfg(noise_cfg, noise_pred_cond, guidance_rescale=0.7):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4

        Recommended value for `guidance_rescale` is 0.7 when using w_cfg = 7.5 (emperically determined on natural images)
        """
        std_text = noise_pred_cond.std(
            dim=list(range(1, noise_pred_cond.ndim)), keepdim=True
        )
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = (
            guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        )

        return noise_cfg

    def _mse_loss_with_snr_weighting(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the MSE loss with SNR weighting as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            from: https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py#L1298
        """
        assert self.objective in [
            "pred_noise",
            "pred_v",
        ], "Only epsilon, v_prediction are supported for SNR weighting"
        snr = compute_snr(self._noise_scheduler, t)
        base_weight = (
            torch.stack([snr, self._snr_gamma * torch.ones_like(t)], dim=1).min(dim=1)[
                0
            ]
            / snr
        )

        if self._objective == "pred_v":
            # Velocity objective needs to be floored to an SNR weight of one.
            mse_loss_weights = base_weight + 1
        else:
            # Epsilon and sample both use the same loss weights.
            mse_loss_weights = base_weight

        loss = mse_loss(model_output.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights

        return loss.mean()

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        return normalize(img, clamp=self._clamp_after_norm)

    def _unnormalize(self, img: torch.Tensor) -> torch.Tensor:
        return unnormalize(img, clamp=self._clamp_after_norm)

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing for the model
        """
        self._unet.enable_gradient_checkpointing()

    def move_non_trainable_params_to(
        self,
        device: Optional[str | torch.device] = None,
        mixed_precision_type: Optional[Literal["fp16", "bf16"]] = None,
    ):
        raise NotImplementedError("The model does not have non-trainable modules")
