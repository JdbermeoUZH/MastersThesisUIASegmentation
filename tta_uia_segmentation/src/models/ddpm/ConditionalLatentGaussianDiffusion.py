import random
from tqdm import tqdm
from typing import Optional, Literal, Union

import torch
from torch.nn.functional import mse_loss
from diffusers import (
    UNet2DConditionModel,
    UNet2DModel,
    AutoencoderKL, 
    DDPMScheduler,
    CosineDPMSolverMultistepScheduler,
    DDIMScheduler,
    SchedulerMixin,
)
from diffusers.models.controlnet import ControlNetConditioningEmbedding

from tta_uia_segmentation.src.models.ddpm import ConditionalGaussianDiffusionInterface
from tta_uia_segmentation.src.models.ddpm.utils import (
    sample_t, sample_noise, generate_unconditional_mask,
    normalize, unnormalize)
from tta_uia_segmentation.src.utils.utils import default
from diffusers.training_utils import compute_snr

objective_to_prediction_type = {
    'pred_noise': 'epsilon', 
    'pred_x_t_m_1': 'sample',
    'pred_v': 'v_prediction',
    'pred_x0': 'pred_original_sample'
    }



class ConditionalLatentGaussianDiffusion(ConditionalGaussianDiffusionInterface):
    """
    Conditional Latent GaussianDdiffusion model
    
    Based on the interface from https://github.com/lucidrains/denoising-diffusion-pytorch

    Attributes:
    ----------
        
    """
    
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel | UNet2DModel,
        image_size: int,
        cond_img_channels: int,
        beta_schedule: Literal['cosine', 'linear', 'sigmoid'] = 'sigmoid',
        objective: Literal['pred_noise', 'pred_xt_m_1', 'pred_v', 'pred_x0'] = 'pred_v',
        forward_type: Literal['model_output', 'train_loss', 'sds'] = 'model_output',
        num_train_timesteps: int = 1000,
        num_sample_timesteps: int = 100,
        unconditional_rate: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        w_cfg: float = 0.0,
        cfg_rescale: float = 0.0, 
        use_offset_noise: bool = False,
        fit_emb_for_cond_img: bool = True,
        cond_type: Literal['concat', 'sum'] = 'sum',
        clamp_after_norm: bool = True,
        snr_weighting_gamma: Optional[float] = None,
        reset_betas_zero_snr: bool = False,
        ):
 
        super().__init__()

        # Check the image is large enough for the downsample blocks
        self._num_downsamples_vae = len(vae.config.down_block_types)
        self._num_downsamples_unet = len(unet.down_block_types) - 1
        min_img_size = 2 ** (self._num_downsamples_vae + self._num_downsamples_unet)
        assert image_size >= min_img_size, f"Image size must be at least {min_img_size}"

        self._vae = vae
        self._unet = unet

        # Freeze the VAE
        self._vae.requires_grad_(False)
        
        # Define encoding model for the conditioning image
        self._fit_emb_for_cond_img = fit_emb_for_cond_img
        if fit_emb_for_cond_img:
            # Learn an embedding for the conditioning image as done in
            #  ControlNet: https://arxiv.org/pdf/2302.05543
            self._cond_emb = ControlNetConditioningEmbedding(
                conditioning_channels=cond_img_channels,
                conditioning_embedding_channels=self._vae.config.in_channels,
                block_out_channels=[16, 32, 96, 256][-self._num_downsamples_vae:],
            )
        else:
            # Use the VAE to encode the conditioning image
            self._cond_emb = self._vae

        # Parameters of the training
        self._device = device
        self._train_image_size = image_size
        self._num_train_timesteps = num_train_timesteps

        # Training objective
        self._objective = objective

        # Noise schedule
        noise_schedule_args = dict(
            num_train_timesteps=num_train_timesteps,
            prediction_type=objective_to_prediction_type[objective],
        )
        if beta_schedule == 'cosine':
            self._noise_scheduler = CosineDPMSolverMultistepScheduler(
                **noise_schedule_args)
        elif beta_schedule == 'linear':
            self._noise_scheduler = DDPMScheduler(
                beta_schedule='linear', **noise_schedule_args)
        elif beta_schedule == 'sigmoid':
            self._noise_scheduler = DDPMScheduler(
                beta_schedule='sigmoid', **noise_schedule_args)
        else:
            raise ValueError('Unknown beta_schedule')

        # Parameters of the model
        self._num_sample_timesteps = num_sample_timesteps
        
        self._cond_type = cond_type
        self._cond_img_channels = cond_img_channels
        
        self._forward_type = forward_type
        self._clamp_after_norm = clamp_after_norm
        
        self._w_cfg = w_cfg
        self._cfg_rescale = cfg_rescale
        
        self._use_offset_noise = use_offset_noise
        self._reset_betas_zero_snr = reset_betas_zero_snr 
        self._snr_weighting_gamma = snr_weighting_gamma

        # Unconditional training rate parameters
        if unconditional_rate is not None:
            assert unconditional_rate >= 0 and unconditional_rate <= 1, \
                 'uncoditional_rate must be between 0 and 1'
            self._only_unconditional = unconditional_rate == 1
            self._also_unconditional = unconditional_rate > 0
        else:
            self._only_unconditional = False
            self._also_unconditional = False

        self.unconditional_rate = unconditional_rate

    @property
    def _num_train_timesteps(self):
        return self._num_train_timesteps

    @property
    def _num_sample_timesteps(self):
        return self._num_sample_timesteps

    @property
    def _device(self):
        return self._device

    @property
    def _train_image_size(self):
        return self._train_image_size     
    
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
        assert cond_img.shape[-2:] == img.shape[-2:], 'cond_img and img must have the same shape in H and W'
        batch_size, _, h, w, = img.shape
        num_channels_cond_img = cond_img.shape[1]

        also_return_t, also_return_noise = False, False
        output_dict = {}

        if num_channels_cond_img != self._cond_img_channels:
            print(f'Warning: cond_img has {num_channels_cond_img} channels, but DDPM trained on {self._cond_img_channels}')

        if h != self._train_image_size or w != self._train_image_size:
            print(f'Warning: img has shape {h}x{w}, but DDPM trained on {self._train_image_size}x{self._train_image_size}')

        # If no noise is given, sample it
        if noise is None:
            also_return_noise = True
            noise = sample_noise(img.shape, self._use_offset_noise, self._device)

        # If no time step is given, sample it
        if t is None:
            also_return_t = True
            min_t = default(min_t, 0)
            max_t = default(max_t, self._num_train_timesteps)
            t = sample_t(min_t, max_t, batch_size, self._device)
        else:
            if min_t is not None and max_t is not None:
                assert (min_t <= t).all() and (t <= max_t).all(), f't must be between {min_t} and {max_t}, but got {t}'

        # Preprocess images and conditioning to be in the correct range (-1, 1)
        if img.max() > 1 or img.min() < 0:
            print('Warning: img is not normalized between 0 and 1 '
                  f'max_value: {img.max()}, min_value: {img.min()}')
            
        if cond_img.max() > 1 or cond_img.min() < 0:
            print('Warning: cond_img is not normalized between 0 and 1'
                f'torch.unique(cond_img): {torch.unique(cond_img)}')

        # If the model has an unconditional rate, randomly use the unconditional mask
        if self._only_unconditional or \
            (self._also_unconditional and random.random() < self.unconditional_rate):
            cond_img = generate_unconditional_mask(
                (batch_size, num_channels_cond_img, h, w),
                device=cond_img.device)

        # Encode the image and conditioning image 
        img_latents = self._encode_image(img)
        cond_img_latents = self._encode_cond_img(cond_img)

        # Noise the image latent
        noised_img_latents = self._noise_scheduler.add_noise(
            img_latents, noise, t)

        # Join the latents with the conditioning image latents
        latents = self._apply_conditioning_to_latents(
            noised_img_latents, cond_img_latents)

        # Make the prediction with the model (typically noise or velocity estimation)
        model_output = self._unet(latents, t, return_dict=False)[0]
        
        if self._forward_type == 'model_output':
            output_dict['model_pred'] = model_output
            if also_return_t:
                output_dict['t'] = t

            if also_return_noise:
                output_dict['noise'] = noise
        
        if self._forward_type == 'train_loss':
            output_dict['loss'] = self._train_loss(
                img_latents, latents, model_output, t, noise)
        
        if self._forward_type == 'sds':
            raise NotImplementedError('sds not implemented yet')

        if return_dict:
            return output_dict
        else:
            return output_dict.values() if len(output_dict) > 1 \
                else list(output_dict.values())[0]

    def _train_loss(
        self,
        img_latents: torch.Tensor,
        model_input: torch.Tensor,
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
        if self._objective == 'pred_noise':
            target = noise
        elif self._objective == "pred_v":
            target = self._noise_scheduler.get_velocity(model_input, noise, t)
        elif self._objective == "pred_x_t_m_1":
            target = self._noise_scheduler.add_noise(
                img_latents, noise, (t - 1).clamp(min=0))
        elif self._noise_scheduler.config.prediction_type == "pred_x0":
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
        ):
        
        ddim_scheduler = DDIMScheduler.from_config(
            self._noise_scheduler.config,
            rescale_betas_zero_snr=self._reset_betas_zero_snr
            )

        return self.sample(
            img_shape,
            x_cond,
            unconditional_sampling=unconditional_sampling,
            return_all_timesteps=return_all_timesteps,
            sample_scheduler=ddim_scheduler,
            num_sample_timesteps=num_sample_timesteps,
            w_cfg=w_cfg
        ) 

    @torch.inference_mode()
    def ddpm_sample(
        self,
        img_shape,
        x_cond,
        beta_schedule: Optional[Literal['cosine', 'linear', 'sigmoid']] = None,
        unconditional_sampling: bool = False,
        return_all_timesteps: bool = False,
        num_sample_timesteps: Optional[int] = None,
        w_cfg: Optional[float] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        
        if isinstance(self._noise_scheduler, DDPMScheduler):
            beta_schedule = default(beta_schedule, self._noise_scheduler.beta_schedule)
        else:
            beta_schedule = default(beta_schedule, 'linear')

        ddpm_scheduler = DDPMScheduler.from_config( 
            self._noise_scheduler.config,
            beta_schedule=beta_schedule,
            rescale_betas_zero_snr=self._reset_betas_zero_snr,
        )

        return self.sample(
            img_shape,
            x_cond,
            unconditional_sampling=unconditional_sampling,
            return_all_timesteps=return_all_timesteps,
            sample_scheduler=ddpm_scheduler,
            num_sample_timesteps=num_sample_timesteps,
            w_cfg=w_cfg
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
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        
        if self._objective in ['pred_x0']:
            raise NotImplementedError('Sampling not implmented yet when' 
                                      'model predicts x0, objective="pred_x0"')

        w_cfg = default(w_cfg, self._w_cfg)
        with_cfg = w_cfg > 0
        
        # If a sample scheduler is provided, use it
        noise_scheduler = default(sample_scheduler, self._noise_scheduler)

        # Set the noise scheduler inference timesteps
        num_sample_timesteps = default(num_sample_timesteps, self._num_sample_timesteps)
        noise_scheduler.set_timesteps(num_sample_timesteps)
        timesteps = noise_scheduler.timesteps

        # Get the latent noise at step T 
        latents = self._prepare_latents(
            batch_size=img_shape[0], height=img_shape[2], width=img_shape[3],
            device=self._device)

        # Get an unconditional mask if needed
        if unconditional_sampling:
            x_cond = generate_unconditional_mask(
                x_cond.shape, device=self._device)

        # If using cfg, create a batch with the conditional and unconditional masks (prompts)
        x_cond = x_cond.to(self._device)

        if with_cfg:
            assert not unconditional_sampling, 'Cannot sample with cfg when unconditional_sampling is True'
            x_cond = torch.cat([
                x_cond, 
                generate_unconditional_mask(x_cond.shape, device=self._device)
                ])

        # Encode the conditioning 
        x_cond_latents = self._encode_cond_img(x_cond)

        # Sample the latents and apply denoising schedule or solver
        latent_history = []
        for t in tqdm(timesteps):
            # If using cfg, batch the forward pass for the conditional and unconditional prompts
            latents = latents.repeat(2, 1, 1, 1) if with_cfg else latents
            latents = noise_scheduler.scale_model_input(latents, t)

            # Apply the conditioning to the latents
            latents = self._apply_conditioning_to_latents(
                latents, x_cond_latents)

             # predict noise model_output
            model_pred = self.unet(latents, t, return_dict=False)[0]

            # Correct the models prediction with CFG
            if with_cfg:
                model_pred_cond, model_pred_uncond = model_pred.chunk(2)
                model_pred = model_pred_cond + w_cfg * (model_pred_cond - model_pred_uncond)

                if self._cfg_rescale > 0:
                    model_pred = self._rescale_noise_cfg(
                        model_pred, model_pred_cond, self._cfg_rescale)

            # 2. compute previous image: x_t -> x_s, where s < t
            latents = noise_scheduler.step(model_pred, t, latents).prev_sample

            if return_all_timesteps:
                latent_history.append(latents)

        if return_all_timesteps:
            return [self._decode_latents(latent) for latent in latent_history]

        return self._decode_latents(latents)            
    
    def _prepare_latents(self, batch_size, height, width, device, latents=None):
        shape = (
            batch_size,
            self._vae.config.latent_channels,
            int(height) // 2 ** self._num_downsamples_vae,
            int(width) // 2 ** self._num_downsamples_vae,
        )
        
        # Sample the initial noise
        latents = sample_noise(shape, use_offset_noise=False, device=device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _decode_latents(self, latents) -> torch.Tensor:
        latents = 1 / self._vae.config.scaling_factor * latents
        image = self._vae.decode(latents, return_dict=False)[0]
        image = self._unnormalize(image)
        return image
    
    def _encode_image(self, img: torch.Tensor) -> torch.Tensor:
        # Normalize the image
        img = self._normalize(img)
        img_latents = self._vae.encode(img).latent_dist.sample()
        if 'scaling_factor' in self._vae.config:             
            img_latents *= self._vae.config.scaling_factor
        return img_latents
    
    def _encode_cond_img(self, cond_img: torch.Tensor) -> torch.Tensor:
        if not self._fit_emb_for_cond_img:
            return self._encode_image(cond_img)
        else:
            # Normalize the conditioning image
            x_cond = self._normalize(x_cond)
            
            return self._cond_emb(cond_img)

    def _apply_conditioning_to_latents(
            self, 
            img_latents: torch.Tensor,
            cond_img_latents: torch.Tensor
        ) -> torch.Tensor:
        if self._cond_type == 'sum':
            return img_latents + cond_img_latents
        elif self._cond_type == 'concat':
            return torch.cat([img_latents, cond_img_latents], dim = 1)
        else:
            raise ValueError('Unknown cond_type')
        
    def _rescale_noise_cfg(noise_cfg, noise_pred_cond, guidance_rescale=0.7):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4

        Recommended value for `guidance_rescale` is 0.7 when using w_cfg = 7.5 (emperically determined on natural images) 
        """
        std_text = noise_pred_cond.std(dim=list(range(1, noise_pred_cond.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        
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
        assert self.objective in ['pred_noise', 'pred_v'], 'Only epsilon, v_prediction are supported for SNR weighting'
        snr = compute_snr(self._noise_scheduler, t)
        base_weight = (
            torch.stack([snr, self._snr_gamma * torch.ones_like(t)], dim=1).min(dim=1)[0] / snr
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