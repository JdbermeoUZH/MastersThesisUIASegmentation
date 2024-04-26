import random
from tqdm import tqdm
from typing import Optional

import torch
import torch.nn.functional as F
from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    default, reduce, rearrange, extract)


class ConditionalGaussianDiffusion(GaussianDiffusion):
    """
    Conditional Gaussian diffusion model
    
    Completely based on https://github.com/lucidrains/denoising-diffusion-pytorch
    """
    
    def __init__(
        self,
        also_unconditional: bool = False,
        unconditional_rate: Optional[float] = None,
        condition_by_concat: bool = True,
        only_unconditional: bool = False,
        *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        if unconditional_rate is None:
            assert unconditional_rate >= 0 and unconditional_rate <= 1, 'uncoditional_rate must be between 0 and 1'
        
        self.only_unconditional = only_unconditional
        self.also_unconditional = also_unconditional
        self.uncoditional_rate = unconditional_rate
        self.condition_by_concat = condition_by_concat
        
        if not only_unconditional:
            assert self.model.condition, 'Unet model must be defined in conditional mode' 
                
    def p_losses_conditioned_on_img(self, x_start, t, x_cond, 
                                    noise=None, offset_noise_strength=None):

        if not self.only_unconditional:
            assert x_cond.shape[-2:] == x_start.shape[-2:], 'x_cond and x_start must have the same Height and Width'
        
        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step
        model_out = self.model(x, t, x_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, cond_img, t: int = None, min_t: int = None, max_t: int = None, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        
        # Normalize the image to be between -1 and 1 and check it follows necessary constraints
        img_size = img_size[0] if isinstance(img_size, tuple) else img_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}, \
            but got {h} and {w} respectively'
            
        img = self.normalize(img)
        if img.max() > 1 or img.min() < -1:
            print('Warning: img is not normalized between -1 and 1'
                  f'max_value: {img.max()}, min_value: {img.min()}')
            
        # Process conditional image if necessary
        if not self.only_unconditional:
            assert cond_img.shape[-2:] == img.shape[-2:], 'cond_img and img must have the same shape in H and W'
            
            if self.also_unconditional:
                assert cond_img.shape[1] > 1, 'cond_img must be one hot encoded if training a single model in conditional and unconditional mode' 
                
                if not self.condition_by_concat:
                    # Add an unconditional channel of zeros to the one hot encoded cond_img
                    #  if conditioning by multiplication
                    cond_img = torch.cat([cond_img, torch.zeros(b, 1, h, w, device = cond_img.device)], dim = 1)
                    
                # Choose randomly whether it will be a conditional or unconditional forward pass
                if random.random() < self.uncoditional_rate:
                    if self.condition_by_concat:
                        cond_img = cond_img * 0
                    else:
                        # Project the image only to the unconditional channel
                        cond_img[:, :-1] = 0
                        cond_img[:, -1] = 1
                        
            cond_img = self.normalize(cond_img)  
            
            if cond_img.max() > 1 or cond_img.min() < -1:
                print('Warning: cond_img is not normalized between -1 and 1'
                    f'torch.unique(cond_img): {torch.unique(cond_img)}')
        else:
            cond_img = None
        
        # Define noise step t    
        if t is None:
            min_t = default(min_t, 0)
            max_t = default(max_t, self.num_timesteps)
            t = torch.randint(min_t, max_t, (b,), device=device).long()
        else:
            if min_t is not None and max_t is not None:
                assert min_t <= t <= max_t, f't must be between {min_t} and {max_t}, but got {t}' 
                    
        return self.p_losses_conditioned_on_img(img, t, cond_img, *args, **kwargs)      
    
    @torch.inference_mode()
    def p_sample_loop(self, img_shape, x_cond, return_all_timesteps = False):
        if not self.only_unconditional:
            x_cond = self.normalize(x_cond)
        
        img = torch.randn(img_shape, device = self.device)
        imgs = [img]

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, _ = self.p_sample(img, t, x_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, img_shape, x_cond, return_all_timesteps = False):
        if not self.only_unconditional:
            x_cond = self.normalize(x_cond)            

        batch, device, total_timesteps, sampling_timesteps, eta, objective = img_shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(img_shape, device = device)
        imgs = [img]

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, img_shape: tuple, x_cond: torch.Tensor, return_all_timesteps = False, unconditional_sampling = False):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        
        if not self.only_unconditional:
            assert img_shape[-2:] == x_cond.shape[-2:], 'img_shape and x_cond must have the same shapes'
            
            if self.also_unconditional and not self.condition_by_concat:
                # Add an unconditional channel of zeros to the one hot encoded cond_img
                #  if conditioning by multiplication and training a model with also unconditional mode
                b, _, h, w = x_cond.shape
                x_cond = torch.cat([x_cond, torch.zeros(b, 1, h, w, device=x_cond.device)], dim = 1)
            
            if unconditional_sampling:
                assert self.also_unconditional, 'unconditional_sampling is only available when also_unconditional is True'
                assert x_cond.shape[1] > 1, 'cond_img must be one hot encoded'
                
                # Select randomly whether it will be a conditional or forward pass
                if self.condition_by_concat:
                    x_cond = x_cond * 0
                else:
                    # Project the image only to the unconditional channel
                    x_cond[:, :-1] = 0
                    x_cond[:, -1] = 1

        else:
            x_cond = None
        
        return sample_fn(img_shape, x_cond, return_all_timesteps)

    def set_sampling_timesteps(self, sampling_timesteps):
        self.sampling_timesteps = sampling_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        