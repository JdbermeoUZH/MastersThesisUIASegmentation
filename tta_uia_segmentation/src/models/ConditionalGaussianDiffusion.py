from tqdm import tqdm

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
        *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.model.self_condition, 'Unet model must be defined in self condition mode' 
                
    def p_losses_conditioned_on_img(self, x_start, t, x_cond, 
                                    noise=None, offset_noise_strength=None):

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

    def forward(self, img, cond_img, min_t=None, max_t=None, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        img_size = img_size[0] if isinstance(img_size, tuple) else img_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}, \
            but got {h} and {w} respectively'
            
        assert cond_img.shape[-2:] == img.shape[-2:], 'cond_img and img must have the same shape in H and W'
        
        min_t = default(min_t, 0)
        max_t = default(max_t, self.num_timesteps)
        t = torch.randint(min_t, max_t, (b,), device=device).long()
        img = self.normalize(img)
        cond_img = self.normalize(cond_img)  
        
        if img.max() > 1 or img.min() < -1:
            print('Warning: img is not normalized between -1 and 1'
                  f'max_value: {img.max()}, min_value: {img.min()}')
        
        if cond_img.max() > 1 or cond_img.min() < -1:
            print('Warning: cond_img is not normalized between -1 and 1'
                  f'torch.unique(cond_img): {torch.unique(cond_img)}')
        
        return self.p_losses_conditioned_on_img(img, t, cond_img, *args, **kwargs)      
    
    @torch.inference_mode()
    def p_sample_loop(self, img_shape, x_cond, return_all_timesteps = False):
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
        x_cond = self.normalize(x_cond)

        batch, device, total_timesteps, sampling_timesteps, eta, objective = x_cond.shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

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
    def sample(self, img_shape: tuple, x_cond: torch.Tensor, return_all_timesteps = False):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(img_shape, x_cond, return_all_timesteps)

    def set_sampling_timesteps(self, sampling_timesteps):
        self.sampling_timesteps = sampling_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        