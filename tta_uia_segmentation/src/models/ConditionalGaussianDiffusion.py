import random
from tqdm import tqdm
from typing import Optional, Literal
from functools import partial

import torch
import torch.nn.functional as F
from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    default, reduce, rearrange, extract, identity, ModelPrediction)


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
        self.unconditional_rate = unconditional_rate
        self.condition_by_concat = condition_by_concat
        
        if not only_unconditional:
            assert self.model.condition, 'Unet model must be defined in conditional mode' 
                
    def model_predictions(self, x, t, x_cond, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
    
    def p_losses_conditioned_on_img(self, x_start, t, x_cond, w_clf_free: float = 0,
                                    pixel_weights: Optional[torch.Tensor] = None, 
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
        
        # calculate unconditional score
        if w_clf_free > 0:
            assert self.also_unconditional, 'w_clf_free is only available when also_unconditional is True'
            assert not self.only_unconditional, 'w_clf_free only makes sense when the model was trained in conditional and unconditional mode'
            x_unconditional = self._generate_unconditional_x_cond(batch_size=x_cond.shape[0], device=x_cond.device)
            model_out_unconditional = self.model(x, t, x_unconditional)
            
            #model_out = (1 + w_clf_free) * model_out - w_clf_free * model_out_unconditional    
            model_out = model_out - w_clf_free / (1 + w_clf_free) * model_out_unconditional

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
        loss = pixel_weights * loss if pixel_weights is not None else loss
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()
    
    
    def distillation_loss(
        self,
        img,
        cond_img,
        t: Optional[torch.Tensor] = None,
        type: Literal['sds', 'dds', 'pds'] = 'sds',
        min_t: Optional[int] = None, max_t: Optional[int] = None, 
        *args, **kwargs
        ):
                
        if not self.only_unconditional:
            assert cond_img.shape[-2:] == img.shape[-2:], 'cond_img and img must have the same Height and Width'
            
        img = self.normalize(img)
        if img.max() > 1 or img.min() < -1:
            print('Warning: img is not normalized between -1 and 1'
                  f'max_value: {img.max()}, min_value: {img.min()}')
        
        # Normalize the conditional image to be between -1 and 1 
        cond_img = self.normalize(cond_img)  
        
        if cond_img.max() > 1 or cond_img.min() < -1:
            print('Warning: cond_img is not normalized between -1 and 1'
                f'torch.unique(cond_img): {torch.unique(cond_img)}')
        
        # Define noise step t if not given    
        if t is None:
            min_t = default(min_t, 0)
            max_t = default(max_t, self.num_timesteps)
            b = img.shape[0]
            t = torch.randint(min_t, max_t, (b,), device=img.device).long()
        else:
            if min_t is not None and max_t is not None:
                assert (min_t <= t).all() and (t <= max_t).all(), f't must be between {min_t} and {max_t}, but got {t}'
                  
        if type == 'sds':
            return self.score_distillation_sampling(img, cond_img, t, *args, **kwargs)
        elif type == 'dds':
            return self.delta_denoising_score(img, t, cond_img, *args, **kwargs)
        elif type == 'pds':
            return self.posterior_distillation_sampling(img, t, cond_img, *args, **kwargs)
        else:
            raise ValueError(f'Unknown distillation loss type: {type}')
    
    
    def score_distillation_sampling(self, x_start, x_cond, t, w_clf_free: float = 0, 
                                    pixel_weights: Optional[torch.Tensor] = None, noise = None):
        """
        Based on implementation from: https://github.com/google/prompt-to-prompt/blob/main/DDS_zeroshot.ipynb
         and https://github.com/ashawkey/stable-dreamfusion/blob/5550b91862a3af7842bb04875b7f1211e5095a63/assets/advanced.md    
        
        """
        
        with torch.inference_mode():
            noise = default(noise, lambda: torch.randn_like(x_start))

            # latent t
            x_t = self.q_sample(x_start = x_start, t = t, noise = noise)

            # Get model predictions
            model_preds = self.model_predictions(x_t, t, x_cond) 
            pred_noise = model_preds.pred_noise
        
            # calculate unconditional score
            if w_clf_free > 0:
                assert self.also_unconditional, 'w_clf_free is only available when also_unconditional is True'
                assert not self.only_unconditional, 'w_clf_free only makes sense when the model was trained in conditional and unconditional mode'

                x_unconditional = self._generate_unconditional_x_cond(batch_size=x_cond.shape[0], device=x_cond.device)
                model_preds_unconditional = self.model_predictions(x_t, t, x_unconditional)
                pred_noise = (1 + w_clf_free) * pred_noise - w_clf_free * model_preds_unconditional.pred_noise
            
            # Calculate SDS term
            score = (pred_noise - noise)
            score = torch.nan_to_num(score, nan = 0.0, posinf = 0.0, neginf = 0.0)
            score = score * pixel_weights if pixel_weights is not None else score
        
        # Form an expression that will have as gradients: score * sqrt_alphas_cumprod_t for x and score for x_cond
        x_start = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start 
        
        if self.condition_by_concat:
            x = torch.cat([x_start, x_cond], dim = 1)
        else:
            # If not, it means we are conditioning by multiplication
            x = x_start * x_cond
        
        loss = score.clone() * x
        loss = reduce(loss, 'b ... -> b', 'mean')
        
        return loss.mean()
            
    def delta_denoising_score(self, x_start, t, x_cond, sampled_noise_ref: tuple[torch.Tensor], w_clf_free: float = 0):
        raise NotImplementedError('delta_denoising_score is not implemented yet')
    
    def posterior_distillation_sampling(self, x_start, t, x_cond, sampled_x_ref: tuple[torch.Tensor], w_clf_free: float = 0):
        raise NotImplementedError('score_distillation_sampling is not implemented yet')
       
    def forward(self, img, cond_img, t: Optional[torch.Tensor] = None,
                min_t: Optional[int] = None, max_t: Optional[int] = None,
                pixel_weights: Optional[torch.Tensor] = None, 
                *args, **kwargs):
        
        b, _, h, w, device, img_size, = *img.shape, img.device, self.image_size
        
        if self.only_unconditional:
            assert cond_img is None, 'cond_img must be None when only_unconditional is True'
        
        # Normalize the image to be between -1 and 1 and check it follows necessary constraints
        img_size = img_size[0] if isinstance(img_size, tuple) else img_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}, \
            but got {h} and {w} respectively'
        
        img = self.normalize(img)
        if img.max() > 1 or img.min() < -1:
            print('Warning: img is not normalized between -1 and 1'
                  f'max_value: {img.max()}, min_value: {img.min()}')
            
        # Process conditional image if necessary
        if self.also_unconditional and self.unconditional_rate > 0: 
            assert cond_img.shape[-2:] == img.shape[-2:], 'cond_img and img must have the same shape in H and W'
            
            if self.also_unconditional:    
                assert cond_img.shape[1] > 1, 'cond_img must be one hot encoded if training a single model in conditional and unconditional mode' 
                # Choose randomly whether it will be a conditional or unconditional forward pass
                if random.random() < self.unconditional_rate:
                    cond_img = self._generate_unconditional_x_cond(batch_size=cond_img.shape[0], device=cond_img.device)
                    pixel_weights = None
        
        # Normalize the conditional image to be between -1 and 1 
        cond_img = self.normalize(cond_img)  
        
        if cond_img.max() > 1 or cond_img.min() < -1:
            print('Warning: cond_img is not normalized between -1 and 1'
                f'torch.unique(cond_img): {torch.unique(cond_img)}')
        
        # Define noise step t if not given    
        if t is None:
            min_t = default(min_t, 0)
            max_t = default(max_t, self.num_timesteps)
            t = torch.randint(min_t, max_t, (b,), device=device).long()
        else:
            if min_t is not None and max_t is not None:
                assert (min_t <= t).all() and (t <= max_t).all(), f't must be between {min_t} and {max_t}, but got {t}' 
        
        return self.p_losses_conditioned_on_img(img, t, cond_img, pixel_weights=pixel_weights,
                                                *args, **kwargs)      
    
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
            
            if unconditional_sampling:
                assert self.also_unconditional, 'unconditional_sampling is only available when also_unconditional is True'
                assert x_cond.shape[1] > 1, 'cond_img must be one hot encoded'
                
                x_cond = self._generate_unconditional_x_cond(batch_size=x_cond.shape[0], device=x_cond.device) 

        else:
            x_cond = None
        
        return sample_fn(img_shape, x_cond, return_all_timesteps)
        
    def _generate_unconditional_x_cond(self, batch_size: int, device: str):
        if self.also_unconditional:
            num_x_cond_channels = self.model.input_channels - self.model.channels
            img_shape = (batch_size, num_x_cond_channels, self.image_size, self.image_size)
            img_shape = tuple([int(i) for i  in img_shape])

            x_cond = torch.zeros(img_shape, device=device)
            
            if not self.condition_by_concat:
                x_cond[:, -1] = 1
            
        elif self.only_unconditional:
            x_cond = None
        
        else:
            raise ValueError('DDPM must be in unconditional. Either in `also_unconditional` or `only_unconditional` mode')
        
        return x_cond


    def set_sampling_timesteps(self, sampling_timesteps):
        self.sampling_timesteps = sampling_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        