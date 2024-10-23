from typing import Literal, Optional

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation
from pytorch_fid.fid_score import calculate_frechet_distance


from tta_uia_segmentation.src.utils.utils import stratified_sampling, torch_to_numpy


def normalize(img: torch.Tensor, clamp: bool = True):
    return (img * 2 - 1).clamp(-1, 1) if clamp else (img * 2 - 1)


def unnormalize(img: torch.Tensor, clamp: bool = True):
    return ((img + 1) / 2).clamp(0, 1) if clamp else (img + 1) / 2


def sample_t(min_t: int, max_t: int, batch_size: int, device: str) -> torch.Tensor:
    return torch.randint(min_t, max_t, (batch_size,), device=device, dtype=torch.long)


def sample_noise(
        img_batch_shape: tuple[int, ...] | torch.Size,
        device: str,
        use_offset_noise: bool = False
    ) -> torch.Tensor:
    """
    Generate noise for the given image batch shape

    Parameters
    ----------
    img_batch_shape : tuple[int, ...]
        Shape of the image batch
    use_offset_noise : bool
        Whether to add offset noise as suggested in: https://www.crosslabs.org//blog/diffusion-with-offset-noise
         If setting rescale_betas_zero_snr=True in Scheduler, then this is not needed
    """
    img_batch_shape = tuple(img_batch_shape)
    noise = torch.randn(img_batch_shape, device=device)
    if use_offset_noise:
        noise += 0.1 * torch.randn(
            img_batch_shape.shape[0], img_batch_shape.shape[1], 1, 1,
            device=device)
    return noise


def generate_unconditional_mask(
        img_shape: tuple[int, ...],
        device: str | torch.device,
        dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    return torch.zeros(img_shape, device=device, dtype=dtype)


def sample_t_noise_pairs(
        num_samples: int,
        noise_shape: tuple[int, int, int],
        min_t: int,
        max_t: int,
        device: str,
        repeat_pairs: int = 1,
        t_sampling_strategy: Literal['uniform', 'stratified'] = 'uniform',
        num_groups_stratified_sampling: Optional[int] = None, #32,
        use_offset_noise: bool = False,
        return_dataloader: bool = False,
        **dataloader_kwargs 
        ) -> tuple[torch.Tensor, ...] | torch.utils.data.DataLoader:
        
        # Sample t values
        if t_sampling_strategy == 'uniform':
            t_values = sample_t(min_t, max_t, (num_samples, ))
        
        elif t_sampling_strategy == 'stratified':
            t_values = stratified_sampling(min_t, max_t, 
                                           num_groups_stratified_sampling, num_samples)
        else:
            raise ValueError('Invalid t_sampling_strategy')
        
        assert len(t_values) == num_samples, 'Number of samples must match the number of t values'
        assert t_values.shape == (num_samples,), 't values must be a 1D tensor'

        # Sample noise
        noise = sample_noise(img_batch_shape=(num_samples, *noise_shape),
                             use_offset_noise=use_offset_noise, device=device)
        
        # Repeat pairs, if necessary
        if repeat_pairs > 1:
            t_values = t_values.repeat_interleave(repeat_pairs)
            noise = noise.repeat_interleave(repeat_pairs, dim=0)
        
        assert len(t_values) == len(noise), 'Number of samples must match the number of noise samples'
        
        if return_dataloader:
            return DataLoader(
                TensorDataset(t_values, noise),
                batch_size=dataloader_kwargs.get('batch_size', 1),
                shuffle=dataloader_kwargs.get('shuffle', False),
                num_workers=dataloader_kwargs.get('num_workers', 0),
            )
        
        return t_values, noise

class FIDEvaluation(FIDEvaluation):
    def fid_score_from_samples(self, samples):
        # Load statistics from the real images
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()

        # create a dataloader with the samples
        dl = DataLoader(
            TensorDataset(samples),
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        # Calculate the inception features of the samples
        stacked_fake_features = []
        for (sample,) in dl:
            fake_features = self.calculate_inception_features(sample.to(self.device))
            stacked_fake_features.append(fake_features)
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0)
        m1 = np.mean(torch_to_numpy(stacked_fake_features), axis=0)
        s1 = np.cov(torch_to_numpy(stacked_fake_features), rowvar=False)

        return calculate_frechet_distance(m1, s1, self.m2, self.s2)
    