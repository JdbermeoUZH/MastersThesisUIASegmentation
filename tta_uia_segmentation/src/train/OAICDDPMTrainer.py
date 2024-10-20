import os
import copy
import math
import functools
from tqdm import tqdm
import blobfile as bf
from typing import Optional
from collections import defaultdict

import wandb
import numpy as np
import torch as th
from PIL import Image   
import torch.distributed as dist
from torchvision.utils import make_grid, save_image 
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    multiscale_structural_similarity_index_measure
)
from torchmetrics.functional.regression import mean_absolute_error

from improved_diffusion import dist_util, logger
from improved_diffusion.fp16_util import (
    zero_grad,
)
from improved_diffusion.train_util import TrainLoop, log_loss_dict, get_blob_logdir
from improved_diffusion.resample import LossAwareSampler
from tta_uia_segmentation.src.dataset import DatasetInMemoryForDDPM
from tta_uia_segmentation.src.dataset.utils import unnormalize_to_zero_to_one, onehot_to_class


metrics_to_log_default = {
    'PSNR': peak_signal_noise_ratio,
    'SSIM': structural_similarity_index_measure,
    'MSSIM': multiscale_structural_similarity_index_measure,
    'MAE': mean_absolute_error,
}


def cycle(dl):
    while True:
        for data in dl:
            yield data
            
            
@th.no_grad()
def tensor_collection_to_image_grid(
    tensor,
    **kwargs,
) -> None:
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", th.uint8).numpy()
    return Image.fromarray(ndarr)


class OAICDDPMTrainer(TrainLoop):
    def __init__(
        self,
        train_data: DataLoader,
        train_num_steps: int,
        batch_size: int,
        wandb_log: bool,
        val_data: Optional[Dataset] = None,
        metrics_to_log: dict[str, callable] = metrics_to_log_default,
        num_samples_for_metrics: int = 50,
        measure_performance_on_train: bool = True,
        measure_performance_on_val: bool = True,
        num_workers: int = 2,
        use_ddim: bool = False,
        clip_denoised: bool = True,
        *args, **kwargs):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_num_steps = train_num_steps
        self.wandb_log = wandb_log
        
        self.train_data = train_data
        self.val_data = val_data
        self.metrics_to_log = metrics_to_log
        self.num_samples_for_metrics = num_samples_for_metrics
        self.measure_performance_on_train = measure_performance_on_train
        self.measure_performance_on_val = measure_performance_on_val

        # Create dataloader for train data 
        data = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            drop_last=True
        )
    
        data = cycle(data)
        
        super().__init__(data=data, batch_size=batch_size, *args, **kwargs)
        
        # Sampling parameters
        self.sample_fn = self.diffusion.p_sample_loop if not use_ddim \
            else self.diffusion.ddim_sample_loop
        self.clip_denoised = clip_denoised
        
                
    def run_loop(self):
        progress_bar = tqdm(total=self.train_num_steps,
                            initial=self.resume_step,
                            desc="Training")
        while (
            (not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps)
            and 
            self.step + self.resume_step < self.train_num_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            
            if self.step % self.log_interval == 0:
                logged_info = logger.dumpkvs()
                
                if self.wandb_log and dist.get_rank() == 0:
                    del logged_info['step']
                    wandb.log(dict(logged_info),
                              step=self.step + self.resume_step)
                
            if self.step % self.save_interval == 0:
                self.save()
                
                # Sample images and log their metrics to wandb
                if self.wandb_log and dist.get_rank() == 0:
                    if self.measure_performance_on_train:
                        self.measure_performance(
                            self.train_data, 'train', self.num_samples_for_metrics)
                    
                    if self.measure_performance_on_val and self.val_data is not None:
                        self.measure_performance(
                            self.val_data, 'val', self.num_samples_for_metrics)
            
            self.step += 1
            
            progress_bar.update(1)  
        
        progress_bar.close()
        
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
    
    
    @th.inference_mode()
    def measure_performance(
        self, 
        dataset: DatasetInMemoryForDDPM,
        dataset_name: str, 
        num_samples: int = 1000,
        num_imgs_visualized: int = 16):
        
        sample_dl = DataLoader(
            dataset.sample_slices(num_samples),
            batch_size=self.microbatch, pin_memory = True,
            num_workers = self.num_workers)
        
        metrics = defaultdict(list)
        
        imgs_for_plot = []
        for img, x_cond in sample_dl:
            img = img.to(dist_util.dev())
            x_cond = x_cond.to(dist_util.dev())
            
            x_gen = self.sample_fn(
                self.model,
                img.shape,
                clip_denoised=self.clip_denoised,
                model_kwargs={"x_cond": x_cond}
            )
            
            img = unnormalize_to_zero_to_one(img).cpu()
            x_gen = unnormalize_to_zero_to_one(x_gen).cpu()
            n_classes = x_cond.shape[1]
            x_cond = onehot_to_class(unnormalize_to_zero_to_one(
                x_cond)).cpu() / (n_classes - 1)
            
            for metric_name, metric_fn in self.metrics_to_log.items():
                metric = metric_fn(x_gen, img).item()
                metrics[metric_name].append(metric)
                
            if len(imgs_for_plot) <= num_imgs_visualized:
                imgs_for_plot.append(
                    th.cat([x_cond, img, x_gen], dim = -1))
            
        all_images = th.cat(imgs_for_plot, dim = 0)
        global_step = self.step + self.resume_step
        milestone = global_step // self.save_interval
        all_images_fn = f'{dataset_name}-sample-m{milestone}-step-{global_step}-img_gt_seg_gt_gen_img.png'
        plot_rows = int(math.sqrt(len(imgs_for_plot)))
        save_image(all_images, os.path.join(get_blob_logdir(), all_images_fn), 
                   nrow = plot_rows)

        for metric_name, metric_values in metrics.items():
            wandb.log({f'{dataset_name}_{metric_name}': np.mean(metric_values)},
                      step=global_step)
        
        # Log images in wandb
        wandb.log({
                all_images_fn: wandb.Image(tensor_collection_to_image_grid(
                    all_images,nrow = plot_rows))}, 
                step = global_step
            )
    
    def forward_backward(self, batch, cond):
        
        assert batch.shape[0] == cond.shape[0] and batch.shape[-2:] == cond.shape[-2:], \
        'Batch and condition must have the batch size and spatial dimensions.'
        
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = cond[i : i + self.microbatch].to(dist_util.dev())
            last_batch = (i + self.microbatch) >= batch.shape[0]
            
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs={"x_cond": micro_cond},
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()
                
                
