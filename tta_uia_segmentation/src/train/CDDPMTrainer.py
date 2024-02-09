import math
from tqdm import tqdm
from pathlib import Path
import torch
from multiprocessing import cpu_count
from typing import Optional

import wandb
from torch.optim import Adam
from ema_pytorch import EMA
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import utils
from PIL import Image   
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    multiscale_structural_similarity_index_measure
)
from torchmetrics.functional.regression import mean_absolute_error

from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    cycle, has_int_squareroot, Trainer, divisible_by, num_to_groups, exists)
from denoising_diffusion_pytorch.version import __version__

from tta_uia_segmentation.src.models import ConditionalGaussianDiffusion
from tta_uia_segmentation.src.dataset import DatasetInMemoryForDDPM


metrics_to_log_default = {
    'PSNR': peak_signal_noise_ratio,
    'SSIM': structural_similarity_index_measure,
    'MSSIM': multiscale_structural_similarity_index_measure,
    'MAE': mean_absolute_error,
}


@torch.no_grad()
def tensor_collection_to_image_grid(
    tensor,
    **kwargs,
) -> None:
    grid = utils.make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(ndarr)


class CDDPMTrainer(Trainer):

    """
           
    """
    def __init__(
        self,
        diffusion_model: ConditionalGaussianDiffusion,
        train_dataset: DatasetInMemoryForDDPM,
        val_dataset: Optional[DatasetInMemoryForDDPM] = None,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        num_workers = cpu_count(),
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False,
        wandb_log: bool = True,
        wandb_dir: str = 'wandb',
        metrics_to_log: dict[str, callable] = metrics_to_log_default, 
    ):
        super(object, self).__init__()
        
        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling


        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm
        
        self.num_workers = num_workers

        # dataset and dataloader

        self.train_ds = train_dataset
        self.val_ds = val_dataset
        assert len(self.train_ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        train_dl = DataLoader(self.train_ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, 
                              num_workers = self.num_workers)
        val_dl = DataLoader(self.val_ds, batch_size = train_batch_size, shuffle = True, pin_memory = True,
                            num_workers = self.num_workers)   
        
        train_dl = self.accelerator.prepare(train_dl)
        self.train_dl = cycle(train_dl)

        val_dl = self.accelerator.prepare(val_dl)
        self.val_dl = cycle(val_dl)
        
        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # Wandb logging
        self.wandb_log = wandb_log
        self.wandb_dir = wandb_dir
        self.metrics_to_log = metrics_to_log
                
        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only
        
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    img, cond_img = next(self.train_dl)
                    img, cond_img = img.to(device), cond_img.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(img, cond_img)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')
                
                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    if self.wandb_log:
                        wandb.log({'total_loss': total_loss}, step = self.step)
                            
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()
                        milestone = self.step // self.save_and_sample_every
                    
                        # Evaluate the model on a sample of the training set
                        train_sample_dl = DataLoader(
                            self.train_ds.sample_slices(self.num_samples), 
                            batch_size=self.batch_size, pin_memory = True, num_workers = self.num_workers)
                        
                        self.evaluate(train_sample_dl, device=device, prefix='train')
                        
                        # Evaluate the model on a sample of the validation set
                        val_sample_dl = DataLoader(
                            self.val_ds.sample_slices(self.num_samples), 
                            batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers)
                        self.evaluate(val_sample_dl, device=device, prefix='val')
                        
                        # whether to calculate fid
                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
        
    @torch.inference_mode()
    def evaluate(self, sample_dl: DataLoader, device, prefix: str = '', ):
        samples_imgs = []                 
        milestone = self.step // self.save_and_sample_every
           
        for img_gt, seg_gt in sample_dl:
            img_gt, seg_gt = img_gt.to(device), seg_gt.to(device)
            generated_img = self.ema.ema_model.sample(x_cond=seg_gt)
            
            # Store the generated image and the segmentation map side by side
            # seg_gt = self.model.normalize(seg_gt)   # So that they are on the same range of intensities
            samples_imgs.append(torch.cat([generated_img, seg_gt], dim = -1)) 
            
            # log metrics
            if self.wandb_log:
                for metric_name, metric_func in self.metrics_to_log.items():
                    metric_value = metric_func(generated_img, img_gt).item()
                    wandb.log({f'{prefix}_{metric_name}': metric_value}, step = self.step)
        
        all_images = torch.cat(samples_imgs, dim = 0)

        all_images_fn = f'{prefix}_sample-m{milestone}-step-{self.step}.png'
        utils.save_image(all_images, str(self.results_folder / all_images_fn), 
                        nrow = int(math.sqrt(self.num_samples)))
        
        # Log images in wandb
        wandb.log(
            {all_images_fn: wandb.Image(tensor_collection_to_image_grid(
                all_images, nrow = int(math.sqrt(self.num_samples))))}, 
            step = self.step
            ) 
        
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        cpt_fp = str(self.results_folder / f'model-{milestone}-step_{self.step}.pt')
        torch.save(data, cpt_fp)
