import math
import logging
from tqdm import tqdm
from pathlib import Path
from packaging import version
from multiprocessing import cpu_count
from typing import Optional, Literal

import wandb
import torch
import transformers
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from torch.optim import Adam, AdamW
from ema_pytorch import EMA
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
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
    divisible_by, exists, Trainer)

from tta_uia_segmentation.src.models import ConditionalGaussianDiffusionInterface
from tta_uia_segmentation.src.dataset import DatasetInMemoryForDDPM
from tta_uia_segmentation.src.dataset.utils import onehot_to_class



metrics_to_log_default = {
    'PSNR': peak_signal_noise_ratio,
    'SSIM': structural_similarity_index_measure,
    'MSSIM': lambda y_gt, y_pred: multiscale_structural_similarity_index_measure(
        y_pred, y_gt, kernel_size=7),
    'MAE': mean_absolute_error,
}

logger = get_logger(__name__)

@torch.no_grad()
def tensor_collection_to_image_grid(
    tensor,
    **kwargs,
) -> None:
    grid = utils.make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(ndarr)

def check_btw_0_1(*args: torch.Tensor, margin_error = 1e-2):
    for tensor in args:
        assert tensor.min() >= 0 - margin_error and tensor.max() <= 1 + margin_error, 'tensor values should be between 0 and 1'    

def check_btw_minus_1_plus_1(*args: torch.Tensor):
    for tensor in args:
        assert tensor.min() >= -1 and tensor.max() <= 1, 'tensor values should be between -1 and 1'


class CDDPMTrainer:

    """
    Trainer for Conditional Diffusion models that follow the ConditionalGaussianDiffusionInterface.
    """
    def __init__(
        self,
        ddpm: ConditionalGaussianDiffusionInterface,
        train_dataset: DatasetInMemoryForDDPM,
        val_dataset: Optional[DatasetInMemoryForDDPM] = None,
        optimizer_type: Literal['adam', 'adamW', 'AdamW8bit'] = 'adam',
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        num_workers = cpu_count(),
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        log_val_loss_every = 250,
        save_and_sample_every = 1000,
        num_validation_samples = 100,
        num_viz_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False,
        use_ddim_sampling = True,
        enable_xformers: bool = False,
        gradient_checkpointing: bool = False,
        allow_tf32: bool = False,
        scale_lr: bool = False,
        wandb_log: bool = True,
        wandb_dir: str = 'wandb',
        metrics_to_log: dict[str, callable] = metrics_to_log_default, 
        accelerator: Accelerator = None,
    ):
       
        # Training hyperparameters
        self.max_grad_norm = max_grad_norm
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, \
            'effective batch size (train_batch_size x gradient_accumulate_every)' + \
            'should be at least 16 or above'

        # dataset and dataloader
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        assert len(self.train_ds) >= 100, 'you should have at least 100 images in your folder.' + \
            'at least 10k images recommended'

        self.num_workers = num_workers

        train_dl = DataLoader(
            self.train_ds, batch_size=train_batch_size,
            shuffle=True, pin_memory=True,
            num_workers = self.num_workers)
        val_dl = DataLoader(
            self.val_ds, batch_size=train_batch_size,
            shuffle=True, pin_memory=True,
            num_workers = self.num_workers)   
        
        self.train_num_steps = train_num_steps
        self.num_update_steps_per_epoch = math.ceil(len(train_dl) / gradient_accumulate_every)
        self.num_train_epochs = math.ceil(train_num_steps / self.num_update_steps_per_epoch)

        # model
        self.model = ddpm
        self.channels = ddpm.num_img_channels
        self.use_ddim_sampling = use_ddim_sampling
        self.image_size = ddpm.image_size

        # setup xformers
        if enable_xformers:
            self._enable_xformers()

        # setup gradient checkpointing
        if gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # optimizer
        if scale_lr:
            train_lr = train_lr * gradient_accumulate_every * train_batch_size * accelerator.num_processes

        if optimizer_type == 'adam':
            opt_class = Adam
        elif optimizer_type == 'adamW':
            self.opt = AdamW
        elif optimizer_type == 'AdamW8bit':
            from bitsandbytes.optim import AdamW8bit
            opt_class = AdamW8bit
        else:
            raise ValueError('optimizer_type should be either "adam", "adamW", "AdamW8bit"')
        
        self.opt = opt_class(self.model.parameters(), lr = train_lr, betas = adam_betas) 

        # accelerator
        self.accelerator = accelerator or Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no',
            gradient_accumulation_steps=gradient_accumulate_every,
            log_with='wandb' if wandb_log else None,
            project_config=ProjectConfiguration(wandb_dir=wandb_dir, project_dir=results_folder)
        )

        # Setup EMA model on the main process
        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt, self.train_dl, self.val_dl = self.accelerator.prepare(
            self.model,
            self.opt,
            train_dl,
            val_dl
        )

        # Move non trainable parameters to half-precision to save memory
        if self.accelerator.mixed_precision in ['fp16', 'bf16']:
            self.model.move_non_trainable_params_to(
                device=self.accelerator.device,
                dtype=self.accelerator.mixed_precision
            )

        # Wandb logging
        self.wandb_log = wandb_log
        self.wandb_dir = wandb_dir
        self.metrics_to_log = metrics_to_log

        # for logging results in a folder periodically
        self.log_val_loss_every = log_val_loss_every
        self.save_and_sample_every = save_and_sample_every
        self.num_validation_samples = num_validation_samples
        self.num_viz_samples = num_viz_samples

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self._setup_logging()

        # step counter state
        self.step = 0

        # FID-score computation
        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not self.use_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )
            # TODO: 
            # 1. Create an appropriate dataloader for the FID computation
            # 2. Modify the FIDEvaluation class so that it works with a conditional model
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

        self.model.train_mode()

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    img, cond_img, _ = next(self.train_dl)
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
                    
                    if self.step != 0 and divisible_by(self.step, self.log_val_loss_every):
                        self._log_val_loss(device=device)

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()
                        milestone = self.step // self.save_and_sample_every
                    
                        # Evaluate the model on a sample of the training set
                        train_sample_dl = DataLoader(
                            self.train_ds.sample_slices(self.num_validation_samples), 
                            batch_size=self.batch_size, pin_memory = True, num_workers = self.num_workers)
                        
                        self.evaluate(train_sample_dl, device=device, prefix='train')
                        
                        # Evaluate the model on a sample of the validation set
                        val_sample_dl = DataLoader(
                            self.val_ds.sample_slices(self.num_validation_samples), 
                            batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers)
                        self.evaluate(val_sample_dl, device=device, prefix='val', unconditional_sampling=False)
                        
                        # Evaluate the model on the sample set in unconditional mode if the model is also trained unconditionally
                        if self.model.also_unconditional:
                            self.evaluate(train_sample_dl, device=device, prefix='train_unconditional',
                                          unconditional_sampling=True)
                            self.evaluate(val_sample_dl, device=device, prefix='val_uncondtional',
                                          unconditional_sampling=True)
                            
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
                        
                        self.ema.ema_model.train()

                pbar.update(1)

        accelerator.print('training complete')
        
    @torch.inference_mode()
    def evaluate(self, sample_dl: DataLoader, device, prefix: str = '', unconditional_sampling = False):
        samples_imgs = []                 
        milestone = self.step // self.save_and_sample_every
        
        metric_cum = {metric_name: 0. for metric_name in self.metrics_to_log.keys()}
        
        for img_gt, seg_gt, _ in sample_dl:
            img_gt, seg_gt = img_gt.to(device), seg_gt.to(device).type(torch.int8)
            ema_model: ConditionalGaussianDiffusionInterface = self.ema.ema_model
            generated_img = ema_model.sample(
                img_shape=img_gt.shape, 
                x_cond=seg_gt,
                unconditional_sampling=unconditional_sampling
            )
            
            # Convert seg_gt to single channel to plot it
            assert seg_gt.max() == 1, 'seg_gt should be one-hot encoded'
            assert seg_gt.min() == 0, 'seg_gt should be one-hot encoded'
            n_classes = seg_gt.shape[1]
            seg_gt = onehot_to_class(seg_gt.type(torch.int8))
            seg_gt = seg_gt / (n_classes - 1)
            assert seg_gt.shape[1] == 1, 'seg_gt should be single channel'
            
            # Store the generated image and the segmentation map side by side
            if len(samples_imgs) * sample_dl.batch_size <= self.num_viz_samples:
                samples_imgs.append(torch.cat([seg_gt, img_gt, generated_img], dim = -1)) 
            
            check_btw_0_1(img_gt, seg_gt, generated_img)
            
            # log metrics
            if self.wandb_log:
                for metric_name, metric_func in self.metrics_to_log.items():
                    metric_cum[metric_name] += metric_func(generated_img, img_gt).item()
                    
        all_images = torch.cat(samples_imgs, dim = 0)[0: self.num_viz_samples]

        all_images_fn = f'{prefix}-sample-m{milestone}-step-{self.step}-img_gt_seg_gt_gen_img.png'
        utils.save_image(all_images, str(self.results_folder / all_images_fn), 
                         nrow = min(5, int(math.sqrt(self.num_viz_samples))))
        
        # Log metrics and images in wandb
        if self.wandb_log:
            for metric_name, metric_val in metric_cum.items():
                metric_val /= len(sample_dl)
                wandb.log({f'{prefix}_{metric_name}': metric_val}, step = self.step)

            wandb.log(
                {all_images_fn: wandb.Image(
                    tensor_collection_to_image_grid(
                        all_images, 
                        nrow = int(math.sqrt(self.num_viz_samples))
                        )
                    )}, 
                step = self.step
                ) 
    
    @torch.no_grad()
    def _log_val_loss(self, device, max_batches = 10):
        sample_size = self.gradient_accumulate_every * self.batch_size * min(max_batches, self.num_validation_samples)
        val_sample_dl = DataLoader(
            self.val_ds.sample_slices(sample_size), 
            batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers)
        
        total_loss_val = 0.
        for img, cond_img, _ in tqdm(val_sample_dl, desc = 'total_loss_val'):
            img, cond_img = img.to(device), cond_img.to(device)
            loss = self.model(img, cond_img)
            total_loss_val += (1 / len(val_sample_dl)) * loss.item()

        if self.wandb_log:
            wandb.log({'total_loss_val': total_loss_val}, step = self.step)
    
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        cpt_fp = str(self.results_folder / f'model-{milestone}.pt')
        torch.save(data, cpt_fp)

    @property
    def device(self):
        return self.accelerator.device
    
    def _setup_logging(self):
         # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

    def _enable_xformers(self):
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            self.model.enable_xformers()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
