import os
import json
from tqdm import tqdm
from typing import Union, Optional, Literal

import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR

from tta_uia_segmentation.src.models import BaseSeg
from tta_uia_segmentation.src.utils.loss import DiceLoss, dice_score
from tta_uia_segmentation.src.utils.utils import nan_sum, default


class SegTrainer:

    def __init__(
        self,
        seg: BaseSeg,
        learning_rate: float,
        device: torch.device,
        loss_func: torch.nn.Module = DiceLoss(),
        optimizer_type: Literal["adam", "adamW"] = "adam",
        weight_decay: float = 1e-5,
        grad_acc_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        is_resumed: bool = False,
        checkpoint_last: str = "checkpoint_last.pth",
        checkpoint_best: str = "checkpoint_best.pth",
        logdir: str = "logs",
        wandb_log: bool = True,
        wandb_dir: Optional[str] = "wandb",
    ):

        self._device = device
        self._seg = seg

        if optimizer_type == "adam":
            opt_type = torch.optim.Adam

        elif optimizer_type == "adamW":
            opt_type = torch.optim.AdamW

        else:
            raise ValueError(f"Unknown optimizer type {optimizer_type}.")

        self._optimizer = opt_type(
            self._seg.trainable_params, lr=learning_rate, weight_decay=weight_decay
        )

        if is_resumed:
            self._load_checkpoint(os.path.join(logdir, checkpoint_last))
        else:
            print("Starting training from scratch.")
            self._best_validation_loss = np.inf
            self._epoch = 0
            self._best_epoch = -1

        self._loss_func = loss_func

        self._grad_acc_steps = grad_acc_steps
        self._max_grad_norm = max_grad_norm

        self._training_losses = []
        self._validation_losses = []
        self._validation_scores = []

        self._logdir = logdir
        self._checkpoint_last = checkpoint_last
        self._checkpoint_best = checkpoint_best

        self._wandb_log = wandb_log
        self._wandb_dir = wandb_dir

    def train(
        self,
        train_dataloader: DataLoader,
        epochs: int,
        validate_every: int,
        warmup_steps: Optional[int] = None,
        checkpoint_best: Optional[str] = None,
        checkpoint_last: Optional[str] = None,
        logdir: Optional[str] = None,
        val_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        wandb_log: Optional[bool] = None,
        wandb_dir: Optional[str] = None,
    ):

        # Define warmup scheduler
        if warmup_steps is not None:
            if 0.0 < warmup_steps < 1.0:
                total_steps = len(train_dataloader) * epochs
                warmup_steps = int(warmup_steps * total_steps)
            warmup_scheduler = LinearLR(
                self._optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        else:
            warmup_scheduler = None

        wandb_log = wandb_log or self._wandb_log
        device = device or self._device
        logdir = logdir or self._logdir
        checkpoint_best = checkpoint_best or self._checkpoint_best
        checkpoint_last = checkpoint_last or self._checkpoint_last
        wandb_dir = wandb_dir or self._wandb_dir

        print("Starting training on epoch", self._epoch)

        pbar = tqdm(range(epochs), desc="Epochs", initial=self._epoch)

        for epoch in pbar:

            training_loss = 0
            n_samples_train = 0

            self._seg.train()

            for step, (x, y, extra_inputs) in enumerate(train_dataloader):
                if isinstance(x, list):
                    x = [x_i.to(device).float() for x_i in x]
                    batch_size = x[0].shape[0]
                else:
                    x = x.to(device).float()
                    batch_size = x.shape[0]
                y = y.to(device).float()

                extra_inputs = self._seg.select_necessary_extra_inputs(extra_inputs)

                y_pred, *_ = self._seg(x, **extra_inputs)

                loss = self._loss_func(y_pred, y) / self._grad_acc_steps

                loss.backward()

                # Update parameters every grad_acc_steps
                if (step + 1) % self._grad_acc_steps == 0 or (step + 1) == len(
                    train_dataloader
                ):
                    # Gradient clipping
                    if self._max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self._seg.trainable_params, self._max_grad_norm
                        )

                    # Update parameters
                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    # Update learning rate
                    if warmup_scheduler is not None:
                        warmup_scheduler.step()

                with torch.no_grad():
                    training_loss += loss.detach() * batch_size * self._grad_acc_steps
                    n_samples_train += batch_size

            training_loss /= n_samples_train
            self._training_losses.append(training_loss.item())

            # Add training loss to progress bar
            pbar.set_postfix({"train_loss": f"{training_loss.item():.3f}"})

            if (epoch + 1) % validate_every != 0 and epoch != epochs - 1:
                continue

            if val_dataloader is not None:
                # Evaluation
                validation_loss, validation_score_fg, validation_scores = self.evaluate(
                    val_dataloader, device=device
                )
                self._validation_losses.append(validation_loss.item())
                self._validation_scores.append(validation_score_fg.item())

                # Checkpoint last state
                self._save_checkpoint(
                    os.path.join(logdir, checkpoint_last), epoch=epoch
                )

                # Add validation loss and score to progress bar
                pbar.set_postfix(
                    {
                        "train_loss": f"{training_loss.item():.3f}",
                        "validation_loss": f"{validation_loss.item():.3f}",
                        "validation_score": f"{validation_score_fg.item():.3f}",
                    }
                )

                if validation_loss < self._best_validation_loss:
                    print(
                        f"New best validation loss: {validation_loss.item()}"
                        + f" at epoch {epoch}."
                    )

                    self._best_epoch = epoch
                    self._best_validation_loss = validation_loss

                    # Checkpoint best state
                    self._save_checkpoint(
                        os.path.join(logdir, checkpoint_best), epoch=epoch
                    )

            if wandb_log:
                if val_dataloader is not None:
                    validation_metrics_to_log = {
                        "validation_loss": self._validation_losses[-1],
                        "validation_score": self._validation_scores[-1],
                    }

                    if validation_scores.shape[0] > 1:
                        for i, score_per_class in enumerate(validation_scores):
                            label_name = val_dataloader.dataset.get_label_name(i)
                            validation_metrics_to_log[
                                f"validation_score_{label_name.zfill(2)}"
                            ] = score_per_class.item()
                else:
                    validation_metrics_to_log = {}

                wandb.log(
                    {"training_loss": training_loss, **validation_metrics_to_log},
                    step=epoch,
                )
                wandb.save(
                    os.path.join(wandb_dir, checkpoint_last), base_path=wandb_dir
                )
                wandb.save(
                    os.path.join(wandb_dir, checkpoint_best), base_path=wandb_dir
                )

        print(f"Training finished. Best epoch: {self.best_epoch}")

    @torch.inference_mode()
    def evaluate(
        self,
        val_dataloader: DataLoader,
        device: Optional[torch.device] = None,
    ):

        device = device or self._device
        validation_loss = 0
        validation_scores = 0
        n_samples_val = 0

        self._seg.eval()

        for x, y, extra_inputs in val_dataloader:
            if isinstance(x, list):
                x = [x_i.to(device).float() for x_i in x]
                batch_size = x[0].shape[0]
            else:
                x = x.to(device).float()
                batch_size = x.shape[0]
            y = y.to(device).float()

            extra_inputs = self._seg.select_necessary_extra_inputs(extra_inputs)

            y_pred, *_ = self._seg(x, **extra_inputs)

            loss = self._loss_func(y_pred, y)

            # Get mean dice score per class
            dices = dice_score(
                y_pred,
                y,
                soft=False,
                reduction="none",
                foreground_only=False,
                bg_channel=0,
            )
            dices = dices.nanmean(0)  # Mean over samples, dice per class

            if isinstance(validation_scores, int) and validation_scores == 0:
                # Must initialize to nan, otherwise nan_sum will never return nan
                #  Necessary to tell appart nan dice scores from 0 dice scores
                validation_scores = torch.zeros_like(dices) * torch.nan
            validation_scores = nan_sum(validation_scores, dices * batch_size)
            validation_loss += loss * batch_size
            n_samples_val += batch_size

        validation_loss /= n_samples_val
        validation_scores /= n_samples_val

        validation_score_fg = validation_scores[1:].nanmean()

        return validation_loss, validation_score_fg, validation_scores

    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self._device)

        print(f'Resuming training at epoch {checkpoint["epoch"] + 1}.')

        # Load trainer specific attributes
        self._epoch = checkpoint["epoch"] + 1
        self._best_epoch = checkpoint["best_epoch"]
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._best_validation_loss = checkpoint["best_validation_loss"]

        # Load model specific attributes
        self._seg.load_checkpoint(checkpoint_path)

        del checkpoint

    def _save_checkpoint(self, checkpoint_path: str, epoch: int):

        self._seg.save_checkpoint(
            path=checkpoint_path,
            epoch=epoch,
            best_epoch=self._best_epoch,
            optimizer_state_dict=self._optimizer.state_dict(),
            best_validation_loss=self._best_validation_loss,
        )

    @property
    def last_checkpoint_name(self) -> str:
        return self._checkpoint_last

    @property
    def best_checkpoint_name(self) -> str:
        return self._checkpoint_best

    @property
    def last_checkpoint_path(self) -> str:
        return os.path.join(self._logdir, self._checkpoint_last)

    @property
    def best_checkpoint_path(self) -> str:
        return os.path.join(self._logdir, self._checkpoint_best)

    @property
    def best_epoch(self):
        return self._best_epoch

    @property
    def best_validation_loss(self):
        return self._best_validation_loss

    @property
    def loss_function(self):
        return self._loss_func

    @property
    def training_losses(self):
        return self._training_losses

    @property
    def validation_losses(self):
        return self._validation_losses

    @property
    def validation_scores(self):
        return self._validation_scores
