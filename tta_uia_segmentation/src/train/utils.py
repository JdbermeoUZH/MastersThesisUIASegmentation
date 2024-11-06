import torch
import torch.optim as optim


class LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps: int | float, total_steps, last_epoch=-1):
        if 0.0 < warmup_steps < 1.0:
            warmup_steps = int(warmup_steps * total_steps)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
        else:
            warmup_factor = 1.0

        return [base_lr * warmup_factor for base_lr in self.base_lrs]
