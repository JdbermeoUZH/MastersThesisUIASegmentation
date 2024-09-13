from dataclasses import dataclass
from typing import Optional

import torch
from tdigest import TDigest


@dataclass
class DomainStatistics:
    """
    TODO:
     - Add runnng calculation of quantiles with https://gist.github.com/davidbau/00a9b6763a260be8274f6ba22df9a145#file-runningstats-py-L753
    """
    mean: Optional[torch.Tensor | float] = 0.0
    std: Optional[torch.Tensor | float] = 0.0
    min: Optional[torch.Tensor | float] = torch.inf
    max: Optional[torch.Tensor | float] = -torch.inf
    quantile_cal: Optional[TDigest | dict] = None
    precalculated_quantiles: Optional[dict] = None
    momentum: float = 0.96
    frozen: bool = False
    update_quantiles: bool = False
    _step_num_px: int = 0
    _step_sum: float = 0.0
    _step_sum_sq: float = 0.0
    _step_min: float = torch.inf
    _step_max: float = -torch.inf
    
    def __post_init__(self):
        if isinstance(self.quantile_cal, dict):
            self.quantile_cal = TDigest.from_dict(self.quantile_cal)

        if self.update_quantiles and self.quantile_cal is None:
            self.quantile_cal = TDigest()
            
    def update_step_statistics(self, x: torch.Tensor) -> None:
        if self.frozen:
            raise ValueError('Statistics are frozen')
        self._step_num_px += x.numel()
        self._step_sum += x.sum().item()
        self._step_sum_sq += (x ** 2).sum().item()        
        self._step_min = min(x.min().item(), self._step_min) 
        self._step_max = max(x.max().item(), self._step_max)

        if self.update_quantiles:
            # Update the TDigest with the flattened tensor
            self.quantile_cal.batch_update(x.flatten().tolist(),
                                           w=1 - self.momentum)

    def update_statistics(self) -> None:
        if self.frozen:
            raise ValueError('Statistics are frozen')
        step_mean = self._step_sum / self._step_num_px
        step_std = (self._step_sum_sq / self._step_num_px - step_mean ** 2) ** 0.5
        
        self.mean = self.momentum * self.mean + (1 - self.momentum) * step_mean
        self.std = self.momentum * self.std + (1 - self.momentum) * step_std
        
        self.min = self.momentum * self.min + (1 - self.momentum) * self._step_min
        self.max = self.momentum * self.max + (1 - self.momentum) * self._step_max
        
        self._reset_step_statistics()
    
    def get_quantile(self, q: float) -> float:
        if q in self.precalculated_quantiles: 
            q = self.precalculated_quantiles[q]
        elif self.quantile_cal is not None:
            q = self.quantile_cal.percentile(q * 100)
        else:
            raise ValueError('Quantiles have not been calculated')
            
        return q
    
    def _reset_step_statistics(self):
        self._step_num_px = 0
        self._step_sum = 0.0
        self._step_sum_sq = 0.0
        self._step_min = torch.inf
        self._step_max = -torch.inf

    def update_statistics_from_volume(self, volume: torch.Tensor) -> None:
        # First and second moments, min and max
        self.update_step_statistics(volume)
        self.update_statistics()

    def reset_class(self):
        # Create an empty DomainStatistics object\
        #  with the attributes that should be preserved
        empty_stats = DomainStatistics(
            momentum = self.momentum, 
            frozen = self.frozen,
            update_quantiles = self.update_quantiles
        )

        # Iterate over the class dictionary and assign attributes
        for attr, value in vars(empty_stats).items():
            setattr(self, attr, value)
