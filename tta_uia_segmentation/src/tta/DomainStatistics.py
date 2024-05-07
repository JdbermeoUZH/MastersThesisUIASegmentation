from dataclasses import dataclass
from typing import Optional

import torch
from tdigest import TDigest


@dataclass
class DomainStatistics:
    mean: Optional[torch.Tensor | float] = 0.0
    std: Optional[torch.Tensor | float] = 0.0
    min: Optional[torch.Tensor | float] = torch.inf
    max: Optional[torch.Tensor | float] = -torch.inf
    quantile_cal: Optional[TDigest | dict] = None
    precalculated_quantiles: Optional[dict] = None
    momentum: float = 0.96
    _step_num_px: int = 0
    _step_sum: float = 0.0
    _step_sum_sq: float = 0.0
    _step_min: float = torch.inf
    _step_max: float = -torch.inf
    
    def __post_init__(self):
        if isinstance(self.quantile_cal, dict):
            self.quantile_cal = TDigest.from_dict(self.quantile_cal)
            
    def update_step_statistics(self, x: torch.Tensor) -> None:
        self._step_num_px += x.numel()
        self._step_sum += x.sum().item()
        self._step_sum_sq += (x ** 2).sum().item()
        
        x_min = x.min().item()
        if x_min < self._step_min:
            self._step_min = x_min
        
        x_max = x.max().item()
        if x_max > self._step_max:
            self._step_max = x_max
            
    def update_statistics(self) -> None:
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
            q = self.quantile_cal.quantile(q)
        else:
            raise ValueError('Quantiles have not been calculated')
            
        return q
    
    def _reset_step_statistics(self):
        self._step_max = -torch.inf
        self._step_min = torch.inf
        self._step_num_px = 0
        self._step_sum = 0.0
        self._step_sum_sq = 0.0
        
