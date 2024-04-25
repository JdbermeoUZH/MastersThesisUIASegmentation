
import torch
import numpy as np


def generate_invalid_image(image: torch.Tensor, pixel_weights: torch.Tensor, ) -> tuple[torch.Tensor, torch.Tensor]:
    