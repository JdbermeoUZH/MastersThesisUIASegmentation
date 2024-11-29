import torch


def flatten_pixels(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    elif len(x.shape) == 4:
        pass
    else:
        raise ValueError(f"Expected 3 or 4 dimensions, got {len(x.shape)}")
    
    b, num_f, h, w = x.shape
    
    return x.permute(0, 2, 3, 1).reshape(-1, num_f), (b, h, w)


def unflatten_pixels(x: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
    return x.reshape(b, h, w, -1).permute(0, 3, 1, 2)