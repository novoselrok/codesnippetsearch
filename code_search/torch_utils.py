from typing import Optional

import torch
import numpy as np


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def np_to_torch(arr: np.ndarray, device: Optional[torch.device] = None):
    tensor = torch.from_numpy(arr)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def torch_gpu_to_np(tensor: torch.Tensor):
    return tensor.cpu().numpy()

