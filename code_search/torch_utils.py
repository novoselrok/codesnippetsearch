import torch
import numpy as np


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def np_to_torch(arr: np.ndarray, device: torch.device):
    return torch.from_numpy(arr).to(device)


def torch_gpu_to_np(tensor: torch.Tensor):
    return tensor.cpu().numpy()

