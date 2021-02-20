import torch


def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)