import torch


def flatten_transform():
    def transform(input: torch.Tensor) -> torch.Tensor:
        return torch.flatten(input, start_dim=0, end_dim=-1)

    return transform


def scale_tanh_range():
    def transform(input: torch.Tensor) -> torch.Tensor:
        return input - 0.5

    return transform
