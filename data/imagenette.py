from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from .transforms import flatten_transform, scale_tanh_range


def load_imagenette_from_torch_dict(
    path: str = "imagenette2-160", 
    img_dims: tuple[int, int] = (128, 128)
) -> tuple[Dataset, Dataset]:
    path = Path(path).resolve()
    
    # the im
    train_data = torch.load(path / f"imagenette2-{img_dims[0]}/imagenette2-{img_dims[1]}-train.pt")
    train_data_dict = torch.load("imagenette2-160/imagenette2-128x128-train.pt")
    val_data_dict = torch.load("imagenette2-160/imagenette2-128x128-val.pt")

    train_data = TensorDataset(train_data_dict['images'], train_data_dict['labels'].long())
    val_data = TensorDataset(val_data_dict['images'], val_data_dict['labels'].long())

    img_shape = train_data[0][0].shape

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False
    )